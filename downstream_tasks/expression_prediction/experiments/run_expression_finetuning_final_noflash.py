# stdlib
import json
import logging
import os
import time
from functools import partial
from itertools import chain, compress
from pathlib import Path

# third-party
import torch
import numpy as np
import pandas as pd
import transformers
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

# accelerate
import accelerate
import random
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list


# local
from lm_experiments_tools import TrainerAccelerate as Trainer
from lm_experiments_tools import TrainerAccelerateArgs as TrainerArgs

from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration, get_git_diff, prepare_run
import lm_experiments_tools.optimizers as optimizers
from lm_experiments_tools import get_optimizer

from downstream_tasks.expression_prediction.expression_dataset import worker_init_fn
from downstream_tasks.expression_prediction.datasets.src.score_ct_specificity import score_predictions, mean_and_residuals_correlation
from downstream_tasks.expression_prediction.datasets.src.correlation_selected_cells import calculate_target_genes_metrics

def set_global_seed(seed: int):
    if seed is None:
        return
    print(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger(__name__)

if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--experiment_config', type=str, help='path to the experiment config')
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')
parser.add_argument('--save_predictions', action='store_true', help='save predictions to file')



def merge_default_params_with_dataset_config(dataset_config, default_params, logger):
    """
    Merge default parameters with dataset configuration.
    """
    def _merge_nested_dicts(target, source, path=""):
        for key, value in source.items():
            current_path = f"{path}.{key}" if path else key
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    _merge_nested_dicts(target[key], value, current_path)
                else:
                    logger.warning(
                        f"Parameter '{current_path}' is specified in both default params and dataset config. "
                        f"Using dataset config value: {target[key]}"
                    )
            else:
                if isinstance(value, dict):
                    target[key] = value.copy()
                else:
                    target[key] = value
                logger.info(f"Added default parameter '{current_path}': {value}")

    merged_config = OmegaConf.create(OmegaConf.to_container(dataset_config, resolve=True))
    default_params_dict = OmegaConf.to_container(default_params, resolve=True) if hasattr(default_params, 'items') else default_params
    _merge_nested_dicts(merged_config, default_params_dict)
    return merged_config

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
from accelerate.utils import broadcast_object_list


def _collect_dataset_configs(experiment_config, prefix: str) -> List[Any]:
    """Берёт все ключи вида train_dataset*, valid_dataset*."""
    return [v for k, v in experiment_config.items() if str(k).startswith(prefix)]


def _target_class_name(cfg: Any) -> str:
    """Hydra _target_ -> последний сегмент имени класса."""
    try:
        t = cfg.get("_target_", "")
    except Exception:
        t = ""
    t = str(t)
    return t.split(".")[-1] if t else ""


def _is_expression_dataset_cfg(cfg: Any) -> bool:
    """True только для ExpressionDataset (Mode2 не подходит)."""
    return _target_class_name(cfg) == "ExpressionDataset"


def infer_global_n_keys_from_expression_datasets(
    train_dataset_cfgs: List[Any],
    shared_dataset_params: Optional[Any],
    merge_fn,
    accelerator,
    alogger,
) -> int:
    """
    Считает n_keys только по train-конфигам ExpressionDataset.
    Делает это дешево: читает targets_path CSV и берёт число уникальных id.
    """
    if len(train_dataset_cfgs) == 0:
        raise ValueError("No train_dataset configs found")

    expr_cfgs = [cfg for cfg in train_dataset_cfgs if _is_expression_dataset_cfg(cfg)]
    if len(expr_cfgs) == 0:
        raise ValueError(
            "Cannot infer n_keys: среди train_dataset* нет ExpressionDataset. "
            "Добавь хотя бы один ExpressionDataset в train."
        )

    if accelerator.is_main_process:
        tmp_cfgs = [cfg.copy() for cfg in expr_cfgs]
        if shared_dataset_params is not None:
            for i, cfg in enumerate(tmp_cfgs):
                tmp_cfgs[i] = merge_fn(cfg, shared_dataset_params, alogger)

        counts = []
        for cfg in tmp_cfgs:
            cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
            targets_path = cfg_resolved.get("targets_path", None)
            if targets_path is None:
                raise ValueError("targets_path not found in dataset config (ExpressionDataset)")

            df = pd.read_csv(targets_path)
            if "id" not in df.columns:
                raise ValueError(f"'id' column not found in targets_path={targets_path}")

            counts.append(int(df["id"].nunique()))

        n_keys = min(counts)
        alogger.info(f"[n_keys] inferred from ExpressionDataset only: min({counts}) = {n_keys}")
    else:
        n_keys = -1

    obj = [n_keys]
    broadcast_object_list(obj)
    n_keys = int(obj[0])
    if n_keys <= 0:
        raise RuntimeError(f"Broadcast n_keys failed: {n_keys}")
    return n_keys


def apply_n_keys_to_all_dataset_cfgs(
    dataset_cfgs: List[Any],
    shared_dataset_params: Optional[Any],
    merge_fn,
    n_keys: int,
    alogger,
) -> List[Any]:
    """
    Мёрджит shared params (если есть), затем принудительно выставляет n_keys всем датасетам.
    Возвращает новый список cfg.
    """
    out = [cfg.copy() for cfg in dataset_cfgs]
    if shared_dataset_params is not None:
        for i, cfg in enumerate(out):
            out[i] = merge_fn(cfg, shared_dataset_params, alogger)

    for cfg in out:
        OmegaConf.update(cfg, "n_keys", int(n_keys), force_add=True)

    return out


def build_dataset_from_cfgs(dataset_cfgs: List[Any]) -> Tuple[Dataset, List[Dataset]]:
    """Instantiate -> Dataset or ConcatDataset."""
    datasets = [instantiate(cfg) for cfg in dataset_cfgs]
    if len(datasets) == 0:
        raise ValueError("No datasets after instantiate()")
    if len(datasets) == 1:
        return datasets[0], datasets
    return ConcatDataset(datasets), datasets


def build_loader(
    dataset: Dataset,
    accelerator,
    batch_size: int,
    seed: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    collate_fn,
    worker_init_fn,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    sampler = DistributedSampler(
        dataset,
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )
    return loader, sampler

def main():
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    #  Accelerate
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )
    alogger = get_logger(__name__) 
   # alogger = get_logger('')  # accelerate logger (привязан к процессу)
    alogger.info(f'num processes: {accelerator.num_processes}')
    alogger.info(f'mixed precision: {accelerator.mixed_precision}')
    alogger.info(f'accelerator state: {accelerator.state}')

    prepare_run(args, alogger, logger_fmt, accelerator=accelerator)

    if accelerator.is_main_process:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
    else:
        timestamp = None
    obj = [timestamp]
    broadcast_object_list(obj)
    timestamp = obj[0]

    experiment_config_path = Path(args.experiment_config).expanduser().absolute()

    with initialize_config_dir(str(experiment_config_path.parents[0])):
        experiment_config = compose(config_name=experiment_config_path.name)

    if "args_params" in experiment_config:
        trainer_kwargs = instantiate(experiment_config["args_params"])
        for k, v in trainer_kwargs.items():
            if hasattr(args, k):
                alogger.warning(f"Setting attr {k}:{v} (overwritten by cfg)")
            else:
                alogger.info(f"Setting attr {k}:{v}")
            args.__setattr__(k, v)

    set_global_seed(args.seed)

    if args.resume is not None:
        alogger.warning(
            f"Resuming from cpt {args.resume}. This will overwrite reset_lr, reset_optimizer, reset_iteration and init_cpt options"
        )
        args.__setattr__("reset_lr", False)
        args.__setattr__("reset_optimizer", False)
        args.__setattr__("reset_iteration", False)
        args.__setattr__("init_checkpoint", args.model_path + "/" + args.resume)
        args.__setattr__("model_path", args.model_path + "/resume_" + args.resume + "/")

    if accelerator.is_main_process:
        if args.model_path is None:
            raise ValueError("Model path should not be None")

    args.model_path = os.path.join(args.model_path, timestamp)
    alogger.info(f"rank: {accelerator.process_index}, Model path: {args.model_path}")

    if accelerator.is_main_process and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        json.dump(args_dict, open(model_path / 'config.json', 'w'), indent=4)
        open(model_path / 'git.diff', 'w').write(get_git_diff())
        # Сохраняем копию Hydra-конфига
        content = "\n".join(open(experiment_config_path).readlines())
        with open(Path(args.model_path) / "experiment_config.yaml", "w") as fout:
            fout.write(content)

    accelerator.wait_for_everyone()

    # Padding
    tokenizer = AutoTokenizer.from_pretrained(args.gen_tokenizer, trust_remote_code=True)
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer, trust_remote_code=True)

    def _pad_1d(x: torch.Tensor, length: int, pad_value: int, pad_left: bool = False) -> torch.Tensor:
        pad_len = length - x.size(0)
        if pad_len <= 0:
            return x
        pad = x.new_full((pad_len,), pad_value)
        return torch.cat([pad, x], dim=0) if pad_left else torch.cat([x, pad], dim=0)


    def _pad_2d(x: torch.Tensor, max_len: int, pad_value, dim: int = 0) -> torch.Tensor:
        pad_len = max_len - x.size(dim)
        if pad_len <= 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        pad = x.new_full(tuple(pad_shape), pad_value)
        return torch.cat([x, pad], dim=dim)


    def _pad_3d(x: torch.Tensor, max_len: int, pad_value, dim: int = 1) -> torch.Tensor:
        pad_len = max_len - x.size(dim)
        if pad_len <= 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        pad = x.new_full(tuple(pad_shape), pad_value)
        return torch.cat([x, pad], dim=dim)

    def collate_fn(batch):
        pad_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'labels_mask']
        no_pad_keys = ['tpm', 'dataset_flag']
        special_keys = ['gene_id', 'selected_keys', 'dataset_description', 'name', 'chrom', 'reverse', 'start', 'end']

        pad_token_ids = {
            'input_ids': tokenizer.pad_token_id,
            'attention_mask': 0,
            'token_type_ids': 0,
            'labels': 0.0,
            'labels_mask': 0,
            'desc_input_ids': text_tokenizer.pad_token_id,
            'desc_attention_mask': 0,
        }

        max_seq_len = max(sample['input_ids'].size(1) for sample in batch)

        n_keys = len(batch[0]['desc_input_ids'])
        max_text_seq_len = 0
        for sample in batch:
            for ids in sample['desc_input_ids']:
                max_text_seq_len = max(max_text_seq_len, ids.size(0))
        if max_text_seq_len == 0:
            max_text_seq_len = 1

        batch_dict = {key: [] for key in pad_keys + no_pad_keys + special_keys}

        desc_ids_batch = []
        desc_mask_batch = []
        for sample in batch:
            sample_ids = []
            sample_masks = []
            for k in range(n_keys):
                ids = sample['desc_input_ids'][k]       # (L_i,)
                mask = sample['desc_attention_mask'][k] # (L_i,)

                ids  = _pad_1d(ids,  max_text_seq_len, pad_token_ids['desc_input_ids'], pad_left=True)
                mask = _pad_1d(mask, max_text_seq_len, pad_token_ids['desc_attention_mask'], pad_left=True)

                sample_ids.append(ids)
                sample_masks.append(mask)

            desc_ids_batch.append(torch.stack(sample_ids, dim=0))   # (n_keys, L_text_max)
            desc_mask_batch.append(torch.stack(sample_masks, dim=0))# (n_keys, L_text_max)

        for sample in batch:
            for key in pad_keys:
                x = sample[key]
                if key in ['input_ids', 'attention_mask', 'token_type_ids']:  # (n_keys, L)
                    x = _pad_2d(x, max_seq_len, pad_token_ids[key], dim=1)    # pad по L -> dim=1
                if key in ['labels', 'labels_mask']:                                                       
                    x = _pad_3d(x, max_seq_len, pad_token_ids[key], dim=1)
                batch_dict[key].append(x)

            for key in no_pad_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])

            for key in special_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])

        # stack
        for key in pad_keys:
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)  # (B, n_keys, Lmax) или (B, n_keys, Lmax, 1)

        for key in no_pad_keys:
            if len(batch_dict[key]) > 0:
                batch_dict[key] = torch.stack(batch_dict[key], dim=0)

        batch_dict['desc_input_ids'] = torch.stack(desc_ids_batch, dim=0)        # (B, n_keys, L_text_max)
        batch_dict['desc_attention_mask'] = torch.stack(desc_mask_batch, dim=0)  # (B, n_keys, L_text_max)

        return batch_dict


    # Data
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs_workers = args.data_n_workers

    train_cfgs = _collect_dataset_configs(experiment_config, "train_dataset")
    valid_cfgs = _collect_dataset_configs(experiment_config, "valid_dataset")
    shared_dataset_params = experiment_config.get("shared_dataset_params", None)

    if len(train_cfgs) == 0:
        raise ValueError("No training datasets found (no train_dataset* in config)")

    # 1) считаем n_keys только по ExpressionDataset из train
    n_keys_global_train = infer_global_n_keys_from_expression_datasets(
        train_dataset_cfgs=train_cfgs,
        shared_dataset_params=shared_dataset_params,
        merge_fn=merge_default_params_with_dataset_config,
        accelerator=accelerator,
        alogger=alogger,
    )

    n_keys_global_valid = infer_global_n_keys_from_expression_datasets(
        train_dataset_cfgs=valid_cfgs,
        shared_dataset_params=shared_dataset_params,
        merge_fn=merge_default_params_with_dataset_config,
        accelerator=accelerator,
        alogger=alogger,
    )

    # 2) выставляем n_keys 
    train_cfgs = apply_n_keys_to_all_dataset_cfgs(
        dataset_cfgs=train_cfgs,
        shared_dataset_params=shared_dataset_params,
        merge_fn=merge_default_params_with_dataset_config,
        n_keys=n_keys_global_train,
        alogger=alogger,
    )

    valid_cfgs = apply_n_keys_to_all_dataset_cfgs(
        dataset_cfgs=valid_cfgs,
        shared_dataset_params=shared_dataset_params,
        merge_fn=merge_default_params_with_dataset_config,
        n_keys=n_keys_global_valid ,
        alogger=alogger,
    )

    # 3) instantiate train
    train_dataset, train_datasets_list = build_dataset_from_cfgs(train_cfgs)
    if accelerator.is_main_process:
        for i, ds in enumerate(train_datasets_list):
            alogger.info(f"train dataset {i}: {ds.describe() if hasattr(ds,'describe') else type(ds)}")
        alogger.info(f"total len(train_dataset)={len(train_dataset)} | n_keys_global={n_keys_global_train}")

    train_dataloader, train_sampler = build_loader(
        dataset=train_dataset,
        accelerator=accelerator,
        batch_size=per_worker_batch_size,
        seed=args.seed,
        shuffle=True,
        drop_last=False,
        num_workers=kwargs_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,   
    )

    if len(valid_cfgs) > 0:
        valid_dataset, valid_datasets_list = build_dataset_from_cfgs(valid_cfgs)
        if accelerator.is_main_process:
            for i, ds in enumerate(valid_datasets_list):
                alogger.info(f"valid dataset {i}: {ds.describe() if hasattr(ds,'describe') else type(ds)}")
            alogger.info(f"total len(valid_dataset)={len(valid_dataset)} | n_keys_global={n_keys_global_valid}")

        valid_dataloader, valid_sampler = build_loader(
            dataset=valid_dataset,
            accelerator=accelerator,
            batch_size=per_worker_batch_size,
            seed=args.seed,
            shuffle=False,
            drop_last=False,
            num_workers=kwargs_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        )

        if args.valid_interval is None:
            args.valid_interval = args.log_interval
    else:
        valid_dataloader = None
        valid_sampler = None
        if accelerator.is_main_process:
            alogger.info("No validation data is used.")


    # Model 

    if "model_kwargs" in experiment_config:
        model_kwargs = instantiate(experiment_config["model_kwargs"])
    else:
        model_kwargs = {}

    model_cls = get_cls_by_name(args.model_cls)
    if accelerator.is_main_process:
        alogger.info(f'Using model class: {model_cls}')
    model = model_cls(**model_kwargs)
    model_activation_fn = model.activation

    # Optimizer 
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    if accelerator.is_main_process:
        alogger.info(f'Using optimizer class: {optimizer_cls}')

    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        optimizer = optimizer_cls(
            model.parameters(), lr=args.lr,
            scale_parameter=args.scale_parameter,
            relative_step=args.relative_step,
            warmup_init=args.warmup_init,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Batch transform
    def batch_transform_fn(batch):
        result = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels_mask': batch['labels_mask'],
            'dataset_flag': batch['dataset_flag'],
            'labels': batch['labels'],
            'gene_id': batch['gene_id'],
            'selected_keys': batch['selected_keys'],
            'dataset_description': batch['dataset_description'],
            'desc_input_ids': batch['desc_input_ids'], 
            'desc_attention_mask': batch['desc_attention_mask']
        }
        return result
    
    # Metrics
    def keep_for_metrics_fn(batch, output):
        logits = output["logits"].detach().cpu()
        labels = output["labels_reshaped"].detach().cpu()
        masks  = output["labels_mask_reshaped"].detach().cpu()

        y_true = labels[:, 0, 0]      
        y_pred = logits[:, 0, 0]     
        mask   = masks[:, 0, 0] > 0   

        y_true = y_true[mask]
        y_pred = y_pred[mask]

        preds = y_pred.unsqueeze(1)   
        target = y_true.unsqueeze(1) 
        reduce_dims = (0, 1)

        data = {}

        loss_components = ['cls_loss', 'other_loss']
        for loss_component in loss_components:
            if f'loss_{loss_component}' in output and output[f'loss_{loss_component}'] is not None:
                data[f'loss_{loss_component}'] = output[f'loss_{loss_component}'].detach().cpu()

        dataset_description = list(chain.from_iterable(batch['dataset_description']))
        mask_idx = mask.nonzero(as_tuple=True)[0].tolist()
        data['tpm_true'] = y_true.tolist()
        data['tpm_preds'] = y_pred.tolist()
        data['gene_id'] =  list(chain.from_iterable(batch['gene_id']))
        data['keys_id'] = list(chain.from_iterable(batch['selected_keys']))
        data['dataset_description'] = [dataset_description[i] for i in mask_idx]

        return data

    def make_metrics_fn(model_path, save_predictions=False):
        def metrics_fn(data):
            metrics = {}

            loss_components = ['cls_loss', 'other_loss']
            for loss_component in loss_components:
                if f'loss_{loss_component}' in data and data[f'loss_{loss_component}'] is not None:
                    metrics[f'loss_{loss_component}'] = torch.mean(data[f'loss_{loss_component}']).item()

            tpm_true = data['tpm_true']
            tpm_preds = data['tpm_preds']
            gene_id = data['gene_id']
            keys_id = data['keys_id']
            dataset_description = data['dataset_description']

            assert len(tpm_true) == len(tpm_preds) == len(gene_id) == len(keys_id) == len(dataset_description), \
                f"Mismatch! tpm_true: {len(tpm_true)}, tpm_preds: {len(tpm_preds)}, gene_id: {len(gene_id)}, keys_id {len(keys_id)}, dataset_description {len(dataset_description)}"

            df = pd.DataFrame({
                'gene_id': gene_id,
                'cell_type': keys_id,
                'tpm_true': tpm_true,
                'tpm_pred': tpm_preds,
                'dataset_description': dataset_description,
            })
            if save_predictions and accelerator.is_main_process:
                df.to_csv(os.path.join(model_path, "labels.csv"))

            for dataset_desc in df['dataset_description'].unique():
                df_dataset = df[df['dataset_description'] == dataset_desc]

                df_pred = df_dataset.pivot_table(
                    index='gene_id',
                    columns='cell_type',
                    values='tpm_pred',
                    aggfunc='first'
                )
                df_true = df_dataset.pivot_table(
                    index='gene_id',
                    columns='cell_type',
                    values='tpm_true',
                    aggfunc='first'
                )
                if save_predictions and accelerator.is_main_process:
                    df_true.to_csv(os.path.join(model_path, f"{dataset_desc}_true.csv"))
                    df_pred.to_csv(os.path.join(model_path, f"{dataset_desc}_pred.csv"))

                if df_true.empty or df_pred.empty:
                    continue

                if not df_true.empty and not df_pred.empty:
                    gene_correlations = []
                    for gene in df_true.index:
                        gene_true = df_true.loc[gene]
                        gene_pred = df_pred.loc[gene]
                        mask = pd.notna(gene_true) & pd.notna(gene_pred)
                        gene_true = gene_true[mask]
                        gene_pred = gene_pred[mask]

                        # if len(gene_pred) > 1 and np.std(gene_pred) == 0:
                        #     alogger.error(f"dataset {dataset_desc} gene {gene} has all predicted values the same")
                        #     raise ValueError(f"All predicted values for {gene} are the same. Are you missing cell type descriptions?")

                        if len(gene_true) > 3 and np.std(gene_true) > 0:
                            try:
                                corr = np.corrcoef(gene_true, gene_pred)[0, 1]
                                if not np.isnan(corr):
                                    gene_correlations.append(corr)
                            except Exception:
                                continue

                    cell_correlations = []
                    for cell_type in df_true.columns:
                        cell_true = df_true[cell_type]
                        cell_pred = df_pred[cell_type]
                        if np.std(cell_pred.values) == 0 and len(cell_pred) > 1:
                            raise ValueError(f"All predicted values for {cell_type} are the same")

                        cell_true = cell_true[pd.notna(cell_true)]
                        cell_pred = cell_pred[pd.notna(cell_pred)]

                        if len(cell_true) > 3 and np.std(cell_true) != 0:
                            try:
                                corr = np.corrcoef(cell_true, cell_pred)[0, 1]
                                if not np.isnan(corr):
                                    cell_correlations.append(corr)
                            except Exception:
                                continue

                    if gene_correlations:
                        metrics[f'pearson_corr_cells_{dataset_desc}'] = float(np.mean(gene_correlations))
                    if cell_correlations:
                        metrics[f'pearson_corr_genes_{dataset_desc}'] = float(np.mean(cell_correlations))

                    ALLOWED = {
                    "Expression_dataset_v1_GRCh38_csv dataset",
                    "Expression_dataset_v1_mm10_CPM dataset",
                    }

                    if dataset_desc in ALLOWED:
                        # клетоспецифичность
                        df_true = df_true.reset_index()
                        df_pred = df_pred.reset_index()
                        score = score_predictions(
                            df_true,
                            df_pred,
                            experiment_config.selected_targets_path,
                            need_log=False,
                            logger=alogger
                        )
                        if score and score.get('deviation_r', None):
                            metrics[f'score_predictions_{dataset_desc}'] = score['deviation_r']
                        score2 = mean_and_residuals_correlation(df_true,
                            df_pred)
                        if isinstance(score2, dict):
                            for k, v in score2.items():
                                if isinstance(v, (np.floating, np.integer)):
                                    v = v.item()
                                metrics[f"mean_residual_{k}_{dataset_desc}"] = v
                    

            return metrics

        return metrics_fn

    metrics_fn = make_metrics_fn(args.model_path, save_predictions=args.save_predictions)

    model, optimizer = accelerator.prepare(model, optimizer)

    # Trainer
    trainer = Trainer(
        args, accelerator, model, optimizer, train_dataloader,
        valid_dataloader=valid_dataloader,
        train_sampler=train_sampler,
        batch_transform_fn=batch_transform_fn,
        keep_for_metrics_fn=keep_for_metrics_fn,
        metrics_fn=metrics_fn
    )

    # Training
    accelerator.wait_for_everyone()
    trainer.train()
    accelerator.wait_for_everyone()

    # Post-training / validation / save
    if args.save_best:
        best_model_path = str(Path(args.model_path) / 'model_best.pth')
        if accelerator.is_main_process:
            alogger.info(f'Loading best saved model from {best_model_path}')
        trainer.load(best_model_path)

    if valid_dataloader is not None:
        if accelerator.is_main_process:
            alogger.info('Runnning validation on valid data:')
        trainer.validate(valid_dataloader, write_tb=False)

    if accelerator.is_main_process:
        trainer.save_metrics(args.model_path)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.exception(e)