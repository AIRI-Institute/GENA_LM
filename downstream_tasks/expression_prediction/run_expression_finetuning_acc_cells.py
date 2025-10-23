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
import torch.nn.functional as F
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
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list


# local
from lm_experiments_tools import TrainerAccelerate as Trainer
from lm_experiments_tools import TrainerAccelerateArgs as TrainerArgs

from lm_experiments_tools.utils import get_cls_by_name, collect_run_configuration, get_git_diff, prepare_run
import lm_experiments_tools.optimizers as optimizers
from lm_experiments_tools import get_optimizer

from downstream_tasks.expression_prediction.expression_dataset import ExpressionDataset, worker_init_fn
from downstream_tasks.expression_prediction.datasets.src.score_ct_specificity import score_predictions
from downstream_tasks.expression_prediction.datasets.src.correlation_selected_cells import calculate_target_genes_metrics


logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
# accelerate logger (оборачивает python logging)
logger = logging.getLogger(__name__)

# Настраиваем видимые GPU, если переменная не установлена
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--experiment_config', type=str, help='path to the experiment config')
parser.add_argument('--save_predictions', type=bool, default=False, help='save predictions to file')
parser.add_argument('--log_level', type=int, default=logging.INFO, help='log level')


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


def main():
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    # --- Инициализация Accelerate ---
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

    # --- Глобальный timestamp синхронизируем между всеми процессами ---
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

    # Предупреждение о resume: оставляем логику прежней
    if args.resume is not None:
        alogger.warning(
            f"Resuming from cpt {args.resume}. This will overwrite reset_lr, reset_optimizer, reset_iteration and init_cpt options"
        )
        args.__setattr__("reset_lr", False)
        args.__setattr__("reset_optimizer", False)
        args.__setattr__("reset_iteration", False)
        args.__setattr__("init_checkpoint", args.model_path + "/" + args.resume)
        args.__setattr__("model_path", args.model_path + "/resume_" + args.resume + "/")

    # Проверка и подготовка каталога эксперимента — только на главном процессе
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

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.gen_tokenizer)

    def collate_fn(batch):
        # Тензоры, которые нужно выровнять по длине L
        pad_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'labels_mask']
        no_pad_keys = ['desc_vectors', 'tpm', 'dataset_flag', 'dataset_deviation', 'dataset_mean']
        special_keys = ['gene_id', 'selected_keys', 'dataset_description', 'name', 'chrom', 'reverse', 'start', 'end']

        pad_token_ids = {
            'input_ids': tokenizer.pad_token_id,
            'token_type_ids': 0,
            'attention_mask': 0,
            'labels': 0.0,
            'labels_mask': 0,
        }

        # 1) найдём максимальную длину L по последней оси среди ВСЕХ ключей, которые паддим
        max_L = 0
        for sample in batch:
            for key in pad_keys:
                t = sample[key]
                max_L = max(max_L, t.shape[-2] if t.dim() == 3 else t.shape[-1])

        def pad_last_dim(x: torch.Tensor, L_target: int, fill):
            """Паддинг по оси L: для
            - (L,)     -> (L_target,)
            - (N, L)   -> (N, L_target)
            - (N, L, C)-> (N, L_target, C)
            """
            if x.dim() == 1:
                pad = (0, L_target - x.shape[0])
            elif x.dim() == 2:
                pad = (0, L_target - x.shape[1], 0, 0)
            elif x.dim() == 3:
                # порядок для F.pad: (W_l, W_r, H_l, H_r, D_l, D_r)
                # здесь C=W, L=H, N=D
                pad = (0, 0, 0, L_target - x.shape[1], 0, 0)
            else:
                raise ValueError(f"Unexpected tensor dim {x.dim()} for shape {tuple(x.shape)}")
            return F.pad(x, pad, value=fill) if (L_target - (x.shape[-2] if x.dim()==3 else x.shape[-1])) > 0 else x

        batch_out = {k: [] for k in pad_keys + no_pad_keys + special_keys}

        # 2) паддим по ключам
        for sample in batch:
            for key in pad_keys:
                t = sample[key]
                fill = pad_token_ids[key]
                if key in ('labels_mask',):
                    # ensure dtype: bool
                    t = t.bool()
                    # F.pad не поддерживает bool — паддим как float и приводим тип
                    t = pad_last_dim(t.float(), max_L, float(fill)).bool()
                elif key in ('attention_mask', 'token_type_ids',):
                    t = pad_last_dim(t.long(), max_L, int(fill))
                elif key in ('input_ids',):
                    t = pad_last_dim(t.long(), max_L, int(fill))
                else:  # labels
                    t = pad_last_dim(t.float(), max_L, float(fill))
                batch_out[key].append(t)

            for key in no_pad_keys:
                batch_out[key].append(sample[key])

            for key in special_keys:
                batch_out[key].append(sample[key])

        # 3) stack для тензоров; списки оставляем
        for key in pad_keys + no_pad_keys:
            batch_out[key] = torch.stack(batch_out[key], dim=0)

        return batch_out


    # --- Data ---
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers, 'collate_fn': collate_fn}

    if accelerator.is_main_process:
        alogger.info(f'preparing training data')

    # Собираем train dataset конфиги
    train_datasets_configs = [v for k, v in experiment_config.items() if k.startswith('train_dataset')]
    shared_dataset_params = experiment_config.get('shared_dataset_params', None)

    # Rank 0-only init для вычисления min_train_chunk_size
    if accelerator.is_main_process:
        tmp_configs = [config.copy() for config in train_datasets_configs]
        if shared_dataset_params is not None:
            for i, config in enumerate(tmp_configs):
                alogger.info(f"Merging default parameters with train_dataset_{i}")
                tmp_configs[i] = merge_default_params_with_dataset_config(config, shared_dataset_params, alogger)
        for config in tmp_configs:
            OmegaConf.update(config, "loglevel", logging.ERROR)
        train_datasets_tmp = [instantiate(config) for config in tmp_configs]
        min_train_chunk_size = min([dataset.get_num_keys() for dataset in train_datasets_tmp])
        alogger.info(f"Chunk size (a.k.a. n_cells) for all datasets will be set to: {min_train_chunk_size}")
    else:
        min_train_chunk_size = -1

    # Рассылаем min_train_chunk_size
    obj = [min_train_chunk_size]
    broadcast_object_list(obj)
    min_train_chunk_size = obj[0]
    assert min_train_chunk_size > 0, f"min_train_chunk_size {min_train_chunk_size} >= 0: possible broadcast issue"

    # Re-init конфиги на всех рангах с учётом shared params и n_keys
    if shared_dataset_params is not None:
        for i, config in enumerate(train_datasets_configs):
            train_datasets_configs[i] = merge_default_params_with_dataset_config(config, shared_dataset_params, alogger)

    for config in train_datasets_configs:
        if "n_keys" in config and config["n_keys"] != min_train_chunk_size:
            raise ValueError(f"n_keys in config is different from min_train_chunk_size: {config['n_keys']} != {min_train_chunk_size}")
        OmegaConf.update(config, "n_keys", min_train_chunk_size, force_add=True)

    train_datasets = [instantiate(config) for config in train_datasets_configs]
    if len(train_datasets) == 0:
        raise ValueError("No training datasets found")
    elif len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)

    if accelerator.is_main_process:
        for i, dataset in enumerate(train_datasets):
            alogger.info(f'dataset {i}: {dataset.describe()}')
        alogger.info(f'total len(train_dataset): {len(train_dataset)}')

    # Самплеры и лоадеры
    train_sampler = DistributedSampler(
        train_dataset,
        rank=accelerator.process_index,
        num_replicas=accelerator.num_processes,
        shuffle=True,
        drop_last=False,
        seed=args.seed
    )
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size,
                                  sampler=train_sampler, worker_init_fn=worker_init_fn, **kwargs)

    # --- Validation datasets (если заданы) ---
    valid_datasets_configs = [v for k, v in experiment_config.items() if k.startswith('valid_dataset')]
    if len(valid_datasets_configs) > 0:
        if accelerator.is_main_process:
            alogger.info(f'preparing validation data')
            tmp_configs = [config.copy() for config in valid_datasets_configs]
            if shared_dataset_params is not None:
                for i, config in enumerate(tmp_configs):
                    alogger.info(f"Merging default parameters with valid_dataset_{i}")
                    tmp_configs[i] = merge_default_params_with_dataset_config(config, shared_dataset_params, alogger)
            for config in tmp_configs:
                OmegaConf.update(config, "loglevel", logging.ERROR)
            valid_datasets_tmp = [instantiate(config) for config in tmp_configs]
            min_valid_chunk_size = min([dataset.get_num_keys() for dataset in valid_datasets_tmp])
            if min_valid_chunk_size != min_train_chunk_size:
                alogger.warning(
                    f"\n -!!!!- min_valid_chunk_size != min_train_chunk_size ({min_valid_chunk_size} != {min_train_chunk_size}), "
                    f"Min number of keys in validation datasets is not the same as in train datasets. "
                    f"This probably means that the validation contain not the same cells as in train and could lead to incorrect results\n -!!!!-"
                )
        else:
            min_valid_chunk_size = -1

        obj = [min_valid_chunk_size]
        broadcast_object_list(obj)
        min_valid_chunk_size = obj[0]
        assert min_valid_chunk_size > 0, f"min_valid_chunk_size {min_valid_chunk_size} >= 0: possible broadcast issue"

        if shared_dataset_params is not None:
            for i, config in enumerate(valid_datasets_configs):
                valid_datasets_configs[i] = merge_default_params_with_dataset_config(config, shared_dataset_params, alogger)

        for config in valid_datasets_configs:
            if "n_keys" in config and config["n_keys"] != min_valid_chunk_size:
                raise ValueError(f"n_keys in config is different from min_valid_chunk_size: {config['n_keys']} != {min_valid_chunk_size}")
            OmegaConf.update(config, "n_keys", min_valid_chunk_size, force_add=True)

        valid_datasets = [instantiate(config) for config in valid_datasets_configs]
        if len(valid_datasets) == 1:
            valid_dataset = valid_datasets[0]
        else:
            valid_dataset = ConcatDataset(valid_datasets)

        for i, dataset in enumerate(valid_datasets):
            alogger.info(f'dataset {i}: {dataset.describe()}')
        alogger.info(f'total len(valid_dataset): {len(valid_dataset)}')

        valid_sampler = DistributedSampler(
            valid_dataset,
            rank=accelerator.process_index,
            num_replicas=accelerator.num_processes,
            shuffle=False
        )
        valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size,
                                      sampler=valid_sampler, worker_init_fn=worker_init_fn, **kwargs)

        if args.valid_interval is None:
            args.valid_interval = args.log_interval
    else:
        valid_dataloader = None
        if accelerator.is_main_process:
            alogger.info('No validation data is used.')

    # --- Model ---
    model_cfg = AutoConfig.from_pretrained(args.model_cfg)

    if "model_kwargs" in experiment_config:
        model_kwargs = instantiate(experiment_config["model_kwargs"])
    else:
        model_kwargs = {}

    if "config" in model_kwargs:
        for k, v in model_kwargs["config"].items():
            model_cfg.__setattr__(k, v)

    model_kwargs["config"] = model_cfg

    model_cls = get_cls_by_name(args.backbone_cls)
    if accelerator.is_main_process:
        alogger.info(f'Using backbone model class: {model_cls}')
    model = model_cls(**model_kwargs)
    model_activation_fn = model.activation

    # --- RMT wrapper (как было) ---
    if args.num_mem_tokens is not None:
        rmt_config = {
            'num_mem_tokens': args.num_mem_tokens,
            'max_n_segments': args.max_n_segments,
            'input_size': args.input_size,
            'bptt_depth': args.bptt_depth,
            'sum_loss': True,
            'tokenizer': tokenizer,
            'mixed_length_ratio': args.mixed_length_ratio,
        }
        rmt_cls = get_cls_by_name(args.model_cls)
        if accelerator.is_main_process:
            alogger.info(f'Wrapping in: {rmt_cls}')
        model = rmt_cls(model, **rmt_config)

            #загрузка чекпоинта horovod
        if hasattr(args, "init_checkpoint_pth") and args.init_checkpoint_pth is not None and os.path.isfile(args.init_checkpoint_pth):
            if accelerator.is_main_process:
                alogger.info(f"Loading checkpoint from {args.init_checkpoint_pth}")
            checkpoint = torch.load(args.init_checkpoint_pth, map_location="cpu")
    
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint
    
            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if accelerator.is_main_process:
                alogger.info(f"Checkpoint loaded. Missing keys: {missing}, Unexpected keys: {unexpected}")

        if args.input_seq_len / model.segment_size > rmt_config['max_n_segments']:
            raise RuntimeError(
                f"Input sequence does not fully fit into selected number of segments: "
                f"{args.input_seq_len} / {model.segment_size} > {rmt_config['max_n_segments']}"
            )

    if not args.backbone_trainable:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                if accelerator.is_main_process:
                    alogger.info(f'{name} is frozen')
                param.requires_grad = False

    # --- Optimizer ---
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

    # --- Batch transform / metrics (без изменений по логике) ---
    def batch_transform_fn(batch):
        result = {
            'input_ids': batch['input_ids'],
            'token_type_ids': batch['token_type_ids'],
            'attention_mask': batch['attention_mask'],
            'labels_mask': batch['labels_mask'],
            'desc_vectors': batch['desc_vectors'],
            'labels': batch['labels'],
            'tpm': batch['tpm'],
            'gene_id': batch['gene_id'],
            'selected_keys': batch['selected_keys'],
            'dataset_description': batch['dataset_description'],
            'dataset_flag': batch['dataset_flag']
        }
        return result

    def keep_for_metrics_fn(batch, output):
        predictions_segm = [[el.detach().cpu() for el in s] for s in output['logits_segm']]
        labels_segm = [[el.detach().cpu() for el in s] for s in output['labels_reshaped']]
        rmt_labels_masks_segm = [[el.detach().cpu().to(torch.bool) for el in s] for s in output['labels_mask_reshaped']]

        y_rmt, p_rmt = [], []

        labels = torch.stack(labels_segm[-1])
        preds = torch.stack(predictions_segm[-1])
        masks = torch.stack(rmt_labels_masks_segm[-1])

        y_segm = labels[:, 0, :].squeeze(-1)
        p_segm = preds[:, 0, :].squeeze(-1)
        mask = masks[:, 0, :].squeeze(-1)

        y_segm = y_segm[mask]
        p_segm = p_segm[mask]

        assert y_segm.shape == p_segm.shape
        y_rmt += [y_segm]
        p_rmt += [p_segm]

        if not y_rmt or not p_rmt:
            return {}

        y_rmt = torch.cat(y_rmt)
        p_rmt = torch.cat(p_rmt)
        assert y_rmt.shape == p_rmt.shape

        flat_gene_id = list(chain.from_iterable(batch['gene_id']))
        masked_gene_id = list(compress(flat_gene_id, mask))
        flat_keys_id = list(chain.from_iterable(batch['selected_keys']))
        dataset_description = list(chain.from_iterable(batch['dataset_description']))

        preds = p_rmt.cpu().unsqueeze(1)
        target = y_rmt.cpu().unsqueeze(1)
        reduce_dims = (0, 1)
        data = {}

        loss_components = ['cls_loss', 'other_loss', 'mean_loss', 'diviation_loss']
        for loss_component in loss_components:
            if f'loss_{loss_component}' in output and output[f'loss_{loss_component}'] is not None:
                data[f'loss_{loss_component}'] = output[f'loss_{loss_component}'].detach().cpu()

        data['tpm_true'] = y_rmt.tolist()
        data['tpm_preds'] = p_rmt.tolist()
        data['gene_id'] = masked_gene_id
        data['keys_id'] = flat_keys_id
        data['dataset_description'] = dataset_description
        data['_product'] = torch.sum(preds * target, dim=reduce_dims).unsqueeze(0)
        data['_true'] = torch.sum(target, dim=reduce_dims).unsqueeze(0)
        data['_true_squared'] = torch.sum(torch.square(target), dim=reduce_dims).unsqueeze(0)
        data['_pred'] = torch.sum(preds, dim=reduce_dims).unsqueeze(0)
        data['_pred_squared'] = torch.sum(torch.square(preds), dim=reduce_dims).unsqueeze(0)
        data['_count'] = torch.sum(torch.ones_like(target), dim=reduce_dims).unsqueeze(0)
        return data

    def make_metrics_fn(model_path, save_predictions=False):
        def metrics_fn(data):
            metrics = {}

            data['_product'] = torch.sum(data['_product'], dim=0)
            data['_true'] = torch.sum(data['_true'], dim=0)
            data['_true_squared'] = torch.sum(data['_true_squared'], dim=0)
            data['_pred'] = torch.sum(data['_pred'], dim=0)
            data['_pred_squared'] = torch.sum(data['_pred_squared'], dim=0)
            data['_count'] = torch.sum(data['_count'], dim=0)

            true_mean = data['_true'] / data['_count']
            pred_mean = data['_pred'] / data['_count']
            covariance = (data['_product'] - true_mean * data['_pred'] - pred_mean * data['_true'] + data['_count'] * true_mean * pred_mean)

            true_var = data['_true_squared'] - data['_count'] * torch.square(true_mean)
            pred_var = data['_pred_squared'] - data['_count'] * torch.square(pred_mean)
            tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
            corr_coef = covariance / tp_var
            metrics['pearson_corr'] = corr_coef.item()

            loss_components = ['cls_loss', 'other_loss', 'mean_loss', 'diviation_loss']
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

                not_nan_values = df_true.notna().values & df_pred.notna().values
                df_true.values[~not_nan_values] = np.nan
                df_pred.values[~not_nan_values] = np.nan

                if not df_true.empty and not df_pred.empty:
                    gene_correlations = []
                    for gene in df_true.index:
                        gene_true = df_true.loc[gene]
                        gene_pred = df_pred.loc[gene]
                        gene_true = gene_true[pd.notna(gene_true)]
                        gene_pred = gene_pred[pd.notna(gene_pred)]

                        if len(gene_pred) > 1 and np.std(gene_pred) == 0:
                            alogger.error(f"dataset {dataset_desc} gene {gene} has all predicted values the same")
                            raise ValueError(f"All predicted values for {gene} are the same. Are you missing cell type descriptions?")

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

                    # 4 таргетных файла из borzoi
                    target_genes = ['ENCFF123KIW', 'ENCFF784MDF', 'ENCFF236XOK', 'ENCFF242BWW']
                    target_metrics = calculate_target_genes_metrics(df_true, df_pred, target_genes, logger=alogger)
                    metrics[f'pearson_corr_4_cells_borzoi_cells_{dataset_desc}'] = target_metrics['corr_selected_cell_types']
                    metrics[f'pearson_corr_4_cells_borzoi_genes_{dataset_desc}'] = target_metrics['corr_selected_genes']

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

            return metrics

        return metrics_fn

    metrics_fn = make_metrics_fn(args.model_path, save_predictions=args.save_predictions)

    # --- Подготовка под Accelerate (модель/оптимизатор)
    model, optimizer = accelerator.prepare(model, optimizer)

    # --- Trainer ---
    trainer = Trainer(
        args, accelerator, model, optimizer, train_dataloader,
        valid_dataloader=valid_dataloader,
        train_sampler=train_sampler,
        batch_transform_fn=batch_transform_fn,
        keep_for_metrics_fn=keep_for_metrics_fn,
        metrics_fn=metrics_fn
    )

    # --- Training ---
    accelerator.wait_for_everyone()
    trainer.train()
    accelerator.wait_for_everyone()

    # --- Post-training / validation / save ---
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
