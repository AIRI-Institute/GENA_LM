"""Fine-tuning runner for the convolutional (BPE-free) expression model.

Same trainer, optimizer and metrics as ``run_expression_finetuning_final.py``.
One item is a gene together with ``n_keys`` tracks, as in the token model, so
the DNA branch is shared across the tracks. The differences are:

* batches carry ``dna_codes`` instead of ``input_ids``/``token_type_ids``, and
  hold one window per gene rather than one per (gene, track) row;
* ``desc_index`` is added so the description encoder runs once per distinct
  track in the batch;
* ``n_keys`` is taken from ``shared_dataset_params`` instead of being inferred,
  and there is no ``dataset_flag``: every item is a gene across tracks.

The metric functions themselves are unchanged, so ``pearson_corr_cells_*``,
``pearson_corr_genes_*``, ``score_predictions_*`` and ``mean_residual_*`` are
computed exactly as before.
"""

# stdlib
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, List, Tuple

# third-party
import accelerate
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, HfArgumentParser

# local
from lm_experiments_tools import TrainerAccelerate as Trainer
from lm_experiments_tools import TrainerAccelerateArgs as TrainerArgs
from lm_experiments_tools.utils import (
    collect_run_configuration,
    get_cls_by_name,
    get_git_diff,
    prepare_run,
)
import lm_experiments_tools.optimizers as optimizers
from lm_experiments_tools import get_optimizer

from downstream_tasks.expression_prediction.expression_dataset_final import worker_init_fn
from downstream_tasks.expression_prediction.datasets.src.score_ct_specificity import (
    mean_and_residuals_correlation,
    score_predictions,
)


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
    """Merge shared_dataset_params into a dataset config; explicit values win."""
    def _merge_nested_dicts(target, source, path=""):
        for key, value in source.items():
            current_path = f"{path}.{key}" if path else key
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    _merge_nested_dicts(target[key], value, current_path)
                else:
                    logger.warning(
                        f"Parameter '{current_path}' is specified in both default params and "
                        f"dataset config. Using dataset config value: {target[key]}"
                    )
            else:
                target[key] = value.copy() if isinstance(value, dict) else value
                logger.info(f"Added default parameter '{current_path}': {value}")

    merged_config = OmegaConf.create(OmegaConf.to_container(dataset_config, resolve=True))
    default_params_dict = (
        OmegaConf.to_container(default_params, resolve=True)
        if hasattr(default_params, 'items')
        else default_params
    )
    _merge_nested_dicts(merged_config, default_params_dict)
    return merged_config


def _collect_dataset_configs(experiment_config, prefix: str) -> List[Any]:
    return [v for k, v in experiment_config.items() if str(k).startswith(prefix)]


def load_selected_gene_ids(path_to_selected: str) -> set:
    """Every gene id referenced by the cell-type-specificity benchmark.

    ``score_predictions`` filters its input down to these genes before asserting
    that the remaining rows share one genome. If the filter empties the frame the
    assertion fires with an empty list instead of taking the function's own
    "nothing to score" branch further down. A metrics window that happens to
    contain none of the benchmark genes is perfectly normal -- especially for
    train metrics, which are collected over a handful of steps -- so pre-checking
    the overlap keeps that from killing a long run.
    """
    selected = pd.read_csv(path_to_selected)
    gene_ids = set()
    for row in selected["gene_id"].dropna():
        gene_ids.update(g.strip() for g in str(row).split(",") if g.strip())
    return gene_ids


def apply_shared_params_to_cfgs(dataset_cfgs, shared_dataset_params, alogger) -> List[Any]:
    if shared_dataset_params is None:
        return list(dataset_cfgs)
    return [
        merge_default_params_with_dataset_config(cfg, shared_dataset_params, alogger)
        for cfg in dataset_cfgs
    ]


def build_dataset_from_cfgs(dataset_cfgs: List[Any]) -> Tuple[Dataset, List[Dataset]]:
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

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )
    alogger = get_logger(__name__)
    alogger.info(f'num processes: {accelerator.num_processes}')
    alogger.info(f'mixed precision: {accelerator.mixed_precision}')

    prepare_run(args, alogger, logger_fmt, accelerator=accelerator)

    timestamp = time.strftime("%Y%m%d-%H%M%S") if accelerator.is_main_process else None
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
            f"Resuming from cpt {args.resume}. This will overwrite reset_lr, reset_optimizer, "
            f"reset_iteration and init_cpt options"
        )
        args.__setattr__("reset_lr", False)
        args.__setattr__("reset_optimizer", False)
        args.__setattr__("reset_iteration", False)
        args.__setattr__("init_checkpoint", args.model_path + "/" + args.resume)
        args.__setattr__("model_path", args.model_path + "/resume_" + args.resume + "/")

    if accelerator.is_main_process and args.model_path is None:
        raise ValueError("Model path should not be None")

    args.model_path = os.path.join(args.model_path, timestamp)
    alogger.info(f"rank: {accelerator.process_index}, Model path: {args.model_path}")

    if accelerator.is_main_process and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            model_path.mkdir(parents=True)
        json.dump(collect_run_configuration(args), open(model_path / 'config.json', 'w'), indent=4)
        open(model_path / 'git.diff', 'w').write(get_git_diff())
        content = "\n".join(open(experiment_config_path).readlines())
        with open(model_path / "experiment_config.yaml", "w") as fout:
            fout.write(content)

    accelerator.wait_for_everyone()

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_tokenizer, trust_remote_code=True)
    desc_pad_id = text_tokenizer.pad_token_id
    if desc_pad_id is None:
        desc_pad_id = text_tokenizer.eos_token_id or 0

    def _pad_left(x: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
        pad_len = length - x.size(0)
        if pad_len <= 0:
            return x
        return torch.cat([x.new_full((pad_len,), pad_value), x], dim=0)

    def collate_fn(batch):
        """Stack one DNA window per gene plus its n_keys tracks.

        ``dna_codes`` is (B, S) -- one window per gene, not per (gene, track)
        row -- and the model broadcasts the tower output over the tracks.
        ``desc_index`` is built here rather than in the dataset: a track index is
        only unique within one dataset, and a ConcatDataset batch mixes human and
        mouse, where index 0 denotes different descriptions.
        """
        window_lengths = {sample['dna_codes'].size(0) for sample in batch}
        if len(window_lengths) != 1:
            raise ValueError(
                f"All datasets must share one dna_window_len, got {sorted(window_lengths)}. "
                "Set dna_window_len in shared_dataset_params."
            )
        seq_lengths = {sample['attention_mask'].size(0) for sample in batch}
        if len(seq_lengths) != 1:
            raise ValueError(f"Inconsistent bin counts across datasets: {sorted(seq_lengths)}")
        n_keys_seen = {sample['labels'].size(0) for sample in batch}
        if len(n_keys_seen) != 1:
            raise ValueError(
                f"All datasets must share one n_keys, got {sorted(n_keys_seen)}. "
                "Set n_keys in shared_dataset_params."
            )

        max_text_len = max(t.size(0) for s in batch for t in s['desc_input_ids'])

        desc_ids, desc_masks, desc_index = [], [], []
        key_to_id = {}
        for sample in batch:
            desc_ids.append(torch.stack(
                [_pad_left(t, max_text_len, desc_pad_id) for t in sample['desc_input_ids']]))
            desc_masks.append(torch.stack(
                [_pad_left(t, max_text_len, 0) for t in sample['desc_attention_mask']]))
            row = []
            for key in sample['selected_keys']:
                token = (sample['dataset_description'], key)
                row.append(key_to_id.setdefault(token, len(key_to_id)))
            desc_index.append(torch.tensor(row, dtype=torch.long))

        batch_dict = {
            'dna_codes': torch.stack([s['dna_codes'] for s in batch], dim=0),
            'attention_mask': torch.stack([s['attention_mask'] for s in batch], dim=0),
            'labels': torch.stack([s['labels'] for s in batch], dim=0),
            'labels_mask': torch.stack([s['labels_mask'] for s in batch], dim=0),
            'desc_input_ids': torch.stack(desc_ids, dim=0),
            'desc_attention_mask': torch.stack(desc_masks, dim=0),
            'desc_index': torch.stack(desc_index, dim=0),
        }
        for key in ('gene_id', 'selected_keys', 'dataset_description', 'chrom',
                    'reverse', 'start', 'end'):
            if key in batch[0]:
                batch_dict[key] = [s[key] for s in batch]
        return batch_dict

    # Data
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    shared_dataset_params = experiment_config.get("shared_dataset_params", None)

    train_cfgs = _collect_dataset_configs(experiment_config, "train_dataset")
    valid_cfgs = _collect_dataset_configs(experiment_config, "valid_dataset")
    if len(train_cfgs) == 0:
        raise ValueError("No training datasets found (no train_dataset* in config)")

    train_cfgs = apply_shared_params_to_cfgs(train_cfgs, shared_dataset_params, alogger)
    valid_cfgs = apply_shared_params_to_cfgs(valid_cfgs, shared_dataset_params, alogger)

    expanded_train_cfgs = []
    for cfg in train_cfgs:
        expanded_train_cfgs.extend([cfg.copy()] * int(cfg.get("repeat_factor", 1)))
    train_cfgs = expanded_train_cfgs

    train_dataset, train_datasets_list = build_dataset_from_cfgs(train_cfgs)
    if accelerator.is_main_process:
        for i, ds in enumerate(train_datasets_list):
            alogger.info(f"train dataset {i}: {ds.describe() if hasattr(ds, 'describe') else type(ds)}")
        alogger.info(f"total len(train_dataset)={len(train_dataset)}")

    train_dataloader, train_sampler = build_loader(
        dataset=train_dataset,
        accelerator=accelerator,
        batch_size=per_worker_batch_size,
        seed=args.seed,
        shuffle=True,
        drop_last=False,
        num_workers=args.data_n_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    if len(valid_cfgs) > 0:
        valid_dataset, valid_datasets_list = build_dataset_from_cfgs(valid_cfgs)
        if accelerator.is_main_process:
            for i, ds in enumerate(valid_datasets_list):
                alogger.info(f"valid dataset {i}: {ds.describe() if hasattr(ds, 'describe') else type(ds)}")
            alogger.info(f"total len(valid_dataset)={len(valid_dataset)}")
        valid_dataloader, valid_sampler = build_loader(
            dataset=valid_dataset,
            accelerator=accelerator,
            batch_size=per_worker_batch_size,
            seed=args.seed,
            shuffle=False,
            drop_last=False,
            num_workers=args.data_n_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
        )
    else:
        valid_dataloader = None
        valid_sampler = None
        if accelerator.is_main_process:
            alogger.info("No validation data is used.")

    # Model
    model_kwargs = instantiate(experiment_config["model_kwargs"]) if "model_kwargs" in experiment_config else {}
    model_cls = get_cls_by_name(args.model_cls)
    if accelerator.is_main_process:
        alogger.info(f'Using model class: {model_cls}')
    model = model_cls(**model_kwargs)

    if accelerator.is_main_process:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for p in model.cnn.parameters())
        alogger.info(
            f"[params] total={total:,} trainable={trainable:,} cnn={cnn_params:,} "
            f"({100.0 * cnn_params / max(total, 1):.1f}% of total)"
        )

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
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def batch_transform_fn(batch):
        return {
            'dna_codes': batch['dna_codes'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels'],
            'labels_mask': batch['labels_mask'],
            'desc_input_ids': batch['desc_input_ids'],
            'desc_attention_mask': batch['desc_attention_mask'],
            'desc_index': batch['desc_index'],
            'gene_id': batch['gene_id'],
            'selected_keys': batch['selected_keys'],
            'dataset_description': batch['dataset_description'],
        }

    def keep_for_metrics_fn(batch, output):
        logits = output["logits"].detach().cpu()
        labels = output["labels_reshaped"].detach().cpu()
        masks = output["labels_mask_reshaped"].detach().cpu()

        # Position 0 is the CLS slot carrying the gene-level target.
        y_true = labels[:, 0, 0]
        y_pred = logits[:, 0, 0]
        mask = masks[:, 0, 0] > 0

        data = {}
        for k in ["cls_loss", "other_loss", "deviation_loss", "multinomial_loss"]:
            if k in output and output[k] is not None:
                data[k] = output[k].detach().cpu()

        # Rows are gene-major: gene 0's n_keys tracks, then gene 1's, matching
        # labels.reshape(B * N, ...) in the model. Expand the per-gene metadata
        # the same way, then filter both by the same mask.
        n_keys = len(batch['selected_keys'][0])
        gene_id = [g for g in batch['gene_id'] for _ in range(n_keys)]
        keys_id = [k for keys in batch['selected_keys'] for k in keys]
        dataset_description = [d for d in batch['dataset_description'] for _ in range(n_keys)]
        assert len(gene_id) == len(keys_id) == mask.numel(), (
            f"metadata/rows mismatch: gene_id={len(gene_id)}, keys_id={len(keys_id)}, "
            f"rows={mask.numel()}"
        )

        keep = mask.tolist()
        data['tpm_true'] = y_true[mask].tolist()
        data['tpm_preds'] = y_pred[mask].tolist()
        data['gene_id'] = [g for g, m in zip(gene_id, keep) if m]
        data['keys_id'] = [k for k, m in zip(keys_id, keep) if m]
        data['dataset_description'] = [d for d, m in zip(dataset_description, keep) if m]

        assert (
            len(data['tpm_true'])
            == len(data['gene_id'])
            == len(data['keys_id'])
            == len(data['dataset_description'])
        ), (
            "Mismatch in metadata sizes: "
            f"tpm_true={len(data['tpm_true'])}, gene_id={len(data['gene_id'])}, "
            f"keys_id={len(data['keys_id'])}, dataset_description={len(data['dataset_description'])}"
        )
        return data

    selected_gene_ids = load_selected_gene_ids(experiment_config.selected_targets_path)
    if accelerator.is_main_process:
        alogger.info(f"[metrics] benchmark genes in selected_targets: {len(selected_gene_ids)}")

    def make_metrics_fn(model_path, save_predictions=False):
        def metrics_fn(data):
            metrics = {}
            for k in ["cls_loss", "other_loss", "deviation_loss", "multinomial_loss"]:
                if k in data and data[k] is not None:
                    metrics[k] = torch.mean(data[k]).item()

            df = pd.DataFrame({
                'gene_id': data['gene_id'],
                'cell_type': data['keys_id'],
                'tpm_true': data['tpm_true'],
                'tpm_pred': data['tpm_preds'],
                'dataset_description': data['dataset_description'],
            })
            if save_predictions and accelerator.is_main_process:
                df.to_csv(os.path.join(model_path, "labels.csv"))

            for dataset_desc in df['dataset_description'].unique():
                df_dataset = df[df['dataset_description'] == dataset_desc]

                df_pred = df_dataset.pivot_table(
                    index='gene_id', columns='cell_type', values='tpm_pred', aggfunc='first'
                )
                df_true = df_dataset.pivot_table(
                    index='gene_id', columns='cell_type', values='tpm_true', aggfunc='first'
                )
                if save_predictions and accelerator.is_main_process:
                    df_true.to_csv(os.path.join(model_path, f"{dataset_desc}_true.csv"))
                    df_pred.to_csv(os.path.join(model_path, f"{dataset_desc}_pred.csv"))

                if df_true.empty or df_pred.empty:
                    continue

                gene_correlations = []
                for gene in df_true.index:
                    gene_true = df_true.loc[gene]
                    gene_pred = df_pred.loc[gene]
                    valid = pd.notna(gene_true) & pd.notna(gene_pred)
                    gene_true = gene_true[valid]
                    gene_pred = gene_pred[valid]
                    if len(gene_true) > 3 and np.std(gene_true) > 0:
                        try:
                            corr = np.corrcoef(gene_true, gene_pred)[0, 1]
                            if not np.isnan(corr):
                                gene_correlations.append(corr)
                        except Exception:
                            continue

                cell_correlations = []
                degenerate_cells = []
                for cell_type in df_true.columns:
                    cell_true = df_true[cell_type]
                    cell_pred = df_pred[cell_type]
                    # The token model raises here, because a pretrained GENA tower
                    # gives varied embeddings from step 0 and a constant column
                    # means something is wrong. This model starts from a randomly
                    # initialised CNN encoder, so an early constant-prediction
                    # phase is expected -- and in bf16 near-equal values round to
                    # bitwise identical ones. Warn and keep training instead of
                    # killing a long run; pred_std_* below makes it visible.
                    if np.std(cell_pred.values) == 0 and len(cell_pred) > 1:
                        degenerate_cells.append(str(cell_type))
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

                # Spread of predictions: if this decays to 0 the model has
                # collapsed to a constant and the correlations are meaningless.
                metrics[f'pred_std_{dataset_desc}'] = float(np.std(df_dataset['tpm_pred']))
                metrics[f'pred_std_within_cell_{dataset_desc}'] = float(
                    np.nanmean(df_pred.std(axis=0, ddof=0).values)
                )
                if degenerate_cells:
                    metrics[f'degenerate_cells_{dataset_desc}'] = len(degenerate_cells)
                    alogger.warning(
                        f"{dataset_desc}: {len(degenerate_cells)} of {len(df_true.columns)} "
                        f"cell types have a constant prediction across genes "
                        f"(e.g. {degenerate_cells[:3]}). Expected while the CNN encoder is "
                        f"still near its initialisation; investigate if it persists."
                    )

                ALLOWED = {
                    "Expression_dataset_v1_GRCh38_csv dataset",
                    "Expression_dataset_v1_mm10_CPM dataset",
                }
                if dataset_desc in ALLOWED:
                    n_selected = int(df_dataset['gene_id'].isin(selected_gene_ids).sum())
                    metrics[f'selected_genes_seen_{dataset_desc}'] = n_selected
                    if n_selected == 0:
                        # Nothing to score; see load_selected_gene_ids.
                        alogger.info(
                            f"{dataset_desc}: this metrics window contains none of the "
                            f"{len(selected_gene_ids)} benchmark genes, skipping "
                            f"score_predictions"
                        )
                    else:
                        df_true_r = df_true.reset_index()
                        df_pred_r = df_pred.reset_index()
                        try:
                            score = score_predictions(
                                df_true_r,
                                df_pred_r,
                                experiment_config.selected_targets_path,
                                need_log=False,
                                logger=alogger,
                            )
                            if score and score.get('deviation_r', None):
                                metrics[f'score_predictions_{dataset_desc}'] = score['deviation_r']
                            score2 = mean_and_residuals_correlation(
                                df_true_r, df_pred_r, need_log=False
                            )
                            if isinstance(score2, dict):
                                for k, v in score2.items():
                                    if isinstance(v, (np.floating, np.integer)):
                                        v = v.item()
                                    metrics[f"mean_residual_{k}_{dataset_desc}"] = v
                        except Exception:
                            # Never let a benchmark-scoring edge case abort training;
                            # the traceback is logged so it stays visible.
                            alogger.exception(
                                f"{dataset_desc}: score_predictions failed on a window with "
                                f"{n_selected} benchmark genes; continuing"
                            )

            return metrics

        return metrics_fn

    metrics_fn = make_metrics_fn(args.model_path, save_predictions=args.save_predictions)

    model, optimizer = accelerator.prepare(model, optimizer)

    trainer = Trainer(
        args, accelerator, model, optimizer, train_dataloader,
        valid_dataloader=valid_dataloader,
        train_sampler=train_sampler,
        batch_transform_fn=batch_transform_fn,
        keep_for_metrics_fn=keep_for_metrics_fn,
        metrics_fn=metrics_fn,
    )

    accelerator.wait_for_everyone()
    trainer.train()
    accelerator.wait_for_everyone()

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
