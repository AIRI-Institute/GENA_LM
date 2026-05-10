import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from importlib import import_module
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import accelerate
import numpy as np
import pandas as pd
import torch
import transformers
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments
from transformers.optimization import Adafactor

from downstream_tasks.expression_prediction.datasets.src.correlation_selected_cells import (
    calculate_target_genes_metrics,
)
from downstream_tasks.expression_prediction.datasets.src.score_ct_specificity import (
    mean_and_residuals_correlation,
    score_predictions,
)
from downstream_tasks.expression_prediction.expression_dataset_final import worker_init_fn


logger_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArgs:
    experiment_config: str = field(metadata={"help": "Path to the Hydra experiment config"})
    model_path: Optional[str] = field(default=None, metadata={"help": "Output path root"})
    model_cls: Optional[str] = field(default=None, metadata={"help": "Model class path module:Class"})
    init_checkpoint: Optional[str] = field(default=None, metadata={"help": "Checkpoint file to initialize weights from"})
    resume_from_checkpoint: Optional[str] = field(default=None, metadata={"help": "Checkpoint directory for HF resume"})
    resume: Optional[str] = field(default=None, metadata={"help": "Alias for resume_from_checkpoint"})
    seed: int = 42
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    data_n_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.0
    optimizer: str = "AdamW"
    lr_scheduler: str = "constant_with_warmup"
    num_warmup_steps: int = 0
    iters: int = 1000
    log_interval: int = 50
    valid_interval: Optional[int] = None
    save_interval: Optional[int] = None
    save_best: bool = False
    optimize_metric: str = "loss"
    optimize_mode: str = "min"
    clip_grad_norm: Optional[float] = 1.0
    bf16: bool = True
    tf32: bool = True
    gen_tokenizer: Optional[str] = None
    text_tokenizer: Optional[str] = None
    save_predictions: bool = False
    log_level: int = logging.INFO
    save_total_limit: int = 2


def set_global_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cls_by_name(path: str):
    module_name, class_name = path.split(":")
    module = import_module(module_name)
    return getattr(module, class_name)


def merge_default_params_with_dataset_config(dataset_config, default_params, local_logger):
    def _merge_nested_dicts(target, source, path=""):
        for key, value in source.items():
            current_path = f"{path}.{key}" if path else key
            if key in target:
                if isinstance(value, dict) and isinstance(target[key], dict):
                    _merge_nested_dicts(target[key], value, current_path)
                else:
                    local_logger.warning(
                        "Parameter '%s' is specified in both default params and dataset config. "
                        "Using dataset config value: %s",
                        current_path,
                        target[key],
                    )
            else:
                target[key] = value.copy() if isinstance(value, dict) else value

    merged_config = OmegaConf.create(OmegaConf.to_container(dataset_config, resolve=True))
    default_params_dict = (
        OmegaConf.to_container(default_params, resolve=True)
        if hasattr(default_params, "items")
        else default_params
    )
    _merge_nested_dicts(merged_config, default_params_dict)
    return merged_config


def _collect_dataset_configs(experiment_config, prefix: str) -> List[Any]:
    return [v for k, v in experiment_config.items() if str(k).startswith(prefix)]


def get_shared_n_keys(shared_dataset_params) -> Optional[int]:
    if shared_dataset_params is None:
        return None
    try:
        value = OmegaConf.select(shared_dataset_params, "n_keys")
    except Exception:
        value = None
    if value is None and isinstance(shared_dataset_params, dict):
        value = shared_dataset_params.get("n_keys")
    if value is None:
        return None
    try:
        value = int(value)
    except Exception:
        return None
    return value if value > 0 else None


def _target_class_name(cfg: Any) -> str:
    try:
        target = cfg.get("_target_", "")
    except Exception:
        target = ""
    return str(target).split(".")[-1] if target else ""


def _is_expression_dataset_cfg(cfg: Any) -> bool:
    return _target_class_name(cfg) == "ExpressionDataset"


def infer_global_n_keys_from_expression_datasets(
    dataset_cfgs: List[Any],
    shared_dataset_params: Optional[Any],
    merge_fn,
    local_logger,
) -> int:
    if len(dataset_cfgs) == 0:
        raise ValueError("No dataset configs found")

    expr_cfgs = [cfg for cfg in dataset_cfgs if _is_expression_dataset_cfg(cfg)]
    if len(expr_cfgs) == 0:
        raise ValueError("Cannot infer n_keys: no ExpressionDataset configs found")

    tmp_cfgs = [cfg.copy() for cfg in expr_cfgs]
    if shared_dataset_params is not None:
        for i, cfg in enumerate(tmp_cfgs):
            tmp_cfgs[i] = merge_fn(cfg, shared_dataset_params, local_logger)

    counts = []
    for cfg in tmp_cfgs:
        cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
        targets_path = cfg_resolved.get("targets_path")
        if targets_path is None:
            raise ValueError("targets_path not found in ExpressionDataset config")
        df = pd.read_csv(targets_path)
        if "id" not in df.columns:
            raise ValueError(f"'id' column not found in targets_path={targets_path}")
        counts.append(int(df["id"].nunique()))

    n_keys = min(counts)
    local_logger.info("[n_keys] inferred from ExpressionDataset configs: min(%s) = %s", counts, n_keys)
    return n_keys


def apply_n_keys_to_all_dataset_cfgs(
    dataset_cfgs: List[Any],
    shared_dataset_params: Optional[Any],
    merge_fn,
    n_keys: int,
    local_logger,
) -> List[Any]:
    out = [cfg.copy() for cfg in dataset_cfgs]
    if shared_dataset_params is not None:
        for i, cfg in enumerate(out):
            out[i] = merge_fn(cfg, shared_dataset_params, local_logger)
    for cfg in out:
        OmegaConf.update(cfg, "n_keys", int(n_keys), force_add=True)
    return out


def build_dataset_from_cfgs(dataset_cfgs: List[Any]) -> Tuple[Dataset, List[Dataset]]:
    datasets = [instantiate(cfg) for cfg in dataset_cfgs]
    if len(datasets) == 0:
        raise ValueError("No datasets after instantiate()")
    if len(datasets) == 1:
        return datasets[0], datasets
    return ConcatDataset(datasets), datasets


def build_expression_collator(gen_tokenizer_name: str, text_tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(gen_tokenizer_name, trust_remote_code=True)
    text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name, trust_remote_code=True)
    text_pad_id = text_tokenizer.pad_token_id if text_tokenizer.pad_token_id is not None else 0

    def _pad_1d(x: torch.Tensor, length: int, pad_value: int, pad_left: bool = False) -> torch.Tensor:
        pad_len = length - x.size(0)
        if pad_len <= 0:
            return x
        pad = x.new_full((pad_len,), pad_value)
        return torch.cat([pad, x], dim=0) if pad_left else torch.cat([x, pad], dim=0)

    def _pad_nd(x: torch.Tensor, max_len: int, pad_value, dim: int) -> torch.Tensor:
        pad_len = max_len - x.size(dim)
        if pad_len <= 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_len
        pad = x.new_full(tuple(pad_shape), pad_value)
        return torch.cat([x, pad], dim=dim)

    def collate_fn(batch):
        pad_keys = ["input_ids", "attention_mask", "token_type_ids", "labels", "labels_mask"]
        no_pad_keys = ["tpm", "dataset_flag"]
        special_keys = [
            "gene_id",
            "selected_keys",
            "dataset_description",
            "name",
            "chrom",
            "reverse",
            "start",
            "end",
        ]

        pad_token_ids = {
            "input_ids": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
            "attention_mask": 0,
            "token_type_ids": 0,
            "labels": 0.0,
            "labels_mask": 0,
            "desc_input_ids": text_pad_id,
            "desc_attention_mask": 0,
        }

        max_seq_len = max(sample["input_ids"].size(1) for sample in batch)
        n_keys = len(batch[0]["desc_input_ids"])
        max_text_seq_len = max(
            max(ids.size(0) for ids in sample["desc_input_ids"])
            for sample in batch
        )
        max_text_seq_len = max(max_text_seq_len, 1)

        batch_dict = {key: [] for key in pad_keys + no_pad_keys + special_keys}
        desc_ids_batch = []
        desc_mask_batch = []

        for sample in batch:
            sample_ids = []
            sample_masks = []
            for k in range(n_keys):
                ids = _pad_1d(sample["desc_input_ids"][k], max_text_seq_len, pad_token_ids["desc_input_ids"], pad_left=True)
                mask = _pad_1d(
                    sample["desc_attention_mask"][k],
                    max_text_seq_len,
                    pad_token_ids["desc_attention_mask"],
                    pad_left=True,
                )
                sample_ids.append(ids)
                sample_masks.append(mask)
            desc_ids_batch.append(torch.stack(sample_ids, dim=0))
            desc_mask_batch.append(torch.stack(sample_masks, dim=0))

        for sample in batch:
            for key in pad_keys:
                x = sample[key]
                if key in ["input_ids", "attention_mask", "token_type_ids"]:
                    x = _pad_nd(x, max_seq_len, pad_token_ids[key], dim=1)
                if key in ["labels", "labels_mask"]:
                    x = _pad_nd(x, max_seq_len, pad_token_ids[key], dim=1)
                batch_dict[key].append(x)
            for key in no_pad_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])
            for key in special_keys:
                if key in sample:
                    batch_dict[key].append(sample[key])

        for key in pad_keys:
            batch_dict[key] = torch.stack(batch_dict[key], dim=0)
        for key in no_pad_keys:
            if len(batch_dict[key]) > 0:
                batch_dict[key] = torch.stack(batch_dict[key], dim=0)

        batch_dict["desc_input_ids"] = torch.stack(desc_ids_batch, dim=0)
        batch_dict["desc_attention_mask"] = torch.stack(desc_mask_batch, dim=0)
        return batch_dict

    return collate_fn


def save_run_metadata(output_dir: Path, args: ScriptArgs, experiment_config: Any, experiment_config_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_args.json", "w") as fout:
        json.dump(asdict(args), fout, indent=2)
    OmegaConf.save(experiment_config, output_dir / "experiment_config_resolved.yaml", resolve=True)
    with open(experiment_config_path, "r") as fin, open(output_dir / "experiment_config.yaml", "w") as fout:
        fout.write(fin.read())


def maybe_load_init_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[str], local_logger: logging.Logger) -> None:
    if not checkpoint_path:
        return
    checkpoint_path = str(Path(checkpoint_path).expanduser())
    local_logger.info("Loading init checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        local_logger.warning("Missing keys while loading init checkpoint: %s", missing[:20])
    if unexpected:
        local_logger.warning("Unexpected keys while loading init checkpoint: %s", unexpected[:20])


class ExpressionTrainer(Trainer):
    def __init__(
        self,
        *args,
        experiment_config: Any,
        raw_model_path: str,
        save_predictions: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.experiment_config = experiment_config
        self.raw_model_path = raw_model_path
        self.save_predictions = save_predictions
        self.model_forward_args = set(self._signature_columns or [])

    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        if dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = worker_init_fn
        return dataloader

    def get_eval_dataloader(self, eval_dataset=None):
        dataloader = super().get_eval_dataloader(eval_dataset)
        if dataloader.worker_init_fn is None:
            dataloader.worker_init_fn = worker_init_fn
        return dataloader

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        optimizer_name = getattr(self.args, "custom_optimizer_name", "AdamW").lower()
        if optimizer_name == "adafactor":
            self.optimizer = Adafactor(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
                scale_parameter=False,
                relative_step=False,
            )
            return self.optimizer

        if optimizer_name == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
            return self.optimizer

        torch_optim_cls = getattr(torch.optim, getattr(self.args, "custom_optimizer_name", "AdamW"), None)
        if torch_optim_cls is None:
            raise ValueError(f"Unsupported optimizer: {self.args.custom_optimizer_name}")
        self.optimizer = torch_optim_cls(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_inputs = {k: v for k, v in inputs.items() if k in self.model_forward_args}
        outputs = model(**model_inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def _flatten_gathered_objects(values: List[Any]) -> List[Any]:
        flattened = []
        for process_value in values:
            for sample_value in process_value:
                if isinstance(sample_value, list):
                    flattened.extend(sample_value)
                else:
                    flattened.append(sample_value)
        return flattened

    def _flatten_metadata(self, batch: Dict[str, Any], mask_idx: List[int]) -> Dict[str, List[Any]]:
        dataset_description = list(chain.from_iterable(batch["dataset_description"]))
        gene_id = list(chain.from_iterable(batch["gene_id"]))
        keys_id = list(chain.from_iterable(batch["selected_keys"]))
        return {
            "gene_id": [gene_id[i] for i in mask_idx],
            "keys_id": [keys_id[i] for i in mask_idx],
            "dataset_description": [dataset_description[i] for i in mask_idx],
        }

    def _compute_eval_metrics_from_records(self, data: Dict[str, List[Any]]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for key in ["cls_loss", "other_loss", "multinomial_loss", "deviation_loss", "loss"]:
            if key in data and len(data[key]) > 0:
                metrics[key] = float(np.mean(data[key]))

        if len(data.get("tpm_true", [])) == 0:
            return metrics

        df = pd.DataFrame(
            {
                "gene_id": data["gene_id"],
                "cell_type": data["keys_id"],
                "tpm_true": data["tpm_true"],
                "tpm_pred": data["tpm_preds"],
                "dataset_description": data["dataset_description"],
            }
        )

        if self.save_predictions and self.is_world_process_zero():
            df.to_csv(os.path.join(self.raw_model_path, "labels.csv"), index=False)

        for dataset_desc in df["dataset_description"].unique():
            df_dataset = df[df["dataset_description"] == dataset_desc]
            df_pred = df_dataset.pivot_table(
                index="gene_id",
                columns="cell_type",
                values="tpm_pred",
                aggfunc="first",
            )
            df_true = df_dataset.pivot_table(
                index="gene_id",
                columns="cell_type",
                values="tpm_true",
                aggfunc="first",
            )

            if self.save_predictions and self.is_world_process_zero():
                safe_name = dataset_desc.replace("/", "_")
                df_true.to_csv(os.path.join(self.raw_model_path, f"{safe_name}_true.csv"))
                df_pred.to_csv(os.path.join(self.raw_model_path, f"{safe_name}_pred.csv"))

            if df_true.empty or df_pred.empty:
                continue

            common_genes = df_true.index.intersection(df_pred.index)
            common_cells = df_true.columns.intersection(df_pred.columns)
            if len(common_genes) == 0 or len(common_cells) == 0:
                continue

            df_true = df_true.loc[common_genes, common_cells]
            df_pred = df_pred.loc[common_genes, common_cells]

            gene_correlations = []
            for gene in common_genes:
                gene_true = df_true.loc[gene]
                gene_pred = df_pred.loc[gene]
                mask = pd.notna(gene_true) & pd.notna(gene_pred)
                gene_true = gene_true[mask]
                gene_pred = gene_pred[mask]
                if len(gene_true) > 3 and np.std(gene_true) > 0:
                    corr = np.corrcoef(gene_true, gene_pred)[0, 1]
                    if not np.isnan(corr):
                        gene_correlations.append(corr)

            cell_correlations = []
            for cell_type in common_cells:
                cell_true = df_true[cell_type]
                cell_pred = df_pred[cell_type]
                cell_true = cell_true[pd.notna(cell_true)]
                cell_pred = cell_pred[pd.notna(cell_pred)]
                if len(cell_true) > 3 and np.std(cell_true) != 0:
                    corr = np.corrcoef(cell_true, cell_pred)[0, 1]
                    if not np.isnan(corr):
                        cell_correlations.append(corr)

            if gene_correlations:
                metrics[f"pearson_corr_cells_{dataset_desc}"] = float(np.mean(gene_correlations))
            if cell_correlations:
                metrics[f"pearson_corr_genes_{dataset_desc}"] = float(np.mean(cell_correlations))

            try:
                selected_metrics = calculate_target_genes_metrics(df_true.reset_index(), df_pred.reset_index())
            except Exception:
                selected_metrics = None
            if isinstance(selected_metrics, dict):
                for key, value in selected_metrics.items():
                    if isinstance(value, (np.floating, np.integer)):
                        value = value.item()
                    if isinstance(value, (int, float)):
                        metrics[f"target_genes_{key}_{dataset_desc}"] = float(value)

            allowed = {
                "Expression_dataset_v1_GRCh38_csv dataset",
                "Expression_dataset_v1_mm10_CPM dataset",
            }
            if dataset_desc in allowed:
                df_true_reset = df_true.reset_index()
                df_pred_reset = df_pred.reset_index()
                score = score_predictions(
                    df_true_reset,
                    df_pred_reset,
                    self.experiment_config.selected_targets_path,
                    need_log=False,
                    logger=logger,
                )
                if score and score.get("deviation_r", None) is not None:
                    metrics[f"score_predictions_{dataset_desc}"] = float(score["deviation_r"])

                score2 = mean_and_residuals_correlation(df_true_reset, df_pred_reset, need_log=False)
                if isinstance(score2, dict):
                    for key, value in score2.items():
                        if isinstance(value, (np.floating, np.integer)):
                            value = value.item()
                        if isinstance(value, (int, float)):
                            metrics[f"mean_residual_{key}_{dataset_desc}"] = float(value)

        return metrics

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()

        collected = {
            "loss": [],
            "cls_loss": [],
            "other_loss": [],
            "multinomial_loss": [],
            "deviation_loss": [],
            "tpm_true": [],
            "tpm_preds": [],
            "gene_id": [],
            "keys_id": [],
            "dataset_description": [],
        }

        for batch in eval_dataloader:
            model_inputs = self._prepare_inputs({k: v for k, v in batch.items() if k in self.model_forward_args})
            with torch.no_grad():
                outputs = model(**model_inputs)

            if outputs.loss is not None:
                collected["loss"].append(float(self.accelerator.gather_for_metrics(outputs.loss.detach().float().view(1)).mean().cpu()))

            for key in ["cls_loss", "other_loss", "multinomial_loss", "deviation_loss"]:
                value = getattr(outputs, key, None)
                if value is not None:
                    gathered = self.accelerator.gather_for_metrics(value.detach().float().view(1))
                    collected[key].extend(gathered.cpu().tolist())

            logits = self.accelerator.gather_for_metrics(outputs.logits.detach())
            labels = self.accelerator.gather_for_metrics(outputs.labels_reshaped.detach())
            masks = self.accelerator.gather_for_metrics(outputs.labels_mask_reshaped.detach())

            y_true = labels[:, 0, 0]
            y_pred = logits[:, 0, 0]
            mask = masks[:, 0, 0] > 0
            mask_idx = mask.nonzero(as_tuple=True)[0].tolist()

            gathered_gene_id = self._flatten_gathered_objects(accelerate.utils.gather_object(batch["gene_id"]))
            gathered_keys_id = self._flatten_gathered_objects(accelerate.utils.gather_object(batch["selected_keys"]))
            gathered_dataset_description = self._flatten_gathered_objects(
                accelerate.utils.gather_object(batch["dataset_description"])
            )

            collected["tpm_true"].extend(y_true[mask].cpu().tolist())
            collected["tpm_preds"].extend(y_pred[mask].cpu().tolist())
            collected["gene_id"].extend([gathered_gene_id[i] for i in mask_idx])
            collected["keys_id"].extend([gathered_keys_id[i] for i in mask_idx])
            collected["dataset_description"].extend([gathered_dataset_description[i] for i in mask_idx])

        metrics = self._compute_eval_metrics_from_records(collected)
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}
        self.log(metrics)
        return metrics


def main():
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]
    logging.getLogger().setLevel(args.log_level)

    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    experiment_config_path = Path(args.experiment_config).expanduser().absolute()
    with initialize_config_dir(str(experiment_config_path.parent), version_base=None):
        experiment_config = compose(config_name=experiment_config_path.name)

    if "args_params" in experiment_config:
        trainer_kwargs = instantiate(experiment_config["args_params"])
        for key, value in trainer_kwargs.items():
            if hasattr(args, key):
                setattr(args, key, value)

    set_global_seed(args.seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if args.model_path is None:
        if "model_path" not in experiment_config.args_params:
            raise ValueError("model_path must be set either via CLI or config")
        args.model_path = experiment_config.args_params.model_path
    args.model_path = os.path.join(str(args.model_path), timestamp)
    output_dir = Path(args.model_path)
    save_run_metadata(output_dir, args, experiment_config, experiment_config_path)

    logger.info("Output dir: %s", output_dir)

    train_cfgs = _collect_dataset_configs(experiment_config, "train_dataset")
    valid_cfgs = _collect_dataset_configs(experiment_config, "valid_dataset")
    shared_dataset_params = experiment_config.get("shared_dataset_params", None)

    if len(train_cfgs) == 0:
        raise ValueError("No training datasets found")

    shared_n_keys = get_shared_n_keys(shared_dataset_params)
    n_keys_train = infer_global_n_keys_from_expression_datasets(
        train_cfgs,
        shared_dataset_params,
        merge_default_params_with_dataset_config,
        logger,
    )
    n_keys_valid = (
        infer_global_n_keys_from_expression_datasets(
            valid_cfgs,
            shared_dataset_params,
            merge_default_params_with_dataset_config,
            logger,
        )
        if len(valid_cfgs) > 0
        else n_keys_train
    )

    if shared_n_keys is not None and shared_n_keys < n_keys_train:
        n_keys_train = shared_n_keys

    train_cfgs = apply_n_keys_to_all_dataset_cfgs(
        train_cfgs,
        shared_dataset_params,
        merge_default_params_with_dataset_config,
        n_keys_train,
        logger,
    )
    valid_cfgs = apply_n_keys_to_all_dataset_cfgs(
        valid_cfgs,
        shared_dataset_params,
        merge_default_params_with_dataset_config,
        n_keys_valid,
        logger,
    )

    expanded_train_cfgs = []
    for cfg in train_cfgs:
        repeat = int(cfg.get("repeat_factor", 1))
        expanded_train_cfgs.extend([cfg.copy()] * repeat)
    train_cfgs = expanded_train_cfgs

    train_dataset, train_datasets_list = build_dataset_from_cfgs(train_cfgs)
    valid_dataset = None
    if len(valid_cfgs) > 0:
        valid_dataset, valid_datasets_list = build_dataset_from_cfgs(valid_cfgs)
    else:
        valid_datasets_list = []

    for i, ds in enumerate(train_datasets_list):
        logger.info("train dataset %s: %s", i, ds.describe() if hasattr(ds, "describe") else type(ds))
    for i, ds in enumerate(valid_datasets_list):
        logger.info("valid dataset %s: %s", i, ds.describe() if hasattr(ds, "describe") else type(ds))

    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    collate_fn = build_expression_collator(args.gen_tokenizer, args.text_tokenizer)

    model_kwargs = instantiate(experiment_config["model_kwargs"]) if "model_kwargs" in experiment_config else {}
    model_cls = get_cls_by_name(args.model_cls)
    model = model_cls(**model_kwargs)
    maybe_load_init_checkpoint(model, args.init_checkpoint, logger)

    resume_from_checkpoint = args.resume_from_checkpoint or args.resume
    if resume_from_checkpoint and os.path.isfile(resume_from_checkpoint):
        maybe_load_init_checkpoint(model, resume_from_checkpoint, logger)
        resume_from_checkpoint = None

    metric_for_best_model = f"eval_{args.optimize_metric}"
    evaluation_strategy = "steps" if valid_dataset is not None else "no"
    save_steps = args.valid_interval if (args.save_best and valid_dataset is not None) else (args.save_interval or args.log_interval)
    if save_steps is None:
        save_steps = args.log_interval

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        do_train=True,
        do_eval=valid_dataset is not None,
        evaluation_strategy=evaluation_strategy,
        eval_steps=args.valid_interval if valid_dataset is not None else None,
        save_strategy="steps",
        save_steps=save_steps,
        logging_strategy="steps",
        logging_steps=args.log_interval,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.iters,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.num_warmup_steps,
        bf16=args.bf16,
        dataloader_num_workers=args.data_n_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=bool(args.save_best and valid_dataset is not None),
        metric_for_best_model=metric_for_best_model if valid_dataset is not None else None,
        greater_is_better=(args.optimize_mode == "max"),
        max_grad_norm=args.clip_grad_norm if args.clip_grad_norm is not None else 0.0,
        report_to=["tensorboard"],
        save_safetensors=False,
        logging_first_step=True,
    )
    setattr(training_args, "custom_optimizer_name", args.optimizer)

    trainer = ExpressionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_fn,
        experiment_config=experiment_config,
        raw_model_path=str(output_dir),
        save_predictions=args.save_predictions,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()

    train_metrics = train_result.metrics
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)

    if valid_dataset is not None:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("HF Trainer fine-tuning failed")
        raise
