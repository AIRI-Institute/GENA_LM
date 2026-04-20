import logging
import os
from pathlib import Path
import shutil
from hydra.utils import instantiate
import torch
from argparse import ArgumentParser
from hydra import initialize_config_dir, compose
import math
from accelerate import Accelerator

from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset, Dataset, Subset
from typing import List, Tuple, Any
from safetensors.torch import load_file
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score
)
import polars as pl  # ← Added Polars

def _collect_dataset_configs(experiment_config, prefix: str) -> List[Any]:
    """Берёт все ключи вида train_dataset*, valid_dataset*."""
    return [v for k, v in experiment_config.items() if str(k).startswith(prefix)]

def build_dataset_from_cfgs(dataset_cfgs: List[Any]) -> Dataset:
    """Instantiate -> Dataset or ConcatDataset."""
    datasets = [instantiate(cfg) for cfg in dataset_cfgs]
    if len(datasets) == 0:
        raise ValueError("No datasets after instantiate()")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)

checkpoint = '/workspace-SR003.nfs2/estsoi/TSSprediction/runs/CRE_prediction/checkpoint-13000/model.safetensors'

class modelValidator():
    
    def __init__(self, checkpoint: str, config: str | Path):
        self.checkpoint = checkpoint
        self.config = config
        self.device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        self.logger = logging.getLogger(__name__)
        self.dataset = None
        self.model = None
        self.balanced_indices = None
        self.balanced_dataset = None
        self.config_dict = None
        
    def instantiate_model(self):
        self.logger.info('INSTANTIATING MODEL')
        if not isinstance(self.config, Path):
            self.config = Path(self.config).absolute()
        with initialize_config_dir(str(self.config.parents[0])):
            experiment_config = compose(config_name=self.config.name)
            self.config_dict = experiment_config
        model = instantiate(experiment_config['model'])
        state_dict = load_file(checkpoint, device="cpu")
        model.load_state_dict(state_dict)
        self.model = model

    def instantiate_dataset(self, prefix: str):
        self.logger.info('INSTANTIATING DATASET')
        if self.config_dict is None:
            with initialize_config_dir(str(self.config.parents[0])):
                self.config_dict = compose(config_name=self.config.name)
        dataset_cfgs = _collect_dataset_configs(experiment_config=self.config_dict, prefix=prefix)
        dataset = build_dataset_from_cfgs(dataset_cfgs=dataset_cfgs)
        self.dataset = dataset

    def compute_balanced_indices(self):
        self.logger.info('COMPUTING BALANCED INDICES')
        assert isinstance(self.dataset, Dataset | ConcatDataset)
        dataset_len = len(self.dataset)
        positive_class, negative_class = [], []
        
        for idx in range(dataset_len):
            dataset_entry = self.dataset[idx]
            label = (dataset_entry['labels'] * dataset_entry['labels_mask']).sum()
            if label == 0:
                negative_class.append(idx)
            elif label == 1:
                positive_class.append(idx)
            else:
                raise Exception(f'Invalid label encountered: {label}')
        
        rng = np.random.default_rng(seed=42)
        len_neg, len_pos = len(negative_class), len(positive_class)
        
        if len_neg > len_pos:
            negative_class = rng.choice(negative_class, len_pos).tolist()
        else:
            positive_class = rng.choice(positive_class, len_neg).tolist()
        
        self.balanced_indices = sorted(negative_class + positive_class)
                    
    def balance_classes(self):
        self.logger.info('BALANCING CLASSES')
        assert self.dataset is not None and self.balanced_indices is not None
        self.balanced_dataset = Subset(self.dataset, indices=self.balanced_indices)
        
    def compute_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                       logits: np.ndarray = None) -> dict:
        """Compute classification metrics with optional ROC-AUC."""
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        f1 = f1_score(targets, predictions, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'balanced_accuracy': balanced_accuracy,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'negative_predictive_value': npv,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(targets)
        }
        
        if logits is not None and len(np.unique(targets)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(targets, logits)
            except Exception as e:
                self.logger.warning(f"Could not compute ROC-AUC: {e}")
        
        return metrics
    
    def compute_metrics_at_threshold(self, logits: np.ndarray, targets: np.ndarray, 
                                     threshold: float) -> dict:
        """Compute metrics at a specific probability threshold."""
        predictions = (logits >= threshold).astype(int)
        return self.compute_metrics(predictions, targets, logits)
    
    def run_inference(self, use_balanced_dataset: bool = False):
        """Run inference and return predictions, labels, and logits."""
        self.logger.info('RUNNING INFERENCE AND EVALUATION')
        assert isinstance(self.model, torch.nn.Module)
        
        eval_dataset = self.balanced_dataset if use_balanced_dataset else self.dataset
        assert eval_dataset is not None, "Dataset not initialized"
        
        self.model.to(self.device)
        self.model.eval()
        dataloader = DataLoader(eval_dataset, batch_size=20, shuffle=False)
        
        all_predictions, all_labels, all_logits = [], [], []
        
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                taxon = batch['taxon'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels_mask=None,
                    labels=None,
                    taxon=taxon
                )
                
                predicts = outputs.predicts[:, 0:1, :].unsqueeze(-1)
                batch_labels = labels[:, 0, 0].cpu().numpy()
                batch_logits = torch.sigmoid(predicts).cpu().numpy().flatten()
                batch_predictions = (batch_logits > 0.5).astype(int)
                
                all_predictions.extend(batch_predictions)
                all_labels.extend(batch_labels)
                all_logits.extend(batch_logits)
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        self.logger.info(f"Total samples: {len(all_labels)}")
        
        # Default metrics at threshold 0.5
        metrics = self.compute_metrics(all_predictions, all_labels, all_logits)
        
        self.logger.info("=" * 50)
        self.logger.info("EVALUATION RESULTS (threshold=0.5)")
        self.logger.info("=" * 50)
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"{metric_name}: {value:.4f}")
            else:
                self.logger.info(f"{metric_name}: {value}")
        
        return metrics, all_predictions, all_labels, all_logits

    def evaluate_at_thresholds(self, logits: np.ndarray, labels: np.ndarray, 
                               thresholds: List[float] = None) -> pl.DataFrame:
        """
        Evaluate model at multiple thresholds and return TRANSPOSED Polars DataFrame.
        
        Layout:
            - Rows: metric names
            - Columns: threshold values (0.5, 0.4, 0.3, 0.1, 0.01)
        
        Args:
            logits: Model output probabilities (post-sigmoid)
            labels: Ground truth binary labels
            thresholds: List of thresholds to evaluate
        
        Returns:
            Transposed Polars DataFrame: metrics × thresholds
        """
        if thresholds is None:
            thresholds = [0.5, 0.4, 0.3, 0.1, 0.01]
        
        # Collect metrics for each threshold
        results = []
        for thresh in thresholds:
            metrics = self.compute_metrics_at_threshold(logits, labels, thresh)
            metrics['threshold'] = thresh
            results.append(metrics)
        
        # Create DataFrame with threshold as index-like column
        df = pl.DataFrame(results)
        
        # Separate metric columns from threshold
        metric_cols = [c for c in df.columns if c != 'threshold']
        
        # Transpose: metrics become rows, thresholds become columns
        transposed_rows = []
        for metric in metric_cols:
            row_data = {'metric': metric}
            for row in df.iter_rows(named=True):
                row_data[str(row['threshold'])] = row[metric]
            transposed_rows.append(row_data)
        
        df_transposed = pl.DataFrame(transposed_rows)
        
        # Optional: sort metrics by category for readability
        priority_order = [
            'accuracy', 'balanced_accuracy', 'f1_score', 'precision', 'recall', 
            'specificity', 'roc_auc', 'false_positive_rate', 'false_negative_rate',
            'negative_predictive_value', 'true_positives', 'true_negatives',
            'false_positives', 'false_negatives', 'total_samples'
        ]
        # Keep metrics that exist, maintain order, append any extras at end
        existing_metrics = set(df_transposed['metric'])
        sorted_metrics = [m for m in priority_order if m in existing_metrics]
        sorted_metrics += [m for m in existing_metrics if m not in priority_order]
        
        df_transposed = df_transposed.sort(
            pl.col('metric').map_elements(
                lambda x: sorted_metrics.index(x) if x in sorted_metrics else 999,
                return_dtype=pl.Int32
            )
        ).drop('metric').insert_column(0, pl.Series('metric', sorted_metrics))
        
        return df_transposed


# ==================== USAGE ====================
validator = modelValidator(checkpoint=checkpoint, config='configs/run_config.yaml')

validator.instantiate_model()
validator.instantiate_dataset(prefix='valid_dataset')
validator.compute_balanced_indices()
validator.balance_classes()

# Run inference on balanced dataset
metrics, predictions, labels, logits = validator.run_inference(use_balanced_dataset=True)

# Evaluate at multiple thresholds using Polars
thresholds = [0.5, 0.4, 0.3, 0.1, 0.01]
results_df = validator.evaluate_at_thresholds(logits, labels, thresholds=thresholds)

# Simple, robust display
print("\n" + "=" * 120)
print("METRICS BY THRESHOLD (BALANCED DATASET)")
print("=" * 120)

# Just round all float columns, leave integers alone
formatted_df = results_df.clone()
for c in results_df.columns:
    if c != 'metric' and results_df[c].dtype.is_float():
        formatted_df = formatted_df.with_columns(pl.col(c).round(4))

# Convert to pandas for clean terminal output
pdf = formatted_df.to_pandas()
pdf['metric'] = pdf['metric'].str.ljust(25)  # Pad metric names for alignment
print(pdf.to_string(index=False, float_format="%.4f"))

# Optional: Export to CSV
results_df.write_csv("validation_metrics_balanced_by_threshold.csv")

# Evaluate on full (imbalanced) dataset
full_metrics, _, full_labels, full_logits = validator.run_inference(use_balanced_dataset=False)
full_results_df = validator.evaluate_at_thresholds(full_logits, full_labels, thresholds=thresholds)

print("\n" + "=" * 100)
print("METRICS AT DIFFERENT THRESHOLDS (FULL DATASET)")
print("=" * 100)
formatted_df = full_results_df.clone()
for c in full_results_df.columns:
    if c != 'metric' and full_results_df[c].dtype.is_float():
        formatted_df = formatted_df.with_columns(pl.col(c).round(4))
# Convert to pandas for clean terminal output
pdf = formatted_df.to_pandas()
pdf['metric'] = pdf['metric'].str.ljust(25)  # Pad metric names for alignment
print(pdf.to_string(index=False, float_format="%.4f"))

# Quick access to key metrics at default threshold
print(f"\nBalanced dataset @ threshold=0.5:")
print(f"   Accuracy:  {metrics['accuracy']:.4f}")
print(f"   F1 Score:  {metrics['f1_score']:.4f}")
print(f"   Precision: {metrics['precision']:.4f}")
print(f"   Recall:    {metrics['recall']:.4f}")
print(f"   ROC-AUC:   {metrics.get('roc_auc', 'N/A')}")