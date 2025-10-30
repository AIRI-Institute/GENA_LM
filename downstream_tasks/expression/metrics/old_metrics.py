import pandas as pd
import numpy as np
import torch

from copy import copy

import accelerate
from logging import getLogger
from itertools import chain, compress

from .score_ct_specificity import score_predictions
from .calculate_metrics import calculate_target_genes_metrics

logger = getLogger(__name__)

DEFAULT_TARGET_GENES = ['ENCFF123KIW', 'ENCFF784MDF', 'ENCFF236XOK', 'ENCFF242BWW']

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
        'dataset_mean': batch['dataset_mean'],
        'dataset_deviation': batch['dataset_deviation'],
        'selected_keys': batch['selected_keys'],
        'dataset_description': batch['dataset_description']
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
    loss_components = ['cls_loss', 'other_loss', 'mean_loss', 'deviation_loss']
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


class OldMetrics:
    def __init__(self,
                 model_path = None,
                 selected_targets_path = None,
                 save_predictions=False, 
                 target_genes = DEFAULT_TARGET_GENES,
                 target_metric = "pearson_corr"):
        super().__init__()
        self.model_path = model_path
        self.selected_targets_path = selected_targets_path
        self.save_predictions = save_predictions
        self.target_metric = target_metric
        self.target_genes = target_genes
        self.data = dict()

    def reset(self):
        self.data = dict()
        return self

    def update(self, batch, outputs):
        transformed = batch_transform_fn(batch)
        data = keep_for_metrics_fn(transformed, outputs)

        for k in sorted(data.keys()):
            value = data[k] 
            if torch.is_tensor(value):
                value = value.detach().cpu()
            if not isinstance(value, list):
                value = [value]
            self.data[k] = [*self.data.get(k, []), *value]

    def compute(self):
        data_keys = set(accelerate.utils.gather_object(list(self.data.keys())))
        data_keys = sorted(data_keys)

        metrics_data = dict()

        count_data = accelerate.utils.gather_object(self.data["_count"])
        metrics_data["_count"] = torch.cat(count_data)

        pred_data = accelerate.utils.gather_object(self.data["_pred"])
        metrics_data["_pred"] = torch.cat(pred_data)

        pred_squared_data = accelerate.utils.gather_object(self.data["_pred_squared"])
        metrics_data["_pred_squared"] = torch.cat(pred_squared_data)

        product_data = accelerate.utils.gather_object(self.data["_product"])
        metrics_data["_product"] = torch.cat(product_data)

        true_data = accelerate.utils.gather_object(self.data["_true"])
        metrics_data["_true"] = torch.cat(true_data)

        true_squared_data = accelerate.utils.gather_object(self.data["_true_squared"])
        metrics_data["_true_squared"] = torch.cat(true_squared_data)

        dataset_description_data = accelerate.utils.gather_object(self.data["dataset_description"])
        metrics_data["dataset_description"] = list(dataset_description_data)

        gene_id_data = accelerate.utils.gather_object(self.data["gene_id"])
        metrics_data["gene_id"] = list(gene_id_data)

        keys_id_data = accelerate.utils.gather_object(self.data["keys_id"])
        metrics_data["keys_id"] = list(keys_id_data)

        loss_cls_loss_data = accelerate.utils.gather_object(self.data["loss_cls_loss"])
        loss_cls_loss_data = list(loss[None] for loss in loss_cls_loss_data)
        metrics_data["loss_cls_loss"] = torch.cat(loss_cls_loss_data)
        
        tpm_preds_data = accelerate.utils.gather_object(self.data["tpm_preds"])
        metrics_data["tpm_preds"] = list(tpm_preds_data)

        tpm_true_data = accelerate.utils.gather_object(self.data["tpm_true"])
        metrics_data["tpm_true"] = list(tpm_true_data)

        result = self.metrics_fn(metrics_data)
        self.reset()
        return result

    def metrics_fn(self, data):
        metrics = {}
        
        data['_product'] = torch.sum(data['_product'], dim=0)
        data['_true'] = torch.sum(data['_true'], dim=0)
        data['_true_squared'] = torch.sum(data['_true_squared'], dim=0)
        data['_pred'] = torch.sum(data['_pred'], dim=0)
        data['_pred_squared'] = torch.sum(data['_pred_squared'], dim=0)
        data['_count'] = torch.sum(data['_count'], dim=0)
        
        true_mean = data['_true'] / data['_count']
        pred_mean = data['_pred'] / data['_count']
        covariance = (data['_product'] - true_mean * data['_pred'] - pred_mean * data['_true'] + 
                     data['_count'] * true_mean * pred_mean)
        true_var = data['_true_squared'] - data['_count'] * torch.square(true_mean)
        pred_var = data['_pred_squared'] - data['_count'] * torch.square(pred_mean)
        tp_var = torch.sqrt(true_var) * torch.sqrt(pred_var)
        corr_coef = covariance / tp_var
        metrics['pearson_corr'] = corr_coef.item()
        
        loss_components = ['cls_loss', 'other_loss', 'mean_loss', 'deviation_loss']
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
                        logger.error(f"dataset {dataset_desc} gene {gene} has all predicted values the same")
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
                target_metrics = calculate_target_genes_metrics(df_true, df_pred, self.target_genes, logger=logger)
                metrics[f'pearson_corr_4_cells_borzoi_cells_{dataset_desc}'] = target_metrics['corr_selected_cell_types']
                metrics[f'pearson_corr_4_cells_borzoi_genes_{dataset_desc}'] = target_metrics['corr_selected_genes']
                
                # клетоспецифичность
                df_true = df_true.reset_index()
                df_pred = df_pred.reset_index()
                score = score_predictions(
                    df_true,
                    df_pred,
                    self.selected_targets_path,
                    need_log=False,
                    logger=logger
                )
                if score and score.get('deviation_r', None):
                    metrics[f'score_predictions_{dataset_desc}'] = score['deviation_r']
        return metrics

