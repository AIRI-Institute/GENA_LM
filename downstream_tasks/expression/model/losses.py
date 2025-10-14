import torch
import torch.nn as nn

from typing import Self

class CellTypeLoss(nn.Module):
    def __init__(self, 
                 weight_mean,
                 loss_fct_mean = nn.MSELoss(reduction="none"), 
                 loss_fct_deviation = nn.MSELoss(reduction="none"),
                 normalize_by_mean = True,
                 ):
        super().__init__()
        assert 0 <= weight_mean <= 1, "weight_mean must be between 0 and 1"
        self.loss_fct_mean = loss_fct_mean
        self.loss_fct_deviation = loss_fct_deviation
        self.weight_mean = weight_mean
        self.weight_deviation = 1 - weight_mean
        self.normalize_by_mean = normalize_by_mean

    def forward(self, cls_targets, cls_preds, cls_mask,
                dataset_mean,
                dataset_deviation):
        
        # mean across cell types
        cls_targets_mean = (cls_targets * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_targets_mean = cls_targets_mean.reshape(cls_targets_mean.shape[0],1)
        cls_preds_mean = (cls_preds * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_preds_mean = cls_preds_mean.reshape(cls_preds_mean.shape[0],1)

        # normalize by mean
        if self.normalize_by_mean:
            cls_targets_deviation = ((cls_targets - cls_targets_mean) / cls_targets_mean)
            cls_preds_deviation = ((cls_preds - cls_preds_mean) / cls_preds_mean)
            debug_test_cls_targets_deviation = cls_targets_deviation
        else:
            cls_targets_deviation = (cls_targets - cls_targets_mean)
            cls_preds_deviation = (cls_preds - cls_preds_mean)
            debug_test_cls_targets_deviation = ((cls_targets - cls_targets_mean) / cls_targets_mean)

        # loss
        cls_loss_mean = (self.loss_fct_mean(cls_preds_mean, cls_targets_mean) * cls_mask).sum() / cls_mask.sum()
        cls_loss_deviation = (self.loss_fct_deviation(cls_preds_deviation, cls_targets_deviation) * cls_mask).sum() / cls_mask.sum()
        full_loss = self.weight_mean * cls_loss_mean + self.weight_deviation * cls_loss_deviation

        return full_loss, cls_loss_mean, cls_loss_deviation

class ExpressionCountsLoss(nn.Module):
    def __init__(self: Self,
                 weight: torch.Tensor | float = 1.0,
                 cell_type_specific_loss_fn: torch.nn.Module = None) -> None:
        super().__init__()

        self.zero: torch.nn.Buffer = torch.nn.Buffer(
            torch.asarray(0.0, dtype = torch.float32)
        )

        self.weight: torch.nn.Buffer = torch.nn.Buffer(
            torch.asarray(weight, dtype = torch.float32)
        )

        self.cell_type_specific_loss_fn = cell_type_specific_loss_fn

    def forward(self: Self,
                logits: torch.Tensor,
                labels: torch.LongTensor | None,
                labels_mask: torch.BoolTensor | None,
                dataset_mean: torch.Tensor | None = None,
                dataset_deviation: torch.Tensor | None = None,):
        if labels is None:
            assert labels_mask is None
            return (self.zero.clone())

        B, seq_len, N = labels.shape
        assert labels_mash.shape == (B, seq_len, N)

        device: torch.device = logits.device
        loss: torch.Tensor = self.zero.clone()

        # labels, labels_mask: (B, seq_len, N)
        # Нужно:   (B*N, seq_len, 1)
        # 1) permute(0,2,1) -> (B, N, seq_len)
        # 2) reshape -> (B*N, seq_len)
        # 3) unsqueeze -> (B*N, seq_len, 1)
        def reshape_labels(values: torch.Tensor) -> torch.Tensor:
            return values.permute(0, 2, 1).reshape(B * N, seq_len, 1).to(device)

        labels_reshaped: torch.Tensor = reshape_labels(labels)
        labels_mask_reshaped: torch.Tensor = reshape_labels(labels_mask)

        # loss
        # Cчитаем общий лосс
        unreduced_loss: torch.Tensor = self.loss_fct(logits, labels_reshaped)  # (B*N, seq_len, 1)

        if torch.any(labels_mask_reshaped).cpu().item():
            # Разделяем маску на CLS и остальные токены
            cls_mask = labels_mask_reshaped[:, :1, :]  # (B*N, 1, 1)
            other_mask = labels_mask_reshaped[:, 1:, :]  # (B*N, seq_len-1, 1)

            # Считаем лосс для CLS  
            cls_loss = self.zero.clone()

            if torch.any(cls_mask).cpu().item():
                if not (self.cell_type_specific_loss_fn is None):
                    cls_loss, mean_loss, deviation_loss = self.cell_type_specific_loss_fn(
                        cls_targets = labels_reshaped[:, :1, :].reshape(B,N),
                        cls_preds = logits[:, :1, :].reshape(B,N),
                        cls_mask = cls_mask.reshape(B,N),
                        dataset_mean = dataset_mean,
                        dataset_deviation = dataset_deviation
                    )
                else:
                    denom: torch.Tensor = torch.sum(cls_mask)
                    cls_loss = (unreduced_loss[:, :1, :] * cls_mask).sum() / denom

                # Считаем лосс для остальных токенов
                other_loss = self.zero.clone()
                if torch.any(other_mask).cpu().item():
                    denom: torch.Tensor = torch.sum(other_mask)
                    other_loss = (unreduced_loss[:, 1:, :] * other_mask).sum() / denom

                
                loss = cls_loss + self.weight * other_loss

        return dict(
            loss = loss,
            cls_loss = cls_loss,
            other_loss = other_loss,
            mean_loss = mean_loss,
            deviation_loss = deviation_loss,
        )
