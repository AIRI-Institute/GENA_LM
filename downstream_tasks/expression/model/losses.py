import torch
import torch.nn as nn

from typing import Self


class CellTypeLoss(nn.Module):
    def __init__(
        self,
        weight_mean,
        loss_fct_mean=nn.MSELoss(reduction="none"),
        loss_fct_deviation=nn.MSELoss(reduction="none"),
        normalize_by_mean=True,
    ):
        super().__init__()
        assert 0 <= weight_mean <= 1, "weight_mean must be between 0 and 1"
        self.loss_fct_mean = loss_fct_mean
        self.loss_fct_deviation = loss_fct_deviation
        self.weight_mean = weight_mean
        self.weight_deviation = 1 - weight_mean
        self.normalize_by_mean = normalize_by_mean

    def forward(
        self, cls_targets, cls_preds, cls_mask, dataset_mean, dataset_deviation
    ):
        # mean across cell types
        cls_targets_mean = (cls_targets * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_targets_mean = cls_targets_mean.reshape(cls_targets_mean.shape[0], 1)
        cls_preds_mean = (cls_preds * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_preds_mean = cls_preds_mean.reshape(cls_preds_mean.shape[0], 1)

        # normalize by mean
        if self.normalize_by_mean:
            cls_targets_deviation = (cls_targets - cls_targets_mean) / cls_targets_mean
            cls_preds_deviation = (cls_preds - cls_preds_mean) / cls_preds_mean
        else:
            cls_targets_deviation = cls_targets - cls_targets_mean
            cls_preds_deviation = cls_preds - cls_preds_mean

        # loss
        cls_loss_mean = (
            self.loss_fct_mean(cls_preds_mean, cls_targets_mean) * cls_mask
        ).sum() / cls_mask.sum()
        cls_loss_deviation = (
            self.loss_fct_deviation(cls_preds_deviation, cls_targets_deviation)
            * cls_mask
        ).sum() / cls_mask.sum()
        full_loss = (
            self.weight_mean * cls_loss_mean
            + self.weight_deviation * cls_loss_deviation
        )

        return full_loss, cls_loss_mean, cls_loss_deviation


class ExpressionCountsLoss(nn.Module):
    def __init__(
        self: Self,
        weight: torch.Tensor | float = 1.0,
        fct_loss_fn: torch.nn.Module | None = nn.MSELoss(reduction="none"),
        cell_type_specific_loss_fn: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.weight: torch.nn.Buffer = torch.nn.Buffer(
            torch.asarray(weight, dtype=torch.float32)
        )

        self.fct_loss_fn = fct_loss_fn
        self.cell_type_specific_loss_fn = cell_type_specific_loss_fn

    def forward(
        self: Self,
        logits: torch.Tensor,
        labels: torch.LongTensor | None,
        labels_mask: torch.BoolTensor | None,
        dataset_mean: torch.Tensor | None = None,
        dataset_deviation: torch.Tensor | None = None,
    ):
        # Loss
        loss = None
        cls_loss = None
        other_loss = None
        mean_loss = None
        deviation_loss = None
        labels_reshaped = None
        labels_mask_reshaped = None

        if labels is not None:
            B, seq_len, N = labels.shape
            assert labels_mask.shape == (B, seq_len, N)

            # labels, labels_mask: (B, seq_len, N)
            # Нужно:   (B*N, seq_len, 1)
            # 1) permute(0,2,1) -> (B, N, seq_len)
            # 2) reshape -> (B*N, seq_len)
            # 3) unsqueeze -> (B*N, seq_len, 1)
            labels_reshaped = (
                labels.permute(0, 2, 1).reshape(B * N, seq_len, 1).to(logits.device)
            )
            labels_mask_reshaped = (
                labels_mask.permute(0, 2, 1)
                .reshape(B * N, seq_len, 1)
                .to(logits.device)
            )

            # loss
            # Cчитаем общий лосс
            unreduced_loss = self.fct_loss_fn(
                logits, labels_reshaped
            )  # (B*N, seq_len, 1)

            if torch.any(labels_mask_reshaped.bool()).cpu().item():
                # Разделяем маску на CLS и остальные токены
                cls_mask = labels_mask_reshaped[:, 0:1, :]  # (B*N, 1, 1)
                other_mask = labels_mask_reshaped[:, 1:, :]  # (B*N, seq_len-1, 1)

                # Считаем лосс для CLS
                cls_loss = None
                if torch.any(cls_mask.bool()).cpu().item():
                    if self.cell_type_specific_loss_fn is not None:
                        cls_loss, mean_loss, deviation_loss = (
                            self.cell_type_specific_loss_fn(
                                cls_targets=labels_reshaped[:, 0:1, :].reshape(B, N),
                                cls_preds=logits[:, 0:1, :].reshape(B, N),
                                cls_mask=cls_mask.reshape(B, N),
                                dataset_mean=dataset_mean,
                                dataset_deviation=dataset_deviation,
                            )
                        )
                    else:
                        cls_loss = (
                            unreduced_loss[:, 0:1, :] * cls_mask
                        ).sum() / cls_mask.sum()
                        mean_loss = None
                        deviation_loss = None

                # Считаем лосс для остальных токенов
                other_loss = None
                if torch.any(other_mask.bool()).cpu().item():
                    other_loss = (
                        unreduced_loss[:, 1:, :] * other_mask
                    ).sum() / other_mask.sum()

                # Объединяем лоссы
                if cls_loss is not None and other_loss is not None:
                    loss = cls_loss + self.weight * other_loss
                elif cls_loss is not None:
                    loss = cls_loss
                elif other_loss is not None:
                    loss = self.weight * other_loss

        return dict(
            loss=loss,
            cls_loss=cls_loss,
            mean_loss=mean_loss,
            other_loss=other_loss,
            deviation_loss=deviation_loss,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
        )
