import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import cached_file
from transformers import BertConfig

from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel


class ExpressionCounts(BertPreTrainedModel):
    """
    Модель для предсказания экспрессии по токенам.
    Базовый лосс: MSE (по CLS и остальным токенам).
    Доп. лосс: корреляционный (1 - Pearson r) по CLS внутри каждой группы из N описаний одного гена.
    Корреляция считается только если внутри группы есть разброс меток.
    """

    def __init__(
        self,
        config,
        loss_fct: nn.Module = nn.MSELoss(reduction="none"),
        activation: nn.Module = nn.Identity(),
        hidden_size_desc: int = 768,
        hidden_ff: int = 1024,
        num_encoder_layers: int = 3,
        nhead: int = 8,
        weight: float = 1.0,
        model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
        # --- Корреляционный лосс ---
        corr_alpha: float = 0.8,   # вес корреляции в суммарном лоссе
        var_thr: float = 1,     # минимум дисперсии в группе, иначе игнор
        eps: float = 1e-8,         # численный стабилизатор
    ):
        super().__init__(config)
        self.config = config
        self.hidden_size_desc = hidden_size_desc

        # 1) BERT backbone
        cfg = BertConfig.from_pretrained("AIRI-Institute/gena-lm-bert-large-t2t")
        self.bert = BertModel(cfg)
        weights_path = cached_file(model_name, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")
        updated_state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items() if k.startswith("bert.")}
        missing_keys, unexpected_keys = self.bert.load_state_dict(updated_state_dict, strict=False)
        if missing_keys:
            print("[ExpressionCounts] Missing keys:")
            for k in missing_keys: print(" -", k)
        if unexpected_keys:
            print("[ExpressionCounts] Unexpected keys:")
            for k in unexpected_keys: print(" -", k)

        # 2) Проекция desc_vectors
        self.desc_fc = nn.Sequential(
            nn.Linear(self.hidden_size_desc, cfg.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )

        # 3) Доп. encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_ff,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        # 4) Head
        self.classifier = nn.Linear(cfg.hidden_size, 1)
        self.activation = activation

        # 5) Лоссы/параметры
        self.loss_fct = loss_fct
        self.weight = weight
        self.corr_alpha = corr_alpha
        self.var_thr = var_thr
        self.eps = eps

        self.post_init()

    # ---------- Корреляционный лосс по группам ----------
    @staticmethod
    def _group_corr_loss(preds: torch.Tensor,
                         targets: torch.Tensor,
                         group_size: int,
                         var_thr: float,
                         eps: float) -> torch.Tensor:
        """
        preds, targets: (B*N,) значения CLS
        Для каждой группы N считаем 1 - Pearson r(preds_group, targets_group)
        Возвращаем среднее по группам с достаточной дисперсией.
        """
        device = preds.device
        total = preds.numel()
        assert total % group_size == 0, "(B*N) должно делиться на N"
        B = total // group_size

        losses = []
        for b in range(B):
            s, e = b * group_size, (b + 1) * group_size
            p_b = preds[s:e]
            t_b = targets[s:e]

            if t_b.var() <= var_thr or t_b.unique().numel() < 2:
                continue

            p_center = p_b - p_b.mean()
            t_center = t_b - t_b.mean()
            num = (p_center * t_center).sum()
            den = (p_center.pow(2).sum().sqrt() * t_center.pow(2).sum().sqrt() + eps)
            rho = num / den
            losses.append(1.0 - rho)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    # ------------------- forward -------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        meta_input_ids=None,
        meta_attention_mask=None,
        desc_vectors=None,
        dataset_mean=None,
        dataset_deviation=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1) BERT
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        sequence_output = bert_outputs.last_hidden_state  # (B, seq_len, hidden)
        B, seq_len, hidden_size = sequence_output.shape
        assert desc_vectors is not None, "desc_vectors обязателен"
        N = desc_vectors.shape[1]

        # 2) расширяем по N
        seq_out_expanded = sequence_output.unsqueeze(1).expand(-1, N, -1, -1).contiguous()

        # 3) desc_vectors -> MLP -> добавляем к CLS
        desc_fc = self.desc_fc(desc_vectors.reshape(B * N, self.hidden_size_desc)).reshape(B, N, hidden_size)
        seq_out_expanded[:, :, 0, :] = seq_out_expanded[:, :, 0, :].clone() + desc_fc

        # 4) схлопываем
        seq_out_flat = seq_out_expanded.view(B * N, seq_len, hidden_size)

        # 5) Encoder
        enc_out = self.transformer_encoder(seq_out_flat)

        # 6) Head
        logits = self.activation(self.classifier(enc_out))  # (B*N, seq_len, 1)

        # --------- ЛОССЫ ---------
        loss = None
        cls_loss = other_loss = corr_loss = None
        labels_reshaped = labels_mask_reshaped = None

        if labels is not None:
            labels_reshaped = labels.permute(0, 2, 1).reshape(B * N, seq_len, 1).to(logits.device)
            if labels_mask is not None:
                labels_mask_reshaped = labels_mask.permute(0, 2, 1).reshape(B * N, seq_len, 1).to(logits.device)
            else:
                labels_mask_reshaped = torch.ones_like(labels_reshaped, dtype=torch.bool)

            unreduced = self.loss_fct(logits, labels_reshaped)
            cls_mask = labels_mask_reshaped[:, 0:1, :]
            other_mask = labels_mask_reshaped[:, 1:, :]

            if cls_mask.sum() > 0:
                cls_loss = (unreduced[:, 0:1, :] * cls_mask).sum() / cls_mask.sum()
            if other_mask.sum() > 0:
                other_loss = (unreduced[:, 1:, :] * other_mask).sum() / other_mask.sum()

            if cls_loss is not None and other_loss is not None:
                loss = cls_loss + self.weight * other_loss
            elif cls_loss is not None:
                loss = cls_loss
            elif other_loss is not None:
                loss = self.weight * other_loss

            # Корреляция по CLS внутри групп N
            cls_pred_all = logits[:, 0, 0]
            cls_true_all = labels_reshaped[:, 0, 0]

            corr_loss = self._group_corr_loss(
                cls_pred_all, cls_true_all,
                group_size=N,
                var_thr=self.var_thr,
                eps=self.eps,
            )

            if corr_loss is not None and corr_loss > 0:
                loss = (1 - self.corr_alpha) * loss + self.corr_alpha * corr_loss if loss is not None else self.corr_alpha * corr_loss

        if not return_dict:
            return (loss, logits)


        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions
        ), labels_reshaped, labels_mask_reshaped, cls_loss, corr_loss 
