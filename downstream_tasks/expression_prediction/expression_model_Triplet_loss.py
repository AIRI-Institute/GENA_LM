import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.utils import cached_file
from transformers import BertConfig
from typing import Optional

from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel


class ExpressionCounts(BertPreTrainedModel):
    """
    Регрессия по токенам + усиленный TripletLoss по CLS для различий между N описаниями одного гена.

    Предполагаем, что labels_mask = 1 везде (ты так сказал), но код оставлен универсальным.
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
        bert_cpt: str = '/mnt/nfs_dna/DNALM/trained_models/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best.pth',
        model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
        # --- Triplet гиперпараметры (могут быть None -> авто по батчу) ---
        triplet_alpha: float = 20,      # усиливаем влияние различий
        triplet_margin: Optional[float] = None,
        delta_pos: Optional[float] = None,
        delta_neg: Optional[float] = None,
    ):
        super().__init__(config)
        self.config = config
        self.hidden_size_desc = hidden_size_desc

        # 1) BERT
        cfg = BertConfig.from_pretrained(model_name)
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

        # 2) MLP для desc_vectors
        self.desc_fc = nn.Sequential(
            nn.Linear(self.hidden_size_desc, cfg.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )

        # 3) Encoder
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

        # 5) Базовые лоссы
        self.loss_fct = loss_fct
        self.weight = weight

        # Triplet гиперпараметры (могут быть None -> авто расчёт внутри лосса)
        self.triplet_alpha = triplet_alpha
        self.triplet_margin = triplet_margin
        self.delta_pos = delta_pos
        self.delta_neg = delta_neg

        self.post_init()

    # -------- TripletLoss helper ---------
    def _triplet_loss_cls(self,
                           preds: torch.Tensor,
                           targets: torch.Tensor,
                           group_size: int,
                           margin: Optional[float],
                           delta_pos: Optional[float],
                           delta_neg: Optional[float]) -> torch.Tensor:
        """
        preds, targets: (B*N,) — скалярные значения CLS.
        group_size = N.
        Если margin/delta_pos/delta_neg = None, вычисляем их по батчу:
            R = max(y) - min(y)
            margin    = 0.07 * R
            delta_pos = 0.01 * R
            delta_neg = 0.20 * R
        (Если R=0, fallback на std.)
        """
        device = preds.device
        total = preds.numel()
        assert total % group_size == 0, "(B*N) должно делиться на N"
        B = total // group_size

        # --- авто-гиперы ---
        if targets.numel() > 1:
            R = (targets.max() - targets.min()).item()
            sigma = targets.std().item()
        else:
            R = 0.0
            sigma = 0.0

        if margin is None:
            margin = 0.07 * R if R > 0 else 0.3 * sigma
        if delta_pos is None:
            delta_pos = 0.01 * R if R > 0 else 0.05 * sigma
        if delta_neg is None:
            delta_neg = 0.20 * R if R > 0 else 0.5 * sigma

        delta_neg = max(delta_neg, delta_pos + 1e-6)

        losses = []
        for b in range(B):
            s = b * group_size
            e = s + group_size
            p_b = preds[s:e]
            t_b = targets[s:e]

            if t_b.unique().numel() < 2:
                continue

            abs_diff = (t_b.unsqueeze(0) - t_b.unsqueeze(1)).abs()  # (N,N)
            for a_idx in range(group_size):
                pos_mask = (abs_diff[a_idx] > delta_pos) & (abs_diff[a_idx] < delta_neg)
                neg_mask = (abs_diff[a_idx] >= delta_neg)
                if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                    continue

                p_idx = torch.argmin(abs_diff[a_idx].masked_fill(~pos_mask, float('inf')))
                n_idx = torch.argmax(abs_diff[a_idx].masked_fill(~neg_mask, float('-inf')))

                d_ap = (p_b[a_idx] - p_b[p_idx]).abs()
                d_an = (p_b[a_idx] - p_b[n_idx]).abs()
                losses.append(F.relu(d_ap - d_an + margin))

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    # ------------- forward -------------
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
        seq_out_expanded = sequence_output.unsqueeze(1).expand(-1, N, -1, -1).contiguous()  # (B,N,seq_len,H)

        # 3) desc_vectors через MLP и добавляем к CLS
        desc_fc = self.desc_fc(desc_vectors.reshape(B * N, self.hidden_size_desc))  # (B*N,H)
        desc_fc = desc_fc.reshape(B, N, hidden_size)
        seq_out_expanded[:, :, 0, :] = seq_out_expanded[:, :, 0, :].clone() + desc_fc

        # 4) схлопнуть B и N
        seq_out_flat = seq_out_expanded.view(B * N, seq_len, hidden_size)

        # 5) Encoder
        enc_out = self.transformer_encoder(seq_out_flat)  # (B*N,seq_len,H)

        # 6) Head
        logits = self.activation(self.classifier(enc_out))  # (B*N,seq_len,1)

        # -------- Loss ---------
        loss = None
        cls_loss = other_loss = triplet_loss = None
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

            # Triplet по CLS
            cls_pred_all = logits[:, 0, 0]
            cls_true_all = labels_reshaped[:, 0, 0]
            triplet_loss = self._triplet_loss_cls(
                cls_pred_all, cls_true_all,
                group_size=N,
                margin=self.triplet_margin,
                delta_pos=self.delta_pos,
                delta_neg=self.delta_neg,
            )
            loss = loss + self.triplet_alpha * triplet_loss if loss is not None else self.triplet_alpha * triplet_loss

        if not return_dict:
            return (loss, logits)

        extra = {
            'cls_loss': cls_loss,
            'other_loss': other_loss,
            'triplet_loss': triplet_loss,
        }

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions
        ), labels_reshaped, labels_mask_reshaped, cls_loss, triplet_loss
