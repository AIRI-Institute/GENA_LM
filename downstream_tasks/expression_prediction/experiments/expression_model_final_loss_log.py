import os
import json
import math
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModel, BertConfig, ModernBertModel
from transformers.utils import cached_file
from transformers.utils import logging as hf_logging

from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel

hf_logging.set_verbosity_info()


class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)


@dataclass
class ExpressionModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None
    deviation_loss: Optional[torch.FloatTensor] = None
    multinomial_loss: Optional[torch.FloatTensor] = None


def cls_deviation_from_mean_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor,
    n_keys: int,
) -> torch.Tensor:
    """
    Loss on matching deviations from the mean: deviations of logits from the group mean
    should match deviations of labels from the group mean (CLS only, mask=1 only).

    logits: (B*N, L, 1)
    labels: (B*N, L, 1)
    labels_mask: (B*N, L, 1)
    n_keys: N — group size
    """
    B = logits.shape[0] // n_keys
    cls_logits = logits[:, 0:1, :]        # (B*N, 1, 1)
    cls_labels = labels[:, 0:1, :]        # (B*N, 1, 1)
    cls_mask = labels_mask[:, 0:1, :]     # (B*N, 1, 1)

    cls_logits = cls_logits.view(B, n_keys, 1, 1)  # (B, N, 1, 1)
    cls_labels = cls_labels.view(B, n_keys, 1, 1)  # (B, N, 1, 1)
    cls_mask = cls_mask.view(B, n_keys, 1, 1)      # (B, N, 1, 1)

    cnt = cls_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1, 1, 1)
    mean_logits = (cls_logits * cls_mask).sum(dim=1, keepdim=True) / cnt
    mean_labels = (cls_labels * cls_mask).sum(dim=1, keepdim=True) / cnt

    logit_dev = (cls_logits - mean_logits) * cls_mask
    label_dev = (cls_labels - mean_labels) * cls_mask
    dev_sq = (logit_dev - label_dev).pow(2)

    loss_per_group = dev_sq.sum(dim=1) / cnt.squeeze(1)  # (B, 1, 1)
    group_has_mask = (cls_mask.sum(dim=1) > 0).squeeze()  # (B,)
    if group_has_mask.sum() == 0:
        return logits.new_zeros(())
    return loss_per_group[group_has_mask].mean()


def cls_multinomial_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor,
    n_keys: int,
    min_target_max: float = 2.0,
) -> torch.Tensor:
    """
    Multinomial (NLL) loss over each n_keys for CLS token only.
    logits:       (B*N, L, 1)
    labels:       (B*N, L, 1)
    labels_mask:  (B*N, L, 1)
    n_keys:       N
    """
    B = logits.shape[0] // n_keys

    cls_logits = logits[:, 0:1, :].squeeze(-1).squeeze(-1)      # (B*N,)
    cls_labels = labels[:, 0:1, :].squeeze(-1).squeeze(-1)      # (B*N,)
    cls_mask   = labels_mask[:, 0:1, :].squeeze(-1).squeeze(-1) # (B*N,)

    cls_logits = cls_logits.view(B, n_keys)
    cls_labels = cls_labels.view(B, n_keys)
    cls_mask   = cls_mask.view(B, n_keys)

    mask_bool = cls_mask > 0
    group_has_mask = mask_bool.any(dim=1)  # (B,)
    if group_has_mask.sum() == 0:
        return logits.new_zeros(())

    target_max = (cls_labels * cls_mask).max(dim=1).values  # (B,)
    keep = group_has_mask & (target_max > min_target_max)
    if keep.sum() == 0:
        return logits.new_zeros(())

    cls_logits = cls_logits[keep]
    cls_labels = cls_labels[keep]
    cls_mask   = cls_mask[keep]
    mask_bool  = mask_bool[keep]

    eps = 1e-8
    w = cls_logits.masked_fill(~mask_bool, 0.0)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)

    p = w / w_sum
    log_probs = torch.log(p.clamp(min=eps))
    log_probs = log_probs.masked_fill(~mask_bool, 0.0)

    target = cls_labels * cls_mask
    target_sum = target.sum(dim=1, keepdim=True).clamp(min=eps)
    target = target / target_sum

    nll_per_group = -(target * log_probs).sum(dim=1)
    # Оставляю логику как у тебя (хотя group_has_mask тут уже не той формы при keep)
    return nll_per_group[group_has_mask].mean()


class ExpressionCounts(nn.Module):
    """
    Expected shapes:
      - input_ids:      (B*N, L)
      - attention_mask: (B*N, L)
      - desc_input_ids: (B, N, D) or (B*N, D)
      - desc_attention_mask: (B, N, D) or (B*N, D)
      - dataset_flag:   (B, N)
      - labels:         (B*N, L, 1)
      - labels_mask:    (B*N, L, 1)
    """

    def __init__(
        self,
        config,
        hf_model_name_decoder,
        loss_fct=nn.MSELoss(reduction="none"),
        activation=nn.Identity(),
        num_encoder_layers=3,
        nhead=8,
        weight=1,
        hidden_ff=1024,
        bert_cpt='/mnt/nfs_dna/DNALM/trained_models/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best.pth',
        hf: bool = False,
        hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
        desc_model_name: str = "intfloat/multilingual-e5-large-instruct",
        use_deviation_loss: bool = False,
        use_multinomial_loss: bool = False,
        weight_deviation_loss: float = 1.0,
        weight_multinomial_loss: float = 1.0,

        # существующий debug print
        debug_print_norms: bool = True,
        debug_max_print: int = 64,

        # ===== file logging options =====
        enable_file_logging: bool = True,
        log_path: str = "/home/jovyan/shares/SR003.nfs2/aspeedok/GENA_LM/downstream_tasks/expression_prediction/train_debug.jsonl",
        log_every_n_steps: int = 1,          # per-sample logs frequency
        log_grads_every_n_steps: int = 1,    # grad logs frequency
        log_param_grads: bool = False,       # WARNING: huge if True
        is_main_process: Optional[bool] = None,  # если задан — используется вместо _is_main_process() (надёжно при DDP)
    ):
        super().__init__()

        self.debug_print_norms = debug_print_norms
        self.debug_max_print = int(debug_max_print)

        # logging: для открытия файла используем переданный is_main_process или fallback на torch.distributed
        self.enable_file_logging = bool(enable_file_logging)
        self.log_path = log_path
        self.log_every_n_steps = int(log_every_n_steps)
        self.log_grads_every_n_steps = int(log_grads_every_n_steps)
        self.log_param_grads = bool(log_param_grads)
        self._log_fh = None
        self._is_main = is_main_process if is_main_process is not None else self._is_main_process()

        if self.enable_file_logging and self._is_main and self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            self._log_fh = open(self.log_path, "a", encoding="utf-8", buffering=1)
            self._log_fh.write(json.dumps({
                "type": "run_start",
                "ts": datetime.utcnow().isoformat() + "Z",
                "desc": "ExpressionCounts debug log started"
            }, ensure_ascii=False) + "\n")

        updated_state_dict = None

        # 1) DNA model (GENA)
        if hf:
            if "modernbert" in hf_model_name.lower():
                print(f"Using ModernBERT from {hf_model_name}")
                self.bert, info = ModernBertModel.from_pretrained(
                    hf_model_name,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    output_loading_info=True
                )
                config = self.bert.config
                print("missing:", len(info["missing_keys"]), info["missing_keys"][:10])
                print("unexpected:", len(info["unexpected_keys"]), info["unexpected_keys"][:10])
                print("mismatched:", info.get("mismatched_keys", [])[:5])
            else:
                hf_config = BertConfig.from_pretrained(hf_model_name)
                self.bert = BertModel(hf_config, add_pooling_layer=False)
                weights_path = cached_file(hf_model_name, "pytorch_model.bin")
                state_dict = torch.load(weights_path, map_location="cpu")
                updated_state_dict = {
                    k.replace("bert.", ""): v for k, v in state_dict.items()
                    if k.startswith("bert.")
                }
                config = hf_config
        else:
            self.bert = BertModel(config, add_pooling_layer=False)
            checkpoint = torch.load(bert_cpt, map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            updated_state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items()}
            missing_k, unexpected_k = self.bert.load_state_dict(updated_state_dict, strict=False)

        if updated_state_dict is not None:
            missing_k, unexpected_k = self.bert.load_state_dict(updated_state_dict, strict=False)
            if len(missing_k) != 0:
                print(f"{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.")
            if len(unexpected_k) != 0:
                print(f"{unexpected_k} were found in checkpoint, but model is not expecting them!")

        self.config = config

        # 2) Description model
        self.desc_model_name = desc_model_name
        self.desc_model = AutoModel.from_pretrained(
            self.desc_model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        )
        for p in self.desc_model.parameters():
            p.requires_grad = False

        # 3) dims
        self.gen_hidden_size = config.hidden_size
        self.desc_hidden_size = self.desc_model.config.hidden_size

        # 4) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_ff,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 5) Loss config
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight
        self.use_deviation_loss = use_deviation_loss
        self.use_multinomial_loss = use_multinomial_loss
        self.weight_deviation_loss = weight_deviation_loss
        self.weight_multinomial_loss = weight_multinomial_loss

        dtype = next(self.bert.parameters()).dtype
        device = next(self.bert.parameters()).device

        self.desc_proj = nn.Sequential(
            nn.Linear(self.desc_hidden_size, self.gen_hidden_size, device=device, dtype=dtype),
            nn.GELU(),
        )
        self.desc_ln = nn.LayerNorm(self.gen_hidden_size, eps=1e-5, device=device, dtype=dtype)

        # 6) Classifier
        self.classifier = nn.Linear(config.hidden_size, 1, device=device, dtype=dtype)

    def __del__(self):
        try:
            self.close_log()
        except Exception:
            pass

    def close_log(self):
        if getattr(self, "_log_fh", None) is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            finally:
                self._log_fh = None

    @staticmethod
    def _is_main_process():
        return (
            (not torch.distributed.is_available())
            or (not torch.distributed.is_initialized())
            or (torch.distributed.get_rank() == 0)
        )

    def _write_jsonl_lines(self, lines):
        if self._log_fh is None:
            return
        self._log_fh.write("\n".join(lines) + "\n")

    def _write_jsonl_record(self, rec: dict):
        if self._log_fh is None:
            return
        self._log_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _masked_mean_no_cls(x: torch.Tensor, mask: Optional[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
        """
        mean over tokens 1: (без CLS)
        x: (B*N, L, H)
        mask: (B*N, L)
        returns: (B*N, H)
        """
        if x.size(1) <= 1:
            return x.mean(dim=1)

        x = x[:, 1:, :]  # drop CLS
        if mask is None:
            return x.mean(dim=1)

        mask = mask[:, 1:]  # drop CLS mask too
        m = mask.unsqueeze(-1).to(dtype=x.dtype)
        denom = m.sum(dim=1).clamp_min(eps)
        return (x * m).sum(dim=1) / denom

    # ===================== GRADIENT LOGGING =====================
    @torch.no_grad()
    def log_gradients(self, global_step: int):
        """
        Вызывать ИЗ train loop ПОСЛЕ loss.backward().
        Пишет summary по градам (global + по модулям) в тот же jsonl.
        """
        if not (self.enable_file_logging and self._log_fh is not None):
            return
        if self.log_grads_every_n_steps <= 0:
            return
        if global_step is None:
            return
        if global_step % self.log_grads_every_n_steps != 0:
            return

        def stats_for_params(named_params):
            total_sq = 0.0
            max_abs = 0.0
            n_tensors = 0
            n_none = 0
            n_nonfinite = 0
            for name, p in named_params:
                if (p is None) or (not p.requires_grad):
                    continue
                if p.grad is None:
                    n_none += 1
                    continue
                g = p.grad.detach()
                n_tensors += 1
                gf = g.float()
                if not torch.isfinite(gf).all():
                    n_nonfinite += 1
                gn = float(torch.norm(gf, p=2).item())
                total_sq += gn * gn
                ma = float(gf.abs().max().item())
                if ma > max_abs:
                    max_abs = ma
            total = math.sqrt(total_sq) if total_sq > 0 else 0.0
            return {
                "grad_l2": total,
                "grad_absmax": max_abs,
                "grad_tensors": int(n_tensors),
                "grad_none": int(n_none),
                "grad_nonfinite_tensors": int(n_nonfinite),
            }

        def param_norm_stats(named_params):
            total_sq = 0.0
            max_abs = 0.0
            n_tensors = 0
            for _, p in named_params:
                if (p is None) or (not p.requires_grad):
                    continue
                n_tensors += 1
                pf = p.detach().float()
                pn = float(torch.norm(pf, p=2).item())
                total_sq += pn * pn
                ma = float(pf.abs().max().item())
                if ma > max_abs:
                    max_abs = ma
            total = math.sqrt(total_sq) if total_sq > 0 else 0.0
            return {
                "param_l2": total,
                "param_absmax": max_abs,
                "param_tensors": int(n_tensors),
            }

        # global
        all_named = list(self.named_parameters())
        gstats = stats_for_params(all_named)
        pstats = param_norm_stats(all_named)

        self._write_jsonl_record({
            "type": "grad_global",
            "step": int(global_step),
            **gstats,
            **pstats,
        })

        # per-module summaries
        modules = {
            "bert": self.bert,
            "desc_proj": self.desc_proj,
            "desc_ln": self.desc_ln,
            "transformer_encoder": self.transformer_encoder,
            "classifier": self.classifier,
        }
        for mname, mod in modules.items():
            named = list(mod.named_parameters(recurse=True))
            g = stats_for_params(named)
            p = param_norm_stats(named)
            self._write_jsonl_record({
                "type": "grad_module",
                "step": int(global_step),
                "module": mname,
                **g,
                **p,
            })

        # OPTIONAL: per-parameter (очень много строк!)
        if self.log_param_grads:
            lines = []
            for name, p in all_named:
                if (p is None) or (not p.requires_grad):
                    continue
                if p.grad is None:
                    rec = {"type": "grad_param", "step": int(global_step), "name": name, "has_grad": False}
                    lines.append(json.dumps(rec, ensure_ascii=False))
                    continue
                gf = p.grad.detach().float()
                rec = {
                    "type": "grad_param",
                    "step": int(global_step),
                    "name": name,
                    "has_grad": True,
                    "grad_l2": float(torch.norm(gf, p=2).item()),
                    "grad_absmax": float(gf.abs().max().item()),
                    "grad_nonfinite": (not torch.isfinite(gf).all().item()),
                }
                lines.append(json.dumps(rec, ensure_ascii=False))
            self._write_jsonl_lines(lines)

    # ===================== FORWARD =====================
    def forward(
        self,
        input_ids=None,              # (B*N, L) or (B, N, L)
        attention_mask=None,         # (B*N, L) or (B, N, L)
        labels_mask=None,            # (B*N, L, 1) or (B, N, L, 1)
        labels=None,                 # (B*N, L, 1) or (B, N, L, 1)
        return_dict=None,
        desc_input_ids=None,         # (B*N, D) or (B, N, D)
        desc_attention_mask=None,    # (B*N, D) or (B, N, D)
        dataset_flag=None,           # (B, N)

        # logging metadata
        global_step: Optional[int] = None,
        sample_ids: Optional[torch.Tensor] = None,  # (B*N,) if you have it
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if dataset_flag is None:
            raise ValueError("dataset_flag must be provided and shaped (B, N)")

        B, N = dataset_flag.shape

        # 1) Reshape inputs if (B, N, ...)
        if input_ids is not None and input_ids.dim() == 3:
            if input_ids.shape[:2] != (B, N):
                raise ValueError(f"input_ids has shape {tuple(input_ids.shape)}, but dataset_flag is {(B, N)}")
            input_ids = input_ids.reshape(B * N, input_ids.shape[-1])

        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.reshape(B * N, attention_mask.shape[-1])

        if labels is not None and labels.dim() == 4:
            labels = labels.reshape(B * N, labels.shape[-2], labels.shape[-1])

        if labels_mask is not None and labels_mask.dim() == 4:
            labels_mask = labels_mask.reshape(B * N, labels_mask.shape[-2], labels_mask.shape[-1])

        if desc_input_ids is not None and desc_input_ids.dim() == 3:
            desc_input_ids = desc_input_ids.reshape(B * N, desc_input_ids.shape[-1])

        if desc_attention_mask is not None and desc_attention_mask.dim() == 3:
            desc_attention_mask = desc_attention_mask.reshape(B * N, desc_attention_mask.shape[-1])

        # 2) DNA model, remove duplicates
        src = input_ids
        if src is None:
            raise ValueError("input_ids must be provided")
        device = src.device
        BxN, seq_len = src.shape[:2]
        if B * N != BxN:
            raise ValueError(f"Batch mismatch: dataset_flag {tuple(dataset_flag.shape)} vs input_ids rows {BxN}")

        flag = dataset_flag.to(device).bool()
        block_flag = flag[:, 0]
        idx_all = torch.arange(BxN, device=device)
        idx_grid = idx_all.view(B, N)

        rep_inputs_idx = idx_grid[block_flag, 0]
        unique_inputs_idx_mode2 = idx_grid[~block_flag, :].reshape(-1)
        idx_unique_inputs = torch.cat([unique_inputs_idx_mode2, rep_inputs_idx], dim=0)

        pos_in_compact = torch.full((BxN,), -1, dtype=torch.long, device=device)
        pos_in_compact[idx_unique_inputs] = torch.arange(idx_unique_inputs.numel(), device=device)

        map_inputs = torch.empty(BxN, dtype=torch.long, device=device)
        map_inputs[unique_inputs_idx_mode2] = pos_in_compact[unique_inputs_idx_mode2]
        if rep_inputs_idx.numel() > 0:
            rows_dup = idx_grid[block_flag, :].reshape(-1)
            rep_pos = pos_in_compact[rep_inputs_idx]
            map_inputs[rows_dup] = rep_pos.repeat_interleave(N)

        if (map_inputs < 0).any():
            bad = (map_inputs < 0).nonzero(as_tuple=False).squeeze(-1)[:20]
            raise RuntimeError(
                f"map_inputs has -1 indices : {bad.tolist()}. "
                "Check dataset_flag/idx_unique_inputs mapping."
            )

        bert_outputs = self.bert(
            input_ids=input_ids[idx_unique_inputs],
            attention_mask=attention_mask[idx_unique_inputs] if attention_mask is not None else None,
            return_dict=True,
        )
        seq_compact = bert_outputs.last_hidden_state
        sequence_output = seq_compact[map_inputs]  # (B*N, L, H)

        # 3) Desc model, remove duplicates
        unique_desc_idx = idx_grid[block_flag, :].reshape(-1)
        rep_desc_idx = idx_grid[~block_flag, 0]
        idx_unique_desc = torch.cat([unique_desc_idx, rep_desc_idx], dim=0)

        pos_in_compact = torch.full((BxN,), -1, dtype=torch.long, device=device)
        pos_in_compact[idx_unique_desc] = torch.arange(idx_unique_desc.numel(), device=device)

        map_desc = torch.empty((BxN,), dtype=torch.long, device=device)
        if unique_desc_idx.numel() > 0:
            map_desc[unique_desc_idx] = pos_in_compact[unique_desc_idx]
        if rep_desc_idx.numel() > 0:
            rows_dup = idx_grid[~block_flag, :].reshape(-1)
            rep_pos = pos_in_compact[rep_desc_idx]
            map_desc[rows_dup] = rep_pos.repeat_interleave(N)

        if (map_desc < 0).any():
            bad = (map_desc < 0).nonzero(as_tuple=False).squeeze(-1)[:20]
            raise RuntimeError(f"map_desc has -1 indices: {bad.tolist()}")

        desc_out = self.desc_model(
            input_ids=desc_input_ids[idx_unique_desc],
            attention_mask=desc_attention_mask[idx_unique_desc] if desc_attention_mask is not None else None,
            return_dict=True,
        )
        desc_pooled = desc_out.last_hidden_state[:, -1]

        proj_dtype = next(self.desc_proj.parameters()).dtype
        desc_pooled = desc_pooled.to(proj_dtype)
        desc_pooled = self.desc_proj(desc_pooled)
        desc_pooled = desc_pooled.to(sequence_output.dtype)
        desc_pooled = self.desc_ln(desc_pooled)

        desc_output = desc_pooled[map_desc]  # (B*N, H)

        # optional console debug (как было)
        if self.debug_print_norms:
            with torch.no_grad():
                seq_mean_no_cls = self._masked_mean_no_cls(sequence_output.float(), attention_mask)
                seq_norm = seq_mean_no_cls.norm(p=2, dim=-1)
                desc_norm = desc_output.float().norm(p=2, dim=-1)

                to_print = min(BxN, max(1, self.debug_max_print))
                if BxN > to_print:
                    print(f"[norms] printing first {to_print}/{BxN} samples (set debug_max_print to change)")

                for i in range(to_print):
                    print(
                        f"[sample {i:04d}] "
                        f"map_in={int(map_inputs[i])} "
                        f"map_desc={int(map_desc[i])} "
                        f"||mean(seq,noCLS)||={seq_norm[i].item():.6f} "
                        f"||desc||={desc_norm[i].item():.6f}"
                    )

        # fuse
        sequence_output = sequence_output.contiguous()
        if attention_mask is not None:
            sequence_output = sequence_output + desc_output[:, None, :] * attention_mask[:, :, None].to(sequence_output.dtype)
        else:
            sequence_output = sequence_output + desc_output[:, None, :]

        # 4) Encoder + classifier
        encoder_output = self.transformer_encoder(sequence_output)  # (B*N, L, H)
        logits_raw = self.classifier(encoder_output)               # (B*N, L, 1) ДО activation
        logits = self.activation(logits_raw)                       # (B*N, L, 1)

        # 5) Loss
        loss = None
        labels_reshaped = labels_mask_reshaped = cls_loss = deviation_loss = multinomial_loss = other_loss = None

        if labels is not None:
            labels_reshaped = labels.to(logits.device)
            labels_mask_reshaped = labels_mask.to(logits.device) if labels_mask is not None else None

            unreduced_loss = self.loss_fct(logits, labels_reshaped)  # (B*N, L, 1)

            # ===================== FILE LOGGING (PER-SAMPLE, ALL SAMPLES) =====================
            do_log = (
                self.enable_file_logging
                and (self._log_fh is not None)
                and self._is_main
                and (global_step is not None)
                and (self.log_every_n_steps > 0)
                and (global_step % self.log_every_n_steps == 0)
            )

            if do_log:
                with torch.no_grad():
                    eps = 1e-8
                    BxN_local = logits.shape[0]
                    L = logits.shape[1]
                    N_local = N

                    # sample_ids -> list
                    if sample_ids is not None:
                        if isinstance(sample_ids, torch.Tensor):
                            sids = sample_ids.detach().to("cpu")
                            if sids.dtype.is_floating_point:
                                sids = sids.long()
                            sids_list = sids.tolist()
                        else:
                            sids_list = list(sample_ids)
                        if len(sids_list) != BxN_local:
                            sids_list = [None] * BxN_local
                    else:
                        sids_list = [None] * BxN_local

                    # norms
                    seq_mean_no_cls = self._masked_mean_no_cls(sequence_output.float(), attention_mask)
                    seq_norm = seq_mean_no_cls.norm(p=2, dim=-1)          # (BxN,)
                    desc_norm = desc_output.float().norm(p=2, dim=-1)     # (BxN,)

                    # lengths
                    if attention_mask is not None:
                        seq_len_real = attention_mask.sum(dim=1).detach().to("cpu").tolist()
                    else:
                        seq_len_real = [None] * BxN_local

                    if desc_attention_mask is not None:
                        # desc_attention_mask already flattened (BxN, D)
                        desc_len_real = desc_attention_mask.sum(dim=1).detach().to("cpu").tolist()
                    else:
                        desc_len_real = [None] * BxN_local

                    # per-sample MSE
                    loss_tok = unreduced_loss.squeeze(-1).float()         # (BxN, L)
                    if labels_mask_reshaped is not None:
                        m_tok = labels_mask_reshaped.squeeze(-1).float()  # (BxN, L)
                        denom = m_tok.sum(dim=1).clamp_min(eps)
                        loss_sample = (loss_tok * m_tok).sum(dim=1) / denom

                        cls_m = m_tok[:, 0].clamp_min(0.0)
                        other_m = m_tok[:, 1:]

                        cls_denom = cls_m.clamp_min(eps)
                        cls_loss_sample = (loss_tok[:, 0] * cls_m) / cls_denom

                        other_denom = other_m.sum(dim=1).clamp_min(eps)
                        other_loss_sample = (loss_tok[:, 1:] * other_m).sum(dim=1) / other_denom
                        labelmask_sum = m_tok.sum(dim=1).detach().to("cpu").tolist()
                    else:
                        loss_sample = loss_tok.mean(dim=1)
                        cls_loss_sample = loss_tok[:, 0]
                        other_loss_sample = loss_tok[:, 1:].mean(dim=1) if L > 1 else loss_tok[:, 0]
                        labelmask_sum = [None] * BxN_local

                    # logits stats per sample (raw)
                    lr = logits_raw.squeeze(-1).float()                   # (BxN, L)
                    logits_raw_absmax = lr.abs().amax(dim=1)
                    logits_raw_max = lr.amax(dim=1)
                    logits_raw_min = lr.amin(dim=1)

                    # finiteness checks
                    isfinite_desc = torch.isfinite(desc_output.float()).all(dim=1)
                    isfinite_seqm = torch.isfinite(seq_mean_no_cls).all(dim=1)
                    isfinite_logits = torch.isfinite(logits.float()).all(dim=(1, 2))

                    # mapping / flags to cpu
                    map_in_cpu = map_inputs.detach().to("cpu").tolist()
                    map_desc_cpu = map_desc.detach().to("cpu").tolist()
                    df_cpu = dataset_flag.detach().to("cpu")
                    block_flag_cpu = (df_cpu[:, 0] > 0).tolist()

                    # to lists
                    seq_norm_list = seq_norm.detach().to("cpu").tolist()
                    desc_norm_list = desc_norm.detach().to("cpu").tolist()
                    loss_sample_list = loss_sample.detach().to("cpu").tolist()
                    cls_loss_list = cls_loss_sample.detach().to("cpu").tolist()
                    other_loss_list = other_loss_sample.detach().to("cpu").tolist()
                    lra_list = logits_raw_absmax.detach().to("cpu").tolist()
                    lmax_list = logits_raw_max.detach().to("cpu").tolist()
                    lmin_list = logits_raw_min.detach().to("cpu").tolist()
                    fin_desc_list = isfinite_desc.detach().to("cpu").tolist()
                    fin_seq_list = isfinite_seqm.detach().to("cpu").tolist()
                    fin_logits_list = isfinite_logits.detach().to("cpu").tolist()

                    lines = []
                    step_val = int(global_step)

                    for i in range(BxN_local):
                        b = i // N_local
                        n = i % N_local
                        rec = {
                            "type": "sample",
                            "step": step_val,
                            "flat_i": int(i),
                            "b": int(b),
                            "n": int(n),
                            "sample_id": sids_list[i],

                            "dataset_flag": int(df_cpu[b, n].item()),
                            "block_flag": bool(block_flag_cpu[b]),
                            "map_inputs": int(map_in_cpu[i]),
                            "map_desc": int(map_desc_cpu[i]),

                            "seq_len": seq_len_real[i],
                            "desc_len": desc_len_real[i],
                            "labelmask_sum": labelmask_sum[i],

                            "loss_sample": float(loss_sample_list[i]),
                            "cls_loss_sample": float(cls_loss_list[i]),
                            "other_loss_sample": float(other_loss_list[i]),

                            "seq_mean_no_cls_norm": float(seq_norm_list[i]),
                            "desc_norm": float(desc_norm_list[i]),
                            "logits_raw_absmax": float(lra_list[i]),
                            "logits_raw_max": float(lmax_list[i]),
                            "logits_raw_min": float(lmin_list[i]),

                            "finite_desc": bool(fin_desc_list[i]),
                            "finite_seq_mean": bool(fin_seq_list[i]),
                            "finite_logits": bool(fin_logits_list[i]),
                        }
                        lines.append(json.dumps(rec, ensure_ascii=False))

                    self._write_jsonl_lines(lines)
            # ====================================================================

            if labels_mask_reshaped is not None and labels_mask_reshaped.sum().item() > 0:
                cls_mask = labels_mask_reshaped[:, 0:1, :]
                other_mask = labels_mask_reshaped[:, 1:, :]

                if cls_mask.sum().item() > 0:
                    cls_loss = (unreduced_loss[:, 0:1, :] * cls_mask).sum() / (cls_mask.sum() + 1e-8)
                    if self.use_deviation_loss:
                        deviation_loss = cls_deviation_from_mean_loss(
                            logits, labels_reshaped, labels_mask_reshaped, n_keys=N
                        )
                    if self.use_multinomial_loss:
                        multinomial_loss = cls_multinomial_loss(
                            logits, labels_reshaped, labels_mask_reshaped, n_keys=N
                        )

                if other_mask.sum().item() > 0:
                    other_loss = (unreduced_loss[:, 1:, :] * other_mask).sum() / (other_mask.sum() + 1e-8)

                if cls_loss is not None and other_loss is not None:
                    loss = cls_loss + self.weight * other_loss
                elif cls_loss is not None:
                    loss = cls_loss
                elif other_loss is not None:
                    loss = self.weight * other_loss

                if loss is not None:
                    if deviation_loss is not None:
                        loss = loss + self.weight_deviation_loss * deviation_loss
                    if multinomial_loss is not None:
                        loss = loss + self.weight_multinomial_loss * multinomial_loss
            else:
                # если mask нет или пустой — можно просто усреднить MSE
                # (оставляю поведение максимально близким к твоему: loss может остаться None)
                pass

        if not return_dict:
            return (loss, logits)

        return ExpressionModelOutput(
            loss=loss,
            logits=logits,
            attentions=bert_outputs.attentions,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            cls_loss=cls_loss,
            other_loss=other_loss,
            deviation_loss=deviation_loss,
            multinomial_loss=multinomial_loss,
        )
