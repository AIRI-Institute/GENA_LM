import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import TokenClassifierOutput
from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModel, BertConfig, ModernBertModel
from transformers.utils import cached_file
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_info()

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

    For each n_keys: group mean is computed only over positions with mask == 1.
    loss = MSE(logit_dev - label_dev) over masked positions in each group.

    logits: (B*N, L, 1)
    labels: (B*N, L, 1)
    labels_mask: (B*N, L, 1)
    n_keys: N — group size
    """
    B = logits.shape[0] // n_keys
    cls_logits = logits[:, 0:1, :]   # (B*N, 1, 1)
    cls_labels = labels[:, 0:1, :]   # (B*N, 1, 1)
    cls_mask = labels_mask[:, 0:1, :]  # (B*N, 1, 1)

    cls_logits = cls_logits.view(B, n_keys, 1, 1)   # (B, N, 1, 1)
    cls_labels = cls_labels.view(B, n_keys, 1, 1)   # (B, N, 1, 1)
    cls_mask = cls_mask.view(B, n_keys, 1, 1)       # (B, N, 1, 1)

    cnt = cls_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1, 1, 1)
    mean_logits = (cls_logits * cls_mask).sum(dim=1, keepdim=True) / cnt   # (B, 1, 1, 1)
    mean_labels = (cls_labels * cls_mask).sum(dim=1, keepdim=True) / cnt   # (B, 1, 1, 1)

    logit_dev = (cls_logits - mean_logits) * cls_mask   # (B, N, 1, 1)
    label_dev = (cls_labels - mean_labels) * cls_mask   # (B, N, 1, 1)
    dev_sq = (logit_dev - label_dev).pow(2)             # (B, N, 1, 1)

    loss_per_group = dev_sq.sum(dim=1) / cnt.squeeze(1)  # (B, 1, 1)
    group_has_mask = (cls_mask.sum(dim=1) > 0).squeeze()  # (B,)
    if group_has_mask.sum() == 0:
        return logits.new_zeros(())
    return loss_per_group[group_has_mask].mean()


import torch
import torch.nn.functional as F

def cls_multinomial_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    labels_mask: torch.Tensor,
    n_keys: int,
) -> torch.Tensor:
    """
    Multinomial (NLL) loss over each n_keys for CLS token only.
    In each group of n: masked log_softmax(logits) and NLL w.r.t. target distribution
    (labels normalized per group only over positions with mask == 1).
    Positions with mask 0 participate neither in target nor in softmax normalization.

    logits:       (B*N, L, 1)
    labels:       (B*N, L, 1)
    labels_mask:  (B*N, L, 1)
    n_keys:       N — group size
    """
    B = logits.shape[0] // n_keys

    # CLS-only (B*N, 1) -> (B*N,)
    cls_logits = logits[:, 0:1, :].squeeze(-1).squeeze(-1)      # (B*N,)
    cls_labels = labels[:, 0:1, :].squeeze(-1).squeeze(-1)      # (B*N,)
    cls_mask   = labels_mask[:, 0:1, :].squeeze(-1).squeeze(-1) # (B*N,)

    # (B, N)
    cls_logits = cls_logits.view(B, n_keys)
    cls_labels = cls_labels.view(B, n_keys)
    cls_mask   = cls_mask.view(B, n_keys)

    mask_bool = cls_mask > 0
    group_has_mask = mask_bool.any(dim=1)  # (B,)
    if group_has_mask.sum() == 0:
        return logits.new_zeros(())

    # masked log_softmax: exclude mask==0 from normalization
    neg = torch.finfo(cls_logits.dtype).min
    masked_logits = cls_logits.masked_fill(~mask_bool, neg)
    log_probs = F.log_softmax(masked_logits, dim=1)  # (B, N)

    # target distribution: normalize labels only over mask==1
    target = cls_labels * cls_mask
    target_sum = target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    target = target / target_sum  # (B, N)

    nll_per_group = -(target * log_probs).sum(dim=1)  # (B,)
    return nll_per_group[group_has_mask].mean()


class ExpressionCounts(nn.Module):
    """
    Expected shapes:
      - input_ids:      (B*N, L)
      - attention_mask: (B*N, L)
      - token_type_ids: (B*N, L) [optional]
      - desc_vectors:   (B, N, D)
      - dataset_flag:   (B, N)   [in block of N elements either all 1 (INPUTS duplicates), or all 0 (DESC duplicates)]
      - labels:         (B*N, L, 1)
      - labels_mask:    (B*N, L, 1)
    """

    def __init__(
        self,
        config,
        hf_model_name_decoder,
        loss_fct=nn.MSELoss(reduction="none"),
        activation = nn.Identity(),
        num_encoder_layers = 3,
        nhead = 8,
        weight = 1,
        hidden_ff = 1024,
        bert_cpt = '/mnt/nfs_dna/DNALM/trained_models/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best.pth',
        hf: bool = False,
        hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
        desc_model_name: str = "intfloat/multilingual-e5-large-instruct",
        use_deviation_loss: bool = False,
        use_multinomial_loss: bool = False,
        weight_deviation_loss: float = 1.0,
        weight_multinomial_loss: float = 1.0,
    ):
        super().__init__()

        updated_state_dict = None

        # 1) DNA model (GENA) 
        if hf:
            if "modernbert" in hf_model_name.lower():
                print(f"Using ModernBERT from {hf_model_name}")
                self.bert, info  = ModernBertModel.from_pretrained(
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

        # 2) Description model (qwen)
        self.desc_model_name = desc_model_name
        self.desc_model = AutoModel.from_pretrained(self.desc_model_name,attn_implementation="flash_attention_2" , torch_dtype=torch.bfloat16)

        for p in self.desc_model.parameters():
            p.requires_grad = False

        backbone = getattr(self.desc_model, "model", None)
        if backbone is None:
            backbone = self.desc_model

        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise RuntimeError("Could not find layers in desc_model (expected .model.layers). Check model architecture.")

        #unfreeze last 4 blocks
        k = 4
        k = min(k, len(layers)) 

        for block in layers[-k:]:
            for p in block.parameters():
                p.requires_grad = True

        if hasattr(backbone, "norm") and backbone.norm is not None:
            for p in backbone.norm.parameters():
                p.requires_grad = True

        for block in layers[:-k]:
            block.eval()
        for block in layers[-k:]:
            block.train()

        if hasattr(backbone, "norm") and backbone.norm is not None:
            backbone.norm.train()

        def _is_main_process():
            return (not torch.distributed.is_available()
                    or not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0)

        if _is_main_process():
            unfrozen_blocks = []
            for i, block in enumerate(layers):
                if any(p.requires_grad for p in block.parameters()):
                    unfrozen_blocks.append(i)

            print(f"[desc_model] unfrozen transformer blocks: {unfrozen_blocks} (total blocks={len(layers)})")

            if hasattr(backbone, "norm") and backbone.norm is not None:
                norm_trainable = any(p.requires_grad for p in backbone.norm.parameters())
                norm_params = sum(p.numel() for p in backbone.norm.parameters() if p.requires_grad)
                print(f"[desc_model] backbone.norm trainable: {norm_trainable} (trainable params={norm_params:,})")
            else:
                print("[desc_model] backbone.norm: not found")

            total_trainable = sum(p.numel() for p in self.desc_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.desc_model.parameters())
            print(f"[desc_model] trainable params: {total_trainable:,} / {total_params:,}")

            if len(names) > 30:
                print(f"  - ... (+{len(names)-30} more)")

        # 3) Projection if dimensions do not match
        self.gen_hidden_size = config.hidden_size
        self.desc_hidden_size = self.desc_model.config.hidden_size
        if self.desc_hidden_size != self.gen_hidden_size:
            self.desc_proj = nn.Linear(self.desc_hidden_size, self.gen_hidden_size)
        else:
            self.desc_proj = nn.Identity()

        # 4) Decoder
        print(f"Using ModernBERT for dercoder from {hf_model_name_decoder}")
        self.decoder, info2 = ModernBertModel.from_pretrained(
                hf_model_name_decoder,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                output_loading_info=True
            )
        print("missing:", len(info2["missing_keys"]), info2["missing_keys"][:10])
        print("unexpected:", len(info2["unexpected_keys"]), info2["unexpected_keys"][:10])
        print("mismatched:", info2.get("mismatched_keys", [])[:5])

        # 6) Loss
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight
        self.use_deviation_loss = use_deviation_loss
        self.use_multinomial_loss = use_multinomial_loss
        self.weight_deviation_loss = weight_deviation_loss
        self.weight_multinomial_loss = weight_multinomial_loss

        dtype = next(self.bert.parameters()).dtype
        device = next(self.bert.parameters()).device

        self.desc_proj = nn.Linear(self.desc_hidden_size, self.gen_hidden_size, device=device, dtype=dtype)

        # 5) Classifier
        self.classifier = nn.Linear(self.decoder.config.hidden_size, 1, device=device, dtype=dtype)

        if hasattr(self.decoder, "embeddings") and hasattr(self.decoder.embeddings, "tok_embeddings"):
            self.decoder.embeddings.tok_embeddings.weight.requires_grad_(False)


    def forward(
        self,
        input_ids=None,              # (B*N, L)
        attention_mask=None,         # (B*N, L) or None
        labels_mask=None,            # (B*N, L, 1)
        labels=None,                 # (B*N, L, 1)
        return_dict=None,
        desc_input_ids=None,           # (B, N, D)
        desc_attention_mask = None,
        dataset_flag=None,           # (B, N): 1 -> INPUTS duplicates; 0 -> DESC duplicates
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if dataset_flag is None:
            raise ValueError("dataset_flag must be provided and shaped (B, N)")
        
        B, N = dataset_flag.shape

        # 1) Reshape
        if input_ids is not None and input_ids.dim() == 3:          # (B, N, L) -> (B*N, L)
            if input_ids.shape[:2] != (B, N):
                raise ValueError(f"input_ids has shape {tuple(input_ids.shape)}, but dataset_flag is {(B, N)}")
            input_ids = input_ids.reshape(B * N, input_ids.shape[-1])

        if attention_mask is not None and attention_mask.dim() == 3: # (B, N, L) -> (B*N, L)
            attention_mask = attention_mask.reshape(B * N, attention_mask.shape[-1])

        # if token_type_ids is not None and token_type_ids.dim() == 3: # (B, N, L) -> (B*N, L)
        #     token_type_ids = token_type_ids.reshape(B * N, token_type_ids.shape[-1])

        if labels is not None and labels.dim() == 4:                 # (B, N, L, 1) -> (B*N, L, 1)
            labels = labels.reshape(B * N, labels.shape[-2], labels.shape[-1])

        if labels_mask is not None and labels_mask.dim() == 4:       # (B, N, L, 1) -> (B*N, L, 1)
            labels_mask = labels_mask.reshape(B * N, labels_mask.shape[-2], labels_mask.shape[-1])

        if desc_input_ids is not None and desc_input_ids.dim() == 3:                              # (B, N, D) -> (B*N, D)
                desc_input_ids = desc_input_ids.reshape(B * N, desc_input_ids.shape[-1])

        if desc_attention_mask is not None and desc_attention_mask.dim() == 3:                              # (B, N, D) -> (B*N, D)
                desc_attention_mask = desc_attention_mask.reshape(B * N, desc_attention_mask.shape[-1])

        # 2) DNA model, remove duplicates
        src = input_ids 
        if src is None:
            raise ValueError("input_ids must be provided")
        device = src.device
        BxN, seq_len = src.shape[:2]
        B, N = dataset_flag.shape
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
                attention_mask=attention_mask[idx_unique_inputs],
                return_dict=True,
            )
        seq_compact = bert_outputs.last_hidden_state                    # (U_inp, L, H)
        sequence_output = seq_compact[map_inputs]                       # (B*N, L, H)
        hidden_size = sequence_output.size(-1)

        # 3) Description model, remove duplicates 
        unique_desc_idx = idx_grid[block_flag, :].reshape(-1)   # (B_true*N,)
        rep_desc_idx = idx_grid[~block_flag, 0]                 # (B_false,)
        idx_unique_desc = torch.cat([unique_desc_idx, rep_desc_idx], dim=0)  # (U_desc,)

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
                attention_mask=desc_attention_mask[idx_unique_desc],
                return_dict=True,
                )
        desc_pooled = desc_out.last_hidden_state[:, -1]
        desc_pooled = self.desc_proj(desc_pooled)                    
        desc_pooled = desc_pooled.to(sequence_output.dtype) 
        desc_output = desc_pooled[map_desc] 


        sequence_output = sequence_output.contiguous()
        if attention_mask is not None:
            sequence_output = sequence_output + desc_output[:, None, :] * attention_mask[:, :, None].to(sequence_output.dtype)
        else:
            sequence_output = sequence_output + desc_output[:, None, :]


        # 4) Decoder
        dec_out = self.decoder(
            inputs_embeds=sequence_output,        # (B*N, L, H)
            attention_mask=attention_mask,        # (B*N, L)
            return_dict=True,
        )
        decoder_output = dec_out.last_hidden_state  # (B*N, L, H)

        logits = self.activation(self.classifier(decoder_output))  # (B*N, L, 1)

        # 5) Loss
        loss = None
        labels_reshaped = labels_mask_reshaped = cls_loss = deviation_loss = multinomial_loss = other_loss = None

        if labels is not None:
            labels_reshaped = labels.to(logits.device)
            labels_mask_reshaped = labels_mask.to(logits.device) if labels_mask is not None else None

            unreduced_loss = self.loss_fct(logits, labels_reshaped)  # (B*N, L, 1)

            if labels_mask_reshaped is not None and labels_mask_reshaped.sum().item() > 0:
                cls_mask = labels_mask_reshaped[:, 0:1, :]           # (B*N, 1, 1)
                other_mask = labels_mask_reshaped[:, 1:, :]          # (B*N, L-1, 1)

                if cls_mask.sum().item() > 0:
                    cls_loss = (unreduced_loss[:, 0:1, :] * cls_mask).sum() / (cls_mask.sum() + 1e-8)
                    # Group-wise losses over N for CLS only (with mask) — only if enabled by flags
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

        if not return_dict:
            return (loss, logits)

        hidden_states_out = (decoder_output,)

        return ExpressionModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states_out,
            attentions=bert_outputs.attentions,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            cls_loss=cls_loss,
            other_loss=other_loss,
            deviation_loss=deviation_loss,
            multinomial_loss=multinomial_loss,
        )
