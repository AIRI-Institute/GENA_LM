import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional
from dataclasses import dataclass
from transformers import AutoConfig, AutoModel, ModernBertModel
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_info()



@dataclass
class ExpressionModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None

class ExpActivation(nn.Module):
    def forward(self, x):
        return torch.exp(x)

class ExpressionCounts(nn.Module):
    """
    Ожидаемые формы:
      - full_input_ids:      (B*N, L_full)
      - full_attention_mask: (B*N, L_full)
      - dataset_flag:   (B, N)   [в блоке из N элементов либо все 1 (дубли INPUTS), либо все 0 (дубли DESC)]
      - labels:         (B*N,)
      - labels_mask:    (B*N,)
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
    ):
        super().__init__()

        # 1) DNA model (GENA) 
        if hf:
            if "caduceus" in hf_model_name.lower():
                print(f"using caduceus: {hf_model_name}")
                self.caduceus = AutoModel.from_pretrained(
                    hf_model_name,
                    trust_remote_code=True #No sdpa or flash attn? 
                )
                config = self.caduceus.config
            else:
                raise ValueError(
                    f"Unsupported hf_model_name for ExpressionCounts: {hf_model_name}. "
                    "Expected a Caduceus checkpoint when hf=True."
                )
        else:
            self.bert = BertModel(config, add_pooling_layer=False)
            checkpoint = torch.load(bert_cpt, map_location="cpu")
            state_dict = checkpoint["model_state_dict"]
            updated_state_dict = {k.replace("bert.", ""): v for k, v in state_dict.items()}
            missing_k, unexpected_k = self.bert.load_state_dict(updated_state_dict, strict=False)

        if not hf:
            if len(missing_k) != 0:
                print(f"{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.")
            if len(unexpected_k) != 0:
                print(f"{unexpected_k} were found in checkpoint, but model is not expecting them!")


        self.config = config

        # 2) Description model (qwen)
        self.desc_model_name = desc_model_name
        self.desc_model = AutoModel.from_pretrained(self.desc_model_name,attn_implementation="sdpa" )

        for p in self.desc_model.parameters():
            p.requires_grad = False

        backbone = getattr(self.desc_model, "model", None)
        if backbone is None:
            backbone = self.desc_model

        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise RuntimeError("Не нашёл слои у desc_model (ожидал .model.layers). Проверь архитектуру модели.")

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

            # if len(names) > 30:
            #     print(f"  - ... (+{len(names)-30} more)")
            # LEV: above was broken - names not defined

        # 3) Проекция, если размерности не совпадают
        encoder_hidden_size = getattr(config, 'hidden_size', config.d_model)
        if getattr(config, 'rcps', False):
            encoder_hidden_size = 2 * encoder_hidden_size
        self.gen_hidden_size = encoder_hidden_size
        self.desc_hidden_size = self.desc_model.config.hidden_size
        # if self.desc_hidden_size != self.gen_hidden_size:
        #     self.desc_proj = nn.Linear(self.desc_hidden_size, self.gen_hidden_size)
        # else:
        #     self.desc_proj = nn.Identity()

        # 4) Decoder — ModernBERT как «второй стек» (inputs_embeds + attention_mask), sdpa как у desc_model (V100).
        print(f"Using ModernBERT for decoder from {hf_model_name_decoder}")
        decoder_config = AutoConfig.from_pretrained(hf_model_name_decoder)
        decoder_config.reference_compile = False
        self.decoder, info2 = ModernBertModel.from_pretrained(
            hf_model_name_decoder,
            config=decoder_config,
            trust_remote_code=True,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        print("missing:", len(info2["missing_keys"]), info2["missing_keys"][:10])
        print("unexpected:", len(info2["unexpected_keys"]), info2["unexpected_keys"][:10])
        print("mismatched:", info2.get("mismatched_keys", [])[:5])

        # 6) Loss
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight

        _encoder = self.caduceus if hasattr(self, "caduceus") else self.bert
        dtype = next(_encoder.parameters()).dtype
        device = next(_encoder.parameters()).device

        decoder_hidden_size = int(self.decoder.config.hidden_size)
        if self.gen_hidden_size != decoder_hidden_size:
            self.encoder_to_decoder = nn.Linear(
                self.gen_hidden_size, decoder_hidden_size, device=device, dtype=dtype
            )
        else:
            self.encoder_to_decoder = nn.Identity()

        self.desc_proj_decoder = nn.Linear(
            self.desc_hidden_size, decoder_hidden_size, device=device, dtype=dtype
        )
        self.dna_ln_dec = nn.LayerNorm(decoder_hidden_size, device=device, dtype=dtype)
        self.desc_ln_dec = nn.LayerNorm(decoder_hidden_size, device=device, dtype=dtype)

        # 5) Classifier — по каждой позиции L (как в исходном коде).
        self.classifier = nn.Linear(decoder_hidden_size, 1, device=device, dtype=dtype)

        if hasattr(self.decoder, "embeddings") and hasattr(self.decoder.embeddings, "tok_embeddings"):
            self.decoder.embeddings.tok_embeddings.weight.requires_grad_(False)

    def forward(
        self,
        full_input_ids=None,         # (B*N, L_full) or None
        full_attention_mask=None,    # (B*N, L_full) or None
        tss_token_idx=None,          # (B*N,) index in unpadded full sequence
        labels_mask=None,            # (B*N,)
        labels=None,                 # (B*N,)
        return_dict=None,
        desc_input_ids=None,           # (B, N, D)
        desc_attention_mask = None,
        dataset_flag=None,           # (B, N): 1 -> дубли INPUTS; 0 -> дубли DESC
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if dataset_flag is None:
            raise ValueError("dataset_flag must be provided and shaped (B, N)")
        if full_input_ids is None:
            raise ValueError("full_input_ids must be provided")

        B, N = dataset_flag.shape

        # 1) Reshape
        if full_input_ids is not None and full_input_ids.dim() == 3:
            full_input_ids = full_input_ids.reshape(B * N, full_input_ids.shape[-1])

        if full_attention_mask is not None and full_attention_mask.dim() == 3:
            full_attention_mask = full_attention_mask.reshape(B * N, full_attention_mask.shape[-1])

        if tss_token_idx is not None and tss_token_idx.dim() == 2:
            tss_token_idx = tss_token_idx.reshape(B * N)

        if labels is not None and labels.dim() == 2:
            labels = labels.reshape(B * N)

        if labels_mask is not None and labels_mask.dim() == 2:
            labels_mask = labels_mask.reshape(B * N)

        if desc_input_ids is not None and desc_input_ids.dim() == 3:                              # (B, N, D) -> (B*N, D)
                desc_input_ids = desc_input_ids.reshape(B * N, desc_input_ids.shape[-1])

        if desc_attention_mask is not None and desc_attention_mask.dim() == 3:                              # (B, N, D) -> (B*N, D)
                desc_attention_mask = desc_attention_mask.reshape(B * N, desc_attention_mask.shape[-1])

        # 2) DNA model, убираем повторы
        # FIX: Encode full gene with Caduceus when available.
        src = full_input_ids
        device = src.device
        BxN, seq_len = src.shape[:2]
        B, N = dataset_flag.shape
        if B * N != BxN:
            raise ValueError(f"Batch mismatch: dataset_flag {tuple(dataset_flag.shape)} vs full_input_ids rows {BxN}")
        
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

        caduceus_outputs = self.caduceus(
                input_ids=src[idx_unique_inputs],
                return_dict=True,
            )
        seq_compact = caduceus_outputs.last_hidden_state                  # (U_inp, L, H)
        sequence_output = seq_compact[map_inputs]                       # (B*N, L_full, H) or (B*N, L, H)
        encoder_attention_mask = full_attention_mask
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(BxN, seq_len, device=device, dtype=torch.long)

        #  Crop fixed 1024 window around TSS 
        if tss_token_idx is None:
            raise ValueError("tss_token_idx must be provided when full_input_ids is used")
        left_context = 512
        right_context = 511
        crop_len = left_context + right_context + 1
        pad_shift = (encoder_attention_mask.shape[1] - encoder_attention_mask.sum(dim=1)).long()
        tss_pos = (tss_token_idx.long().to(device) + pad_shift).clamp(0, encoder_attention_mask.shape[1] - 1)

        cropped_hidden = []
        cropped_mask = []
        base = torch.arange(crop_len, device=device, dtype=torch.long)
        for i in range(BxN):
            idx = tss_pos[i] - left_context + base
            valid = (idx >= 0) & (idx < encoder_attention_mask.shape[1])
            idx_safe = idx.clamp(0, encoder_attention_mask.shape[1] - 1)
            hid = sequence_output[i, idx_safe]
            msk = encoder_attention_mask[i, idx_safe] > 0
            msk = msk & valid
            hid = hid * msk[:, None].to(hid.dtype)
            cropped_hidden.append(hid)
            cropped_mask.append(msk.long())

        sequence_output = torch.stack(cropped_hidden, dim=0)
        attention_mask = torch.stack(cropped_mask, dim=0)
        seq_len = crop_len

        # 3) Description model, убираем повторы 
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

        
        sequence_output = sequence_output.contiguous()
        if attention_mask is None:
            attention_mask = torch.ones(BxN, seq_len, device=device, dtype=torch.long)
        sequence_output = sequence_output * attention_mask[:, :, None].to(sequence_output.dtype)

        dec_dtype = next(self.decoder.parameters()).dtype
        dna_emb = self.encoder_to_decoder(sequence_output).to(dtype=dec_dtype)
        dna_emb = self.dna_ln_dec(dna_emb)
        dna_attn = attention_mask

        #
        desc_out = self.desc_model(
            input_ids=desc_input_ids[idx_unique_desc],
            attention_mask=desc_attention_mask[idx_unique_desc],
            return_dict=True,
        )
        desc_seq_compact = desc_out.last_hidden_state
        desc_seq = desc_seq_compact[map_desc]
        desc_seq = self.desc_proj_decoder(desc_seq).to(dtype=dec_dtype)
        desc_seq = self.desc_ln_dec(desc_seq)
        desc_attn = desc_attention_mask

        #
        combined_emb = torch.cat([desc_seq, dna_emb], dim=1)
        combined_attn = torch.cat([desc_attn, dna_attn], dim=1)

        # 
        dec_out = self.decoder(
            inputs_embeds=combined_emb,
            attention_mask=combined_attn,
            return_dict=True,
        )
        decoder_output = dec_out.last_hidden_state

        #
        dna_pool_mask = torch.cat(
            [torch.zeros_like(desc_attn), dna_attn], dim=1
        ).to(decoder_output.dtype).unsqueeze(-1)

        pooled = (decoder_output * dna_pool_mask).sum(dim=1) / dna_pool_mask.sum(dim=1).clamp_min(1e-8)
        logits = self.activation(self.classifier(pooled)).squeeze(-1)

        # 5) Loss
        loss = None
        labels_reshaped = labels_mask_reshaped = None
        cls_loss = None
        other_loss = None

        if labels is not None:
            if labels_mask is None:
                raise ValueError("labels_mask must be provided when labels are provided")
            labels_reshaped = labels.reshape(-1).to(logits.device, dtype=logits.dtype)
            labels_mask_reshaped = labels_mask.reshape(-1).to(logits.device, dtype=logits.dtype)
            unreduced_loss = self.loss_fct(logits, labels_reshaped)
            denom = labels_mask_reshaped.sum().clamp_min(1.0)
            loss = (unreduced_loss * labels_mask_reshaped).sum() / denom

        if not return_dict:
            return (loss, logits)

        hidden_states_out = (decoder_output,)

        return ExpressionModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states_out,
            attentions=None,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            cls_loss=cls_loss,
            other_loss=other_loss,
        )
