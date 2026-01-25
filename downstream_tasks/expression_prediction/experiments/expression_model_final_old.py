import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModel, BertConfig
from transformers.utils import cached_file

@dataclass
class ExpressionModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None
    mean_loss: Optional[torch.FloatTensor] = None
    diviation_loss: Optional[torch.FloatTensor] = None
    

class ExpressionCounts(BertPreTrainedModel):
    """
    Ожидаемые формы:
      - input_ids:      (B*N, L)
      - attention_mask: (B*N, L)
      - token_type_ids: (B*N, L) [опционально]
      - desc_vectors:   (B, N, D)
      - dataset_flag:   (B, N)   [в блоке из N элементов либо все 1 (дубли INPUTS), либо все 0 (дубли DESC)]
      - labels:         (B*N, L, 1)
      - labels_mask:    (B*N, L, 1)
    """

    def __init__(
        self,
        config,
        loss_fct=nn.MSELoss(reduction="none"),
        activation = nn.Identity(),
        hidden_size_desc = 768,
        hidden_ff = 1024,
        num_encoder_layers = 3,
        nhead = 8,
        weight = 1,
        bert_cpt = '/mnt/nfs_dna/DNALM/trained_models/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best.pth',
        hf: bool = False,
        hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
        desc_model_name: str = "intfloat/multilingual-e5-large-instruct"
    ):
        super().__init__(config)
        self.config = config
        self.hidden_size_desc = hidden_size_desc 

        # 1) GENA (BERT)
        if hf:
            if "modernbert" in hf_model_name.lower():
                print(f"Using ModernBERT from {hf_model_name}")
                self.bert = AutoModel.from_pretrained(
                    hf_model_name,
                    trust_remote_code=True
                )
                config = self.bert.config
                weights_path = cached_file(hf_model_name, "pytorch_model.bin")
                updated_state_dict = torch.load(weights_path, map_location="cpu")
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

        if len(missing_k) != 0:
            print(f"{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.")
        if len(unexpected_k) != 0:
            print(f"{unexpected_k} were found in checkpoint, but model is not expecting them!")

        # Текстовая модель
        self.desc_model_name = desc_model_name
        self.desc_model = AutoModel.from_pretrained(self.desc_model_name,attn_implementation="flash_attention_2" , torch_dtype=torch.bfloat16)

        for p in self.desc_model.parameters():
            p.requires_grad = False

        # Qwen обычно хранит блоки здесь: desc_model.model.layers (ModuleList)
        backbone = getattr(self.desc_model, "model", None)
        if backbone is None:
            backbone = self.desc_model

        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise RuntimeError("Не нашёл слои у desc_model (ожидал .model.layers). Проверь архитектуру модели.")

        # --- unfreeze last transformer block ---
        for p in layers[-1].parameters():
            p.requires_grad = True

        # часто полезно разморозить финальный norm тоже
        if hasattr(backbone, "norm") and backbone.norm is not None:
            for p in backbone.norm.parameters():
                p.requires_grad = True

        # (опционально) отключить dropout в замороженных слоях, чтобы они были детерминированными
        for block in layers[:-1]:
            block.eval()
        layers[-1].train()

        # --------- PRINT WHAT IS UNFROZEN ----------
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

            # если хочешь увидеть имена (первые 30)
            names = [n for n, p in self.desc_model.named_parameters() if p.requires_grad]
            print(f"[desc_model] trainable tensors: {len(names)}")
            for n in names[:30]:
                print("  -", n)
            if len(names) > 30:
                print(f"  - ... (+{len(names)-30} more)")

        # Проекция, если размерности не совпадают
        self.gen_hidden_size = config.hidden_size
        self.desc_hidden_size = self.desc_model.config.hidden_size
        if self.desc_hidden_size != self.gen_hidden_size:
            self.desc_proj = nn.Linear(self.desc_hidden_size, self.gen_hidden_size)
        else:
            self.desc_proj = nn.Identity()


        # 3) Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_ff,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4) Classifier
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 5) Loss
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight

        self.post_init()

    def forward(
        self,
        input_ids=None,              # (B*N, L)
        attention_mask=None,         # (B*N, L) or None
        token_type_ids=None,         # (B*N, L) or None
        position_ids=None,
        labels_mask=None,            # (B*N, L, 1)
        head_mask=None,
        inputs_embeds=None,
        labels=None,                 # (B*N, L, 1)
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        desc_input_ids=None,           # (B, N, D)
        desc_attention_mask = None,
        dataset_flag=None,           # (B, N): 1 -> дубли INPUTS; 0 -> дубли DESC
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if dataset_flag is None:
            raise ValueError("dataset_flag must be provided and shaped (B, N)")
        
        B, N = dataset_flag.shape

        # --- input_ids / masks ---
        if input_ids is not None and input_ids.dim() == 3:          # (B, N, L) -> (B*N, L)
            if input_ids.shape[:2] != (B, N):
                raise ValueError(f"input_ids has shape {tuple(input_ids.shape)}, but dataset_flag is {(B, N)}")
            input_ids = input_ids.reshape(B * N, input_ids.shape[-1])

        if attention_mask is not None and attention_mask.dim() == 3: # (B, N, L) -> (B*N, L)
            attention_mask = attention_mask.reshape(B * N, attention_mask.shape[-1])

        if token_type_ids is not None and token_type_ids.dim() == 3: # (B, N, L) -> (B*N, L)
            token_type_ids = token_type_ids.reshape(B * N, token_type_ids.shape[-1])

        # --- inputs_embeds ---
        if inputs_embeds is not None and inputs_embeds.dim() == 4:   # (B, N, L, H) -> (B*N, L, H)
            inputs_embeds = inputs_embeds.reshape(B * N, *inputs_embeds.shape[2:])

        # --- labels / labels_mask ---
        if labels is not None and labels.dim() == 4:                 # (B, N, L, 1) -> (B*N, L, 1)
            labels = labels.reshape(B * N, labels.shape[-2], labels.shape[-1])

        if labels_mask is not None and labels_mask.dim() == 4:       # (B, N, L, 1) -> (B*N, L, 1)
            labels_mask = labels_mask.reshape(B * N, labels_mask.shape[-2], labels_mask.shape[-1])

        # --- desc_vectors ---
        if desc_input_ids is not None and desc_input_ids.dim() == 3:                              # (B, N, D) -> (B*N, D)
                desc_input_ids = desc_input_ids.reshape(B * N, desc_input_ids.shape[-1])

        if desc_attention_mask is not None and desc_attention_mask.dim() == 3:                              # (B, N, D) -> (B*N, D)
                desc_attention_mask = desc_attention_mask.reshape(B * N, desc_attention_mask.shape[-1])

        src = inputs_embeds if inputs_embeds is not None else input_ids
        if src is None:
            raise ValueError("Either inputs_embeds or input_ids must be provided")
        device = src.device
        BxN, seq_len = src.shape[:2]
        B, N = dataset_flag.shape
        if B * N != BxN:
            raise ValueError(f"Batch mismatch: dataset_flag {tuple(dataset_flag.shape)} vs input_ids rows {BxN}")

        # -------- индексы по блокам --------
        flag = dataset_flag.to(device).bool()     # (B, N), в блоке все одинаковы
        block_flag = flag[:, 0]                   # (B,) : True=дубли INPUTS, False=дубли DESC
        idx_all = torch.arange(BxN, device=device)
        idx_grid = idx_all.view(B, N)             # (B, N)

        # --------------- BERT: считаем только уникальные входы ----------------
        rep_inputs_idx = idx_grid[block_flag, 0]                          # (B_true,)
        unique_inputs_idx_mode2 = idx_grid[~block_flag, :].reshape(-1)    # (B_false*N,)
        idx_unique_inputs = torch.cat([unique_inputs_idx_mode2, rep_inputs_idx], dim=0)

        pos_in_compact = torch.full((BxN,), -1, dtype=torch.long, device=device)
        pos_in_compact[idx_unique_inputs] = torch.arange(idx_unique_inputs.numel(), device=device)

        map_inputs = torch.empty(BxN, dtype=torch.long, device=device)
        map_inputs[unique_inputs_idx_mode2] = pos_in_compact[unique_inputs_idx_mode2]
        if rep_inputs_idx.numel() > 0:
            rows_dup = idx_grid[block_flag, :].reshape(-1)
            rep_pos = pos_in_compact[rep_inputs_idx]                     # (B_true,)
            map_inputs[rows_dup] = rep_pos.repeat_interleave(N)

        if (map_inputs < 0).any():
            bad = (map_inputs < 0).nonzero(as_tuple=False).squeeze(-1)[:20]
            raise RuntimeError(
                f"map_inputs has -1 indices : {bad.tolist()}. "
                "Check dataset_flag/idx_unique_inputs mapping."
    )

        if inputs_embeds is not None:
            bert_outputs = self.bert(
                input_ids=None,
                inputs_embeds=inputs_embeds[idx_unique_inputs],                 # <— важно
                attention_mask=attention_mask[idx_unique_inputs] if attention_mask is not None else None,
                token_type_ids=token_type_ids[idx_unique_inputs] if token_type_ids is not None else None,
                position_ids=position_ids[idx_unique_inputs] if position_ids is not None else None,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        else:
            bert_outputs = self.bert(
                input_ids=input_ids[idx_unique_inputs],                         # fallback
                attention_mask=attention_mask[idx_unique_inputs] if attention_mask is not None else None,
                token_type_ids=token_type_ids[idx_unique_inputs] if token_type_ids is not None else None,
                position_ids=position_ids[idx_unique_inputs] if position_ids is not None else None,
                head_mask=head_mask,
                inputs_embeds=None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        seq_compact = bert_outputs.last_hidden_state                    # (U_inp, L, H)
        sequence_output = seq_compact[map_inputs]                       # (B*N, L, H)
        hidden_size = sequence_output.size(-1)


        unique_desc_idx = idx_grid[block_flag, :].reshape(-1)   # (B_true*N,)
        rep_desc_idx = idx_grid[~block_flag, 0]                 # (B_false,)
        idx_unique_desc = torch.cat([unique_desc_idx, rep_desc_idx], dim=0)  # (U_desc,)

        pos_in_compact = torch.full((BxN,), -1, dtype=torch.long, device=device)
        pos_in_compact[idx_unique_desc] = torch.arange(idx_unique_desc.numel(), device=device)
        map_desc = torch.empty((BxN,), dtype=torch.long, device=device)
        if unique_desc_idx.numel() > 0:
            map_desc[unique_desc_idx] = pos_in_compact[unique_desc_idx]
        if rep_desc_idx.numel() > 0:
            rows_dup = idx_grid[~block_flag, :].reshape(-1)          # все строки этих блоков
            rep_pos = pos_in_compact[rep_desc_idx]                   # (B_false,)
            map_desc[rows_dup] = rep_pos.repeat_interleave(N)        # раздаем на N строк

        if (map_desc < 0).any():
            bad = (map_desc < 0).nonzero(as_tuple=False).squeeze(-1)[:20]
            raise RuntimeError(f"map_desc has -1 indices: {bad.tolist()}")

        desc_out = self.desc_model(
                input_ids=desc_input_ids[idx_unique_desc],
                attention_mask=desc_attention_mask[idx_unique_desc],
                return_dict=True,
                )
        desc_pooled = desc_out.last_hidden_state[:, -1]
        desc_pooled = self.desc_proj(desc_pooled)                     # (B*N, H_gen)
        desc_pooled = desc_pooled.to(sequence_output.dtype) 
        desc_output = desc_pooled[map_desc]                        # (BxN, H_gen)

        sequence_output = sequence_output.contiguous()
        sequence_output[:, 0, :] = sequence_output[:, 0, :] + desc_output


        # ---------------- Encoder + classifier ----------------
        encoder_output = self.transformer_encoder(sequence_output)      # (B*N, L, H)
        logits = self.activation(self.classifier(encoder_output))       # (B*N, L, 1)

        # ---------------- ЛОСС (labels/mask уже (B*N, L, 1)) ----------------
        loss = None
        labels_reshaped = labels_mask_reshaped = cls_loss = mean_loss = diviation_loss = other_loss = None

        if labels is not None:
            labels_reshaped = labels.to(logits.device)
            labels_mask_reshaped = labels_mask.to(logits.device) if labels_mask is not None else None

            unreduced_loss = self.loss_fct(logits, labels_reshaped)  # (B*N, L, 1)

            if labels_mask_reshaped is not None and labels_mask_reshaped.sum().item() > 0:
                cls_mask = labels_mask_reshaped[:, 0:1, :]           # (B*N, 1, 1)
                other_mask = labels_mask_reshaped[:, 1:, :]          # (B*N, L-1, 1)

                if cls_mask.sum().item() > 0:
                        cls_loss = (unreduced_loss[:, 0:1, :] * cls_mask).sum() / (cls_mask.sum() + 1e-8)
                        mean_loss = None
                        diviation_loss = None

                if other_mask.sum().item() > 0:
                    other_loss = (unreduced_loss[:, 1:, :] * other_mask).sum() / (other_mask.sum() + 1e-8)

                if cls_loss is not None and other_loss is not None:
                    loss = cls_loss + self.weight * other_loss
                elif cls_loss is not None:
                    loss = cls_loss
                elif other_loss is not None:
                    loss = self.weight * other_loss

        if not return_dict:
            return (loss, logits)

        last_hidden = encoder_output # (B*N, L, H)
        hidden_states_out = (last_hidden,)

        return ExpressionModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states_out,
            attentions=bert_outputs.attentions,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            cls_loss=cls_loss,
            mean_loss=mean_loss,
            diviation_loss=diviation_loss,
            other_loss=other_loss
        )