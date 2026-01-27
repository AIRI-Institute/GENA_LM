import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModel, BertConfig 
from transformers.utils import cached_file
from huggingface_hub import login



@dataclass
class ExpressionModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None
    mean_loss: Optional[torch.FloatTensor] = None
    diviation_loss: Optional[torch.FloatTensor] = None


def average_pool(
    last_hidden_states: torch.Tensor,   # (B*N, L, H)
    attention_mask: torch.Tensor        # (B*N, L)
) -> torch.Tensor:                      # -> (B*N, H)
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    denom = attention_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return last_hidden.sum(dim=1) / denom


class ExpressionCounts(BertPreTrainedModel):
    """
    Размерности:
      - input_ids: (B, seq_len)
      - attention_mask: (B, seq_len)
      - token_type_ids: (B, seq_len)
      - desc_input_ids: (B, N, L_desc)
      - labels: (B, seq_len, N)
      - labels_mask: (B, seq_len, N)
    """

    def __init__(
        self,
        config,
        loss_fct=nn.MSELoss(reduction="none"),
        activation=nn.Identity(),
        hidden_size_desc=768,
        hidden_ff=1024,
        num_encoder_layers=3,
        nhead=8,
        weight=1,
        bert_cpt='/mnt/nfs_dna/DNALM/trained_models/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best.pth',
        hf: bool = False,
        desc_model_name: str = "intfloat/multilingual-e5-large-instruct",
        hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
    ):
        super().__init__(config)
        self.config = config
        self.hidden_size_desc = hidden_size_desc

        # GENA
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
            updated_state_dict = {
                k.replace("bert.", ""): v for k, v in state_dict.items()
            }

        if updated_state_dict is not None:
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

#         lora_cfg = LoraConfig(
#             task_type=TaskType.FEATURE_EXTRACTION,
#             r=8,
#             lora_alpha=16,
#             lora_dropout=0.05,
#             bias="none",
#             target_modules=["q_proj","k_proj","v_proj","o_proj"], 
# )
#         self.desc_model = get_peft_model(self.desc_model, lora_cfg)
#         self.desc_model.print_trainable_parameters()

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

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_ff,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Classifier
        self.classifier = nn.Linear(config.hidden_size, 1)

        # Loss
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight

        self.post_init()

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
        desc_input_ids=None,
        desc_attention_mask=None,
        desc_vectors=None,
        dataset_mean=None,
        dataset_deviation=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        sequence_output = bert_outputs.last_hidden_state              # (B, seq_len, H_gen)
        B, seq_len, hidden_size = sequence_output.shape

        assert desc_input_ids is not None and desc_attention_mask is not None, \
            "desc_input_ids и desc_attention_mask обязательны!"

        N = desc_input_ids.shape[1]
        desc_seq_len = desc_input_ids.shape[2]

        # (B, N, L_desc) -> (B*N, L_desc)
        desc_input_ids_2d = desc_input_ids.reshape(B * N, desc_seq_len)
        desc_attention_mask_2d = desc_attention_mask.reshape(B * N, desc_seq_len)

        
        desc_outputs = self.desc_model(
            input_ids=desc_input_ids_2d,
            attention_mask=desc_attention_mask_2d,
        )  # last_hidden_state: (B*N, L_desc, H_desc)


        # ibm
        # token_embeddings = desc_outputs[0]
        # desc_pooled = token_embeddings[:, 0] 
        
        # gemma
        # desc_pooled = average_pool(
        #     last_hidden_states=desc_outputs.last_hidden_state,
        #     attention_mask=desc_attention_mask_2d,
        # )  # (B*N, H_desc)

        # qwen
        desc_pooled = desc_outputs.last_hidden_state[:, -1]

        desc_pooled = self.desc_proj(desc_pooled)                     # (B*N, H_gen)
        desc_output = desc_pooled.reshape(B, N, hidden_size)          # (B, N, H_gen)

        # (B, seq_len, H) -> (B, N, seq_len, H)
        seq_out_expanded = sequence_output.unsqueeze(1).expand(-1, N, -1, -1)

        seq_out_expanded = seq_out_expanded.contiguous()
        seq_out_expanded[:, :, 0, :] = seq_out_expanded[:, :, 0, :].clone() + desc_output

        seq_out_flat = seq_out_expanded.reshape(B * N, seq_len, hidden_size)  # (B*N, seq_len, H)

        encoder_output = self.transformer_encoder(seq_out_flat)      # (B*N, seq_len, H)

        logits = self.classifier(encoder_output)                     # (B*N, seq_len, 1)
        logits = self.activation(logits)

        loss = None
        labels_reshaped = None
        labels_mask_reshaped = None
        cls_loss = None
        other_loss = None
        mean_loss = None
        diviation_loss = None

        if labels is not None:
            labels_reshaped = labels.permute(0, 2, 1).reshape(B * N, seq_len, 1).to(logits.device)
            labels_mask_reshaped = labels_mask.permute(0, 2, 1).reshape(B * N, seq_len, 1).to(logits.device)

            unreduced_loss = self.loss_fct(logits, labels_reshaped)  # (B*N, seq_len, 1)

            if labels_mask_reshaped.sum() > 0:
                cls_mask = labels_mask_reshaped[:, 0:1, :]           # (B*N, 1, 1)
                other_mask = labels_mask_reshaped[:, 1:, :]          # (B*N, seq_len-1, 1)

                if cls_mask.sum() > 0:
                    cls_loss = (unreduced_loss[:, 0:1, :] * cls_mask).sum() / cls_mask.sum()

                if other_mask.sum() > 0:
                    other_loss = (unreduced_loss[:, 1:, :] * other_mask).sum() / other_mask.sum()

                if cls_loss is not None and other_loss is not None:
                    loss = cls_loss + self.weight * other_loss
                elif cls_loss is not None:
                    loss = cls_loss
                elif other_loss is not None:
                    loss = self.weight * other_loss

        if not return_dict:
            return (loss, logits)

        return ExpressionModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            cls_loss=cls_loss,
            mean_loss=mean_loss,
            diviation_loss=diviation_loss,
            other_loss=other_loss,
        )

