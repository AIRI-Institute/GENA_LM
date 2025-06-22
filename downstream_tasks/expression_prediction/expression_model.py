import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional
from dataclasses import dataclass
from transformers import AutoConfig
import numpy as np

import torch.nn.functional as F

@dataclass
class ExpressionModelOutput(TokenClassifierOutput):
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None
    mean_loss: Optional[torch.FloatTensor] = None
    diviation_loss: Optional[torch.FloatTensor] = None

class FlashTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm       = nn.LayerNorm(d_model)
        self.dropout        = nn.Dropout(dropout)

        # feed-forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.nhead = nhead
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask=None):
        # --- Flash self-attention block (pre-norm) ---
        x = self.self_attn_norm(src)                                          # (B, S, D)
        # note: PyTorch expects shape (S, B, D) for multi-head, so we transpose/untranspose
        x_t = x.transpose(0, 1)                                               # (S, B, D)
        attn_out = F.scaled_dot_product_attention(
            x_t, x_t, x_t,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False
        )                                                                     # (S, B, D)
        attn_out = attn_out.transpose(0, 1)                                   # (B, S, D)
        src = src + self.dropout(attn_out)

        # --- Feed-forward block (pre-norm) ---
        y = self.ffn_norm(src)
        y = self.linear2(self.dropout(F.gelu(self.linear1(y))))
        src = src + self.dropout(y)
        return src

class ExpressionCounts(BertPreTrainedModel):
    """
    Размерности:
      - input_ids: (B, seq_len)
      - attention_mask: (B, seq_len)
      - token_type_ids: (B, seq_len) [опционально]
      - desc_vectors: (B, N, hidden_size)        
      - labels: (B, seq_len, N) -> приводим к (B*N, seq_len, 1)
      - labels_mask: (B, seq_len, N) -> приводим к (B*N, seq_len, 1)
    
    Шаги:
      1) Прогоняем через GENA -> (B, seq_len, hidden_size)
      2) Расширяем выход -> (B, N, seq_len, hidden_size)
      3) Прогоняем desc_vectors через MLP 
      4) Складываем desc_vectors с CLS
      5) Превращаем (B, N, seq_len, hidden_size) -> (B*N, seq_len, hidden_size)
      6) Прогоняем через Encoder
      7) classifier -> (B*N, seq_len, 1)
      8) Меняем labels и labels_mask -> (B*N, seq_len, 1), считаем loss
    """

    def __init__(
        self,
        config,
        loss_fct=nn.MSELoss(reduction="none"),
        activation = nn.Identity(),
        hidden_size_desc = 768,
        hidden_ff = 1024,
        num_encoder_layers = 12,
        nhead = 12,
        weight = 1,
        bert_cpt = None,
        cell_type_specific_loss_fn = None
    ):
        super().__init__(config)

        model_name  = 'AIRI-Institute/gena-lm-bert-base-t2t'
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Initialize model components
        self.hidden_size_desc = hidden_size_desc
        model_hidden_size = self.config.hidden_size  # This will be 1024 for BERT-large

        # Initialize all model components
        if bert_cpt is None:
            print(f"No checkpoint provided. Loading BERT weights from HuggingFace model: {model_name}")
            self.bert = BertModel.from_pretrained(model_name, config=self.config, add_pooling_layer=False)
        else:
            self.bert = BertModel(config, add_pooling_layer=False)

        self.desc_fc = nn.Sequential(
            nn.Linear(self.hidden_size_desc, model_hidden_size),
            nn.LeakyReLU(),
            nn.Linear(model_hidden_size, model_hidden_size),
        )

        # 3) Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_ff,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        self.classifier = nn.Linear(model_hidden_size, 1)
        
        # Initialize new layers with Xavier/Glorot initialization
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        self.transformer_encoder.apply(_init_weights)
        self.desc_fc.apply(_init_weights)
        self.classifier.apply(_init_weights)

        # Load checkpoint if provided
        if bert_cpt is not None:
            print(f"Loading checkpoint from {bert_cpt}")
            try:
                # First try loading with weights_only=True and numpy support
                import torch.serialization
                with torch.serialization.safe_globals(['numpy.core.multiarray.scalar']):
                    checkpoint = torch.load(bert_cpt, map_location='cpu', weights_only=True)
                    print(f"Successfully loaded checkpoint with weights_only=True, checkpoint type: {type(checkpoint)}")
            except Exception as e:
                print(f"Warning: Failed to load with weights_only=True, attempting legacy load: {str(e)}")
                # Fallback to legacy loading if the checkpoint is from an older version
                checkpoint = torch.load(bert_cpt, map_location='cpu', weights_only=False)
                print(f"Successfully loaded checkpoint with legacy method, checkpoint type: {type(checkpoint)}")
            
            # Print checkpoint keys to verify structure
            if isinstance(checkpoint, dict):
                print(f"Checkpoint top-level keys: {list(checkpoint.keys())}")
            
            # Handle both full model checkpoint and BERT-only checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full model checkpoint
                print("Loading full model checkpoint with 'model_state_dict'")
                state_dict = checkpoint['model_state_dict']
                missing_k, unexpected_k = self.load_state_dict(state_dict, strict=True)
                   # Add after loading checkpoint
                print("Descriptor FC weights loaded:", torch.sum(self.desc_fc[0].weight).item())
                print("Transformer Encoder weights loaded:", torch.sum(self.transformer_encoder[0].self_attn.in_proj_weight).item())
                print("Classifier weights loaded:", torch.sum(self.classifier.weight).item())
            else:
                # BERT-only checkpoint
                print("Loading BERT-only checkpoint")
                if isinstance(checkpoint, dict):
                    state_dict = {k.replace('bert.', ''): v for k, v in checkpoint.items()}
                else:
                    state_dict = checkpoint
                missing_k, unexpected_k = self.bert.load_state_dict(state_dict, strict=False)
            
            if len(missing_k) > 0:
                print(f'Warning: {len(missing_k)} keys were not loaded from checkpoint!')
                print(f'First few missing keys: {missing_k[:5]}')
            if len(unexpected_k) > 0:
                print(f'Warning: {len(unexpected_k)} unexpected keys in checkpoint!')
                print(f'First few unexpected keys: {unexpected_k[:5]}')

        # Verify model loaded weights properly
        sample_weight = next(self.bert.parameters())
        print(f"Model loaded with weights (sample mean: {sample_weight.mean().item():.6f})")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")

        # Loss setup
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight
        self.cell_type_specific_loss_fn = cell_type_specific_loss_fn        

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
        meta_input_ids=None,
        meta_attention_mask=None,
        desc_vectors = None,
        dataset_mean = None,
        dataset_deviation = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Прогоняем через GENA
        bert_outputs = self.bert(
            input_ids = input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # (B, seq_len, hidden_size)
        sequence_output = bert_outputs.last_hidden_state
        B, seq_len, hidden_size = sequence_output.shape

        # Предполагаем, что desc_vectors.shape -> (B, N, hidden_size)
        N = desc_vectors.shape[1]

        # Расширяем выход
        # (B, seq_len, hidden_size) -> (B, N, seq_len, hidden_size)      
        seq_out_expanded = sequence_output.unsqueeze(1).expand(-1, N, -1, -1)

        # Прогоняем desc_vectors через MLP 
        # (B, N, hidden_size) -> (B*N, hidden_size)
        desc_vectors_2d = desc_vectors.reshape(B*N, self.hidden_size_desc)  
        desc_fc_output = self.desc_fc(desc_vectors_2d)  # (B*N, hidden_size)
        # (B*N, hidden_size) -> (B, N, hidden_size)
        desc_fc_output = desc_fc_output.reshape(B, N, hidden_size)

        # Складываем desc_vectors с CLS
        # CLS-токен — seq_out_expanded[:, :, 0, :]  (B, N, hidden_size)
        seq_out_expanded = seq_out_expanded.contiguous()
        seq_out_expanded[:, :, 0, :] = seq_out_expanded[:, :, 0, :].clone() + desc_fc_output

        # (B, N, seq_len, hidden_size) -> (B*N, seq_len, hidden_size)
        seq_out_flat = seq_out_expanded.reshape(B*N, seq_len, hidden_size)

        # Прогоняем через Encoder
        encoder_output = self.transformer_encoder(seq_out_flat)  # (B*N, seq_len, hidden_size)

        # Classifier -> (B*N, seq_len, 1)
        logits = self.classifier(encoder_output)
        logits = self.activation(logits)

        # Loss
        loss = None
        if labels is not None:
            # labels, labels_mask: (B, seq_len, N)
            # Нужно:   (B*N, seq_len, 1)
            # 1) permute(0,2,1) -> (B, N, seq_len)
            # 2) reshape -> (B*N, seq_len)
            # 3) unsqueeze -> (B*N, seq_len, 1)
            labels_reshaped = labels.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)
            labels_mask_reshaped = labels_mask.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)

            # loss
            # Cчитаем общий лосс
            unreduced_loss = self.loss_fct(logits, labels_reshaped)  # (B*N, seq_len, 1)

            if labels_mask_reshaped.sum() > 0:
                # Разделяем маску на CLS и остальные токены
                cls_mask = labels_mask_reshaped[:, 0:1, :]  # (B*N, 1, 1)
                other_mask = labels_mask_reshaped[:, 1:, :]  # (B*N, seq_len-1, 1)

                # if torch.isnan(labels_reshaped[:, 0:1, :].reshape(B,N)).any():
                #     print (f"cls_mask: {cls_mask}")
                #     print (f"labels: {labels_reshaped[:, 0:1, :].reshape(B,N)}")
                #     raise ValueError("labels_reshaped contains NaN values")
                # Считаем лосс для CLS
                cls_loss = None
                if cls_mask.sum() > 0:
                    if self.cell_type_specific_loss_fn is not None:
                        cls_loss, mean_loss, diviation_loss = self.cell_type_specific_loss_fn(
                            cls_targets = labels_reshaped[:, 0:1, :].reshape(B,N),
                            cls_preds = logits[:, 0:1, :].reshape(B,N),
                            cls_mask = cls_mask.reshape(B,N),
                            dataset_mean = dataset_mean,
                            dataset_deviation = dataset_deviation
                        )
                    else:
                        cls_loss = (unreduced_loss[:, 0:1, :] * cls_mask).sum() / cls_mask.sum()
                        mean_loss = None
                        diviation_loss = None

                # Считаем лосс для остальных токенов
                other_loss = None
                if other_mask.sum() > 0:
                    other_loss = (unreduced_loss[:, 1:, :] * other_mask).sum() / other_mask.sum()

                # Объединяем лоссы
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
            other_loss=other_loss
        )


class cell_type_specific_loss_fn(nn.Module):
    def __init__(self, 
                 weight_mean,
                 loss_fct_mean = nn.MSELoss(reduction="none"), 
                 loss_fct_diviation = nn.MSELoss(reduction="none"),
                 normalize_by_mean = True,
                 ):
        super().__init__()
        assert 0 <= weight_mean <= 1, "weight_mean must be between 0 and 1"
        self.loss_fct_mean = loss_fct_mean
        self.loss_fct_diviation = loss_fct_diviation
        self.weight_mean = weight_mean
        self.weight_diviation = 1 - weight_mean
        self.normalize_by_mean = normalize_by_mean

    def forward(self, cls_targets, cls_preds, cls_mask,
                dataset_mean,
                dataset_deviation):
        # check that cls_mask.sum(dim=1) != 0
        if cls_mask.sum(dim=1).eq(0).any():
            raise ValueError("cls_mask.sum(dim=1) is 0 for some samples. This case is not supported.")
            # this might happen if we have coverage data for a region where there is no tpm
            # to handle this, we need to modify division by cls_mask.sum(dim=1) to be a sum over non-zero elements
        
        # mean across cell types
        cls_targets_mean = (cls_targets * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_targets_mean = cls_targets_mean.reshape(cls_targets_mean.shape[0],1)
        cls_preds_mean = (cls_preds * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_preds_mean = cls_preds_mean.reshape(cls_preds_mean.shape[0],1)

        # TODO: DEBUG, remove at some point
        # Check that dataset_mean tensor is close to cls_targets_mean tensor
        if not torch.allclose(dataset_mean, torch.squeeze(cls_targets_mean), atol=1e-6, rtol=1e-5):
            print (dataset_mean)
            print (torch.squeeze(cls_targets_mean))
            print ("----CLS mask:---")
            print (cls_mask)
            raise ValueError(f"dataset_mean tensor is not close to cls_targets_mean tensor. "
                f"Max difference: {torch.max(torch.abs(dataset_mean - torch.squeeze(cls_targets_mean))) :.6f}")
        # normalize by mean
        if self.normalize_by_mean:
            cls_targets_diviation = ((cls_targets - cls_targets_mean) / cls_targets_mean)
            cls_preds_diviation = ((cls_preds - cls_preds_mean) / cls_preds_mean)
            debug_test_cls_targets_diviation = cls_targets_diviation
        else:
            cls_targets_diviation = (cls_targets - cls_targets_mean)
            cls_preds_diviation = (cls_preds - cls_preds_mean)
            debug_test_cls_targets_diviation = ((cls_targets - cls_targets_mean) / cls_targets_mean)

        # TODO: DEBUG, remove at some point
        if not torch.allclose(dataset_deviation[cls_mask.bool()], debug_test_cls_targets_diviation[cls_mask.bool()], atol=1e-6, rtol=1e-5):
            # shape is (B, N)
            # find batch where allclose is False
            for batch_idx in range(cls_mask.shape[0]):
                if not torch.allclose(dataset_deviation[batch_idx], debug_test_cls_targets_diviation[batch_idx], atol=1e-6, rtol=1e-5):
                    print (f"Found batch_idx: {batch_idx}")
                    break
            # provide some debug information
            print (f"dataset_deviation tensor is not close to cls_targets_deviation tensor. ")
            print (f"batch_idx: {batch_idx}")
            print (f"dataset_mean: {dataset_mean[batch_idx]}")
            print (f"cls_targets_mean: {cls_targets_mean[batch_idx]}")
            print (f"dataset_deviation: {dataset_deviation[batch_idx]}")
            print (f"cls_targets_diviation: {cls_targets_diviation[batch_idx]}")
            print (f"cls_mask: {cls_mask[batch_idx]}")

            raise ValueError(f"dataset_deviation tensor is not close to cls_targets_deviation tensor. "
                           f"Max difference: {torch.max(torch.abs(dataset_deviation[cls_mask.bool()] - cls_targets_diviation[cls_mask.bool()])):.6f}")

        # loss
        cls_loss_mean = (self.loss_fct_mean(cls_preds_mean, cls_targets_mean) * cls_mask).sum() / cls_mask.sum()
        cls_loss_diviation = (self.loss_fct_diviation(cls_preds_diviation, cls_targets_diviation) * cls_mask).sum() / cls_mask.sum()
        full_loss = self.weight_mean * cls_loss_mean + self.weight_diviation * cls_loss_diviation

        return full_loss, cls_loss_mean, cls_loss_diviation
