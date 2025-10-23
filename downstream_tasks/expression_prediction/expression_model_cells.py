import torch
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
    
class QuantileLoss(nn.Module):
    def __init__(self, quantile: float, reduction: str = "none"):
        super().__init__()
        if not (0 < quantile < 1):
            raise ValueError("quantile must be in (0, 1)")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        self.q = quantile
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        errors = target - preds
        loss = torch.max(self.q * errors, (self.q - 1) * errors)  
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        else:  
            return loss.sum()
            
class OneHotEncoder(nn.Module):
    def __init__(self, max_N_cell_types, hidden_size_desc=768):
        super().__init__()
        self.max_N_cell_types = max_N_cell_types
        self.embedding = nn.Embedding(self.max_N_cell_types, hidden_size_desc)
        self.fc = nn.Linear(hidden_size_desc, hidden_size_desc)
        self.activation = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.embedding(x.long())
        x = self.fc(x)
        x = self.activation(x)
        x = x.reshape(x.shape[0], -1)
        return x

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
        cell_type_specific_loss_fn = None,
        text_model = None,
        hf: bool = False,
        hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
    ):
        super().__init__(config)
        self.config = config
        self.hidden_size_desc = hidden_size_desc 

        # 1) GENA (BERT)
        if hf:
            hf_config = BertConfig.from_pretrained(hf_model_name)
            self.bert = BertModel(hf_config, add_pooling_layer=False)
            weights_path = cached_file(hf_model_name, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")
            updated_state_dict = {
                k.replace("bert.", ""): v for k, v in state_dict.items() if k.startswith("bert.")
            }

            missing_k, unexpected_k = self.bert.load_state_dict(updated_state_dict, strict=False)
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

        if text_model is not None:
            self.desc_fc = text_model
        else:
            # 2) MLP для desc_vectors
            self.desc_fc = nn.Sequential(
                nn.Linear(self.hidden_size_desc, config.hidden_size),
                nn.LeakyReLU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )

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
        self.cell_type_specific_loss_fn = cell_type_specific_loss_fn

        self.post_init()

    def save_cell_type_embeddings(self, save_path, cell_type_names=None):
        embeddings = self.cell_type_embedding.weight.data.cpu()
        save_dict = {
            'embeddings': embeddings,
            'embedding_dim': embeddings.shape[1],
            'num_cell_types': embeddings.shape[0],
            'config_max_N_cell_types': self.max_N_cell_types
        }
        if cell_type_names is not None:
            if len(cell_type_names) != embeddings.shape[0]:
                print(f"Warning: Number of cell type names ({len(cell_type_names)}) "
                      f"doesn't match number of embeddings ({embeddings.shape[0]})")
            else:
                save_dict['cell_type_names'] = cell_type_names
        torch.save(save_dict, save_path)
        print(f"Cell type embeddings saved to {save_path}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding statistics - Mean: {embeddings.mean().item():.4f}, Std: {embeddings.std().item():.4f}")
        return save_dict

    def load_cell_type_embeddings(self, load_path):
        loaded_dict = torch.load(load_path, map_location='cpu')
        if loaded_dict['embeddings'].shape != self.cell_type_embedding.weight.shape:
            raise ValueError(f"Loaded embedding shape {loaded_dict['embeddings'].shape} "
                           f"doesn't match model embedding shape {self.cell_type_embedding.weight.shape}")
        self.cell_type_embedding.weight.data = loaded_dict['embeddings'].to(self.cell_type_embedding.weight.device)
        print(f"Cell type embeddings loaded from {load_path}")
        print(f"Embedding shape: {loaded_dict['embeddings'].shape}")
        return loaded_dict

    def get_cell_type_embeddings(self, cell_type_indices=None):
        embeddings = self.cell_type_embedding.weight.data
        if cell_type_indices is not None:
            embeddings = embeddings[cell_type_indices]
        return embeddings

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
        meta_input_ids=None,
        meta_attention_mask=None,
        desc_vectors=None,           # (B, N, D)
        dataset_mean=None,           # (B,)    (исп. только в спец-лоссе)
        dataset_deviation=None,      # (B, N)  (исп. только в спец-лоссе)
        dataset_flag=None,           # (B, N): 1 -> дубли INPUTS; 0 -> дубли DESC
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if dataset_flag is None:
            raise ValueError("dataset_flag must be provided and shaped (B, N)")

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

        # ---------------- BERT: считаем только уникальные входы ----------------
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

        # ---------------- desc_fc: по правилу блоков ----------------
        if desc_vectors is not None:
            _BN, D = desc_vectors.shape
            if _BN != BxN:
                raise ValueError(f"desc_vectors (B*N,D) has first dim {_BN}, but B*N is {BxN}")

            desc_fc_out = torch.empty((BxN, hidden_size), device=device, dtype=sequence_output.dtype)

            # flag==1: описания разные — считаем каждую строку
            idx_desc_unique = idx_grid[block_flag, :].reshape(-1)
            if idx_desc_unique.numel() > 0:
                desc_fc_out[idx_desc_unique] = self.desc_fc(desc_vectors[idx_desc_unique])

            # flag==0: описания одинаковые внутри блока — считаем 1 раз на блок и раздаём
            idx_blocks_same_desc = torch.nonzero(~block_flag, as_tuple=False).squeeze(-1)  # (B_false,)
            if idx_blocks_same_desc.numel() > 0:
                reps = idx_grid[idx_blocks_same_desc, 0]                                   # (B_false,)
                rep_out = self.desc_fc(desc_vectors[reps])                            # (B_false, H)
                rows = idx_grid[idx_blocks_same_desc, :].reshape(-1)                       # все строки этих блоков
                desc_fc_out[rows] = rep_out.repeat_interleave(N, dim=0)

            # прибавляем к CLS
            sequence_output = sequence_output.contiguous()
            sequence_output[:, 0, :] = sequence_output[:, 0, :].clone() + desc_fc_out

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
                    if self.cell_type_specific_loss_fn is not None and dataset_mean is not None and dataset_deviation is not None:
                        cls_t = labels_reshaped[:, 0:1, :].reshape(B, N)
                        cls_p = logits[:, 0:1, :].reshape(B, N)
                        cls_m = cls_mask.reshape(B, N).float()
                        cls_loss, mean_loss, diviation_loss = self.cell_type_specific_loss_fn(
                            cls_targets=cls_t,
                            cls_preds=cls_p,
                            cls_mask=cls_m,
                            dataset_mean=dataset_mean,
                            dataset_deviation=dataset_deviation
                        )
                    else:
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
        if cls_mask.sum(dim=1).eq(0).any():
            raise ValueError("cls_mask.sum(dim=1) is 0 for some samples. This case is not supported.")

        cls_targets_mean = (cls_targets * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_targets_mean = cls_targets_mean.reshape(cls_targets_mean.shape[0],1)
        cls_preds_mean = (cls_preds * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
        cls_preds_mean = cls_preds_mean.reshape(cls_preds_mean.shape[0],1)

        if not torch.allclose(dataset_mean, torch.squeeze(cls_targets_mean), atol=1e-6, rtol=1e-5):
            raise ValueError(f"dataset_mean tensor is not close to cls_targets_mean tensor. "
                f"Max difference: {torch.max(torch.abs(dataset_mean - torch.squeeze(cls_targets_mean))) :.6f}")

        if self.normalize_by_mean:
            cls_targets_diviation = ((cls_targets - cls_targets_mean) / cls_targets_mean)
            cls_preds_diviation = ((cls_preds - cls_preds_mean) / cls_preds_mean)
            debug_test_cls_targets_diviation = cls_targets_diviation
        else:
            cls_targets_diviation = (cls_targets - cls_targets_mean)
            cls_preds_diviation = (cls_preds - cls_preds_mean)
            debug_test_cls_targets_diviation = ((cls_targets - cls_targets_mean) / cls_targets_mean)

        if not torch.allclose(dataset_deviation[cls_mask.bool()], debug_test_cls_targets_diviation[cls_mask.bool()], atol=1e-6, rtol=1e-5):
            for batch_idx in range(cls_mask.shape[0]):
                if not torch.allclose(dataset_deviation[batch_idx], debug_test_cls_targets_diviation[batch_idx], atol=1e-6, rtol=1e-5):
                    print (f"Found batch_idx: {batch_idx}")
                    break
            print (f"dataset_deviation tensor is not close to cls_targets_deviation tensor. ")
            print (f"batch_idx: {batch_idx}")
            print (f"dataset_mean: {dataset_mean[batch_idx]}")
            print (f"cls_targets_mean: {cls_targets_mean[batch_idx]}")
            print (f"dataset_deviation: {dataset_deviation[batch_idx]}")
            print (f"cls_targets_diviation: {cls_targets_diviation[batch_idx]}")
            print (f"cls_mask: {cls_mask[batch_idx]}")

            raise ValueError(f"dataset_deviation tensor is not close to cls_targets_deviation tensor. "
                           f"Max difference: {torch.max(torch.abs(dataset_deviation[cls_mask.bool()] - cls_targets_diviation[cls_mask.bool()])):.6f}")

        cls_loss_mean = (self.loss_fct_mean(cls_preds_mean, cls_targets_mean) * cls_mask).sum() / cls_mask.sum()
        cls_loss_diviation = (self.loss_fct_diviation(cls_preds_diviation, cls_targets_diviation) * cls_mask).sum() / cls_mask.sum()
        full_loss = self.weight_mean * cls_loss_mean + self.weight_diviation * cls_loss_diviation

        return full_loss, cls_loss_mean, cls_loss_diviation
