import torch
import torch.nn as nn
import importlib
import os, logging
from transformers.modeling_outputs import TokenClassifierOutput
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModel, BertConfig, AutoConfig

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

class ExpressionCounts(nn.Module):
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
        num_encoder_layers = 3,
        nhead = 8,
        weight = 1,
        bert_cpt = '/mnt/nfs_dna/DNALM/trained_models/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best.pth',
        cell_type_specific_loss_fn = None,
        text_model = None,
        hf: bool = False,
        hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
    ):
        super().__init__()
        self.hidden_size_desc = hidden_size_desc 
        self.hidden_size = 768
        self.logger = logging.getLogger(__name__)
		
        # 1) GENA
        if hf:
            # for moderngena (minja)
            from modernbert_utils import load_flexbert_model
            self.logger.info(f"Loading ModernBERT model from {hf_model_name}")
            self.bert = load_flexbert_model(hf_model_name, logger=self.logger)
            self.is_modernbert_model = True
        #     config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        #     super().__init__(config)
        #     self.bert = AutoModel.from_pretrained(
        #         hf_model_name,
        #         config = config,
        #         trust_remote_code=True, 
        #     )
        #     # проверка загруженных весов
        #     missing_k, unexpected_k = self.bert.load_state_dict(self.bert.state_dict(), strict=False)


        # if len(missing_k) != 0:
        #     print(f"{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.")
        # if len(unexpected_k) != 0:
        #     print(f"{unexpected_k} were found in checkpoint, but model is not expecting them!")

        if text_model is not None:
            self.desc_fc = text_model
        else:
            # 2) MLP для desc_vectors
            self.desc_fc = nn.Sequential(
                nn.Linear(self.hidden_size_desc, self.hidden_size),
                nn.LeakyReLU(),
            # nn.Dropout(p=0.1),
                nn.Linear(self.hidden_size, self.hidden_size),
            )

        # 3) Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_ff,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4) Classifier
        self.classifier = nn.Linear(self.hidden_size, 1)

        # 5) Loss
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight
        self.cell_type_specific_loss_fn = cell_type_specific_loss_fn

        # self.post_init()

    def save_cell_type_embeddings(self, save_path, cell_type_names=None):
        """
        Save the learned cell type embeddings from the nn.Embedding layer.
        
        Args:
            save_path (str): Path where to save the embeddings (e.g., 'cell_type_embeddings.pt')
            cell_type_names (list, optional): List of cell type names corresponding to embedding indices.
                                            If None, will use indices as names.
        """
        # Get the embedding weights from the separate embedding layer
        embeddings = self.cell_type_embedding.weight.data.cpu()  # Shape: (num_embeddings, embedding_dim)
        
        # Create a dictionary to save
        save_dict = {
            'embeddings': embeddings,
            'embedding_dim': embeddings.shape[1],
            'num_cell_types': embeddings.shape[0],
            'config_max_N_cell_types': self.max_N_cell_types
        }
        
        # Add cell type names if provided
        if cell_type_names is not None:
            if len(cell_type_names) != embeddings.shape[0]:
                print(f"Warning: Number of cell type names ({len(cell_type_names)}) "
                      f"doesn't match number of embeddings ({embeddings.shape[0]})")
            else:
                save_dict['cell_type_names'] = cell_type_names
        
        # Save the embeddings
        torch.save(save_dict, save_path)
        print(f"Cell type embeddings saved to {save_path}")
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding statistics - Mean: {embeddings.mean().item():.4f}, Std: {embeddings.std().item():.4f}")
        
        return save_dict

    def load_cell_type_embeddings(self, load_path):
        """
        Load cell type embeddings from a saved file.
        
        Args:
            load_path (str): Path to the saved embeddings file
            
        Returns:
            dict: Dictionary containing the loaded embeddings and metadata
        """
        # Load the saved embeddings
        loaded_dict = torch.load(load_path, map_location='cpu')
        
        # Check if dimensions match
        if loaded_dict['embeddings'].shape != self.cell_type_embedding.weight.shape:
            raise ValueError(f"Loaded embedding shape {loaded_dict['embeddings'].shape} "
                           f"doesn't match model embedding shape {self.cell_type_embedding.weight.shape}")
        
        # Load the embeddings into the model
        self.cell_type_embedding.weight.data = loaded_dict['embeddings'].to(self.cell_type_embedding.weight.device)
        
        print(f"Cell type embeddings loaded from {load_path}")
        print(f"Embedding shape: {loaded_dict['embeddings'].shape}")
        
        return loaded_dict

    def get_cell_type_embeddings(self, cell_type_indices=None):
        """
        Get the current cell type embeddings.
        
        Args:
            cell_type_indices (list, optional): Specific indices to retrieve. If None, returns all embeddings.
            
        Returns:
            torch.Tensor: Cell type embeddings
        """
        embeddings = self.cell_type_embedding.weight.data
        
        if cell_type_indices is not None:
            embeddings = embeddings[cell_type_indices]
            
        return embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels_mask=None,
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

        return_dict = True

        # Прогоняем через GENA
        bert_outputs = self.bert(
            input_ids = input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        # Notaton:
        # B - batch size
        # N - number of cell types (a.k.a experiment descriptors)
        # seq_len - sequence length (number of tokens in the input sequence)
        # hidden_size - hidden size

        # (B, seq_len, hidden_size)

        sequence_output = bert_outputs
        
        B, seq_len, hidden_size = sequence_output.shape # (B, seq_len, hidden_size)

        # Assuming that desc_vectors.shape -> (B, N, hidden_size_desc), where N - number of cell types (a.k.a experiment descriptors)
        N = desc_vectors.shape[1]

        # Расширяем выход
        # (B, seq_len, hidden_size) -> (B, N, seq_len, hidden_size)      
        seq_out_expanded = sequence_output.unsqueeze(1).expand(-1, N, -1, -1) # B, N, seq_len, hidden_size

        # Прогоняем desc_vectors через MLP 
        # (B, N, hidden_size) -> (B*N, hidden_size)
        desc_vectors_2d = desc_vectors.reshape(B*N, desc_vectors.size()[-1])
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
            hidden_states=encoder_output,
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

# import torch
# import torch.nn as nn
# from transformers.modeling_outputs import TokenClassifierOutput
# from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
# from typing import Optional
# from dataclasses import dataclass
# from transformers import AutoModel, AutoConfig

# @dataclass
# class ExpressionModelOutput(TokenClassifierOutput):
#     labels_reshaped: Optional[torch.FloatTensor] = None
#     labels_mask_reshaped: Optional[torch.FloatTensor] = None
#     cls_loss: Optional[torch.FloatTensor] = None
#     other_loss: Optional[torch.FloatTensor] = None
#     mean_loss: Optional[torch.FloatTensor] = None
#     diviation_loss: Optional[torch.FloatTensor] = None

# class QuantileLoss(nn.Module):
#     def __init__(self, quantile: float, reduction: str = "none"):
#         super().__init__()
#         if not (0 < quantile < 1):
#             raise ValueError("quantile must be in (0, 1)")
#         if reduction not in ("none", "mean", "sum"):
#             raise ValueError("reduction must be 'none', 'mean', or 'sum'")
#         self.q = quantile
#         self.reduction = reduction

#     def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         errors = target - preds
#         loss = torch.max(self.q * errors, (self.q - 1) * errors)
#         if self.reduction == "none":
#             return loss
#         elif self.reduction == "mean":
#             return loss.mean()
#         else:
#             return loss.sum()

# class OneHotEncoder(nn.Module):
#     def __init__(self, max_N_cell_types, hidden_size_desc=768):
#         super().__init__()
#         self.max_N_cell_types = max_N_cell_types
#         self.embedding = nn.Embedding(self.max_N_cell_types, hidden_size_desc)
#         self.fc = nn.Linear(hidden_size_desc, hidden_size_desc)
#         self.activation = nn.LeakyReLU()

#     def forward(self, x):
#         x = self.embedding(x.long())
#         x = self.fc(x)
#         x = self.activation(x)
#         x = x.reshape(x.shape[0], -1)
#         return x


# class ExpressionCounts(BertPreTrainedModel):
#     """
#     Важные размеры:
#       - input_ids: (B, seq_len)
#       - attention_mask: (B, seq_len)
#       - desc_vectors: (B, N, hidden_size_desc)
#       - labels: (B, seq_len, N)         -> приводим к (B*N, seq_len, 1)
#       - labels_mask: (B, seq_len, N)    -> приводим к (B*N, seq_len, 1)

#     Учим/мерим только по CLS (позиция 0 по seq_len).
#     """

#     def __init__(
#         self,
#         config,
#         loss_fct=nn.MSELoss(reduction="none"),
#         activation=nn.Identity(),
#         hidden_size_desc=768,
#         hidden_ff=1024,
#         num_encoder_layers=3,
#         nhead=8,
#         weight=1,
#         bert_cpt='/mnt/nfs_dna/DNALM/trained_models/bert_base_512_t2t_1000G_bs256_lr_1e-04_fp16/model_best.pth',
#         cell_type_specific_loss_fn=None,
#         text_model=None,
#         hf: bool = False,
#         hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
#     ):
#         def _tensor_stats(x: torch.Tensor, name: str):
#             x_flat = x.detach().float().view(-1)
#             finite = torch.isfinite(x_flat)
#             n_all = x_flat.numel()
#             n_fin = int(finite.sum().item())
#             if n_fin > 0:
#                 x_f = x_flat[finite]
#                 print(f"[STAT] {name}: shape={tuple(x.shape)} dtype={x.dtype} "
#                     f"finite={n_fin}/{n_all} min={float(x_f.min().item()):.6g} "
#                     f"max={float(x_f.max().item()):.6g} mean={float(x_f.mean().item()):.6g}")
#             else:
#                 print(f"[STAT] {name}: shape={tuple(x.shape)} dtype={x.dtype} finite=0/{n_all} (все NaN/Inf)")

#         def _print_example_layers(model: nn.Module, max_print: int = 5):
#             print("[MODEL] Примеры слоёв и параметров:")
#             printed = 0
#             for name, module in model.named_modules():
#                 # интересны линейные/LayerNorm/attention-подобные
#                 if isinstance(module, (nn.Linear, nn.LayerNorm)):
#                     print(f"  [LAYER] {name}: {module.__class__.__name__}")
#                     for p_name, p in module.named_parameters(recurse=False):
#                         _tensor_stats(p.data, f"{name}.{p_name}")
#                     printed += 1
#                 if printed >= max_print:
#                     break

#         def _print_attn_impl(model):
#             # попробуем найти поле конфигурации или атрибуты в модулях внимания
#             attn_found = False
#             for name, module in model.named_modules():
#                 if hasattr(module, "attn_implementation"):
#                     print(f"[ATTN] {name}.attn_implementation = {getattr(module, 'attn_implementation')}")
#                     attn_found = True
#             if not attn_found:
#                 print("[ATTN] Не найден явный атрибут attn_implementation на подмодулях (это не критично).")
#         config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
#         print(f"[HF] Загружаю модель: {hf_model_name}")
#         print(f"[HF] Тип конфига: {config.__class__.__name__}")
#         print(f"[HF] Ключевые размеры: hidden_size={getattr(config, 'hidden_size', 'N/A')}, "
#             f"num_hidden_layers={getattr(config, 'num_hidden_layers', 'N/A')}, "
#             f"num_attention_heads={getattr(config, 'num_attention_heads', 'N/A')}")
#         super().__init__(config)
#         self.bert = AutoModel.from_pretrained(
#                 hf_model_name,
#                 config=config,
#                 trust_remote_code=True,
#                 attn_implementation="sdpa",
#             )

#         print(f"[HF] Тип модели: {self.bert.__class__.__name__}")
#         missing_k, unexpected_k = self.bert.load_state_dict(self.bert.state_dict(), strict=False)

#         if len(missing_k) != 0:
#             print(f"[INIT] WARN: not loaded from checkpoint -> {missing_k}")
#         if len(unexpected_k) != 0:
#             print(f"[INIT] WARN: unexpected in checkpoint -> {unexpected_k}")

#         self.hidden_size_desc = hidden_size_desc

#         if text_model is not None:
#             self.desc_fc = text_model
#         else:
#             self.desc_fc = nn.Sequential(
#                 nn.Linear(self.hidden_size_desc, self.self.hidden_size),
#                 nn.LeakyReLU(),
#                 nn.Linear(self.self.hidden_size, self.self.hidden_size),
#             )
#             print(f"[INIT] HIDDEN SIZE -> {self.self.hidden_size}")

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.self.hidden_size,
#             nhead=nhead,
#             dim_feedforward=hidden_ff,
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#         self.classifier = nn.Linear(self.self.hidden_size, 1)

#         self.activation = activation
#         self.loss_fct = loss_fct
#         self.weight = weight
#         self.cell_type_specific_loss_fn = cell_type_specific_loss_fn

#         self.post_init()

#     def _stat_print(self, name, x: torch.Tensor):
#         try:
#             xd = x.detach()
#             finite = torch.isfinite(xd)
#             numel = xd.numel()
#             num_fin = int(finite.sum().item())
#             if num_fin > 0:
#                 xf = xd[finite]
#                 _min = float(xf.min().item())
#                 _max = float(xf.max().item())
#                 _mean = float(xf.mean().item())
#                 print(f"[STAT] {name}: shape={tuple(xd.shape)} finite={num_fin}/{numel} min={_min:.6g} max={_max:.6g} mean={_mean:.6g} dtype={xd.dtype}")
#             else:
#                 print(f"[STAT] {name}: shape={tuple(xd.shape)} finite=0/{numel} (all NaN/Inf) dtype={xd.dtype}")
#         except Exception as e:
#             print(f"[STAT] {name}: <failed: {e}>")

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         position_ids=None,
#         labels_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#         meta_input_ids=None,
#         meta_attention_mask=None,
#         desc_vectors=None,
#         dataset_mean=None,
#         dataset_deviation=None,
#     ):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # ---- Encoder ----
#         bert_outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=True,
#         )

#         sequence_output = bert_outputs.last_hidden_state   # (B, seq_len, H)
#         B, seq_len, hidden_size = sequence_output.shape
#         print(f"[FWD] B={B} seq_len={seq_len} hidden_size={hidden_size}")

#         # ---- Desc vectors -> MLP ----
#         N = desc_vectors.shape[1]
#         print(f"[FWD] N={N} (num cell types)")
#         desc_vectors_2d = desc_vectors.reshape(B*N, desc_vectors.size(-1))
#         self._stat_print("desc_vectors_2d", desc_vectors_2d)

#         desc_fc_output = self.desc_fc(desc_vectors_2d).reshape(B, N, hidden_size)
#         self._stat_print("desc_fc_output", desc_fc_output)

#         # ---- Inject desc into CLS ----
#         seq_out_expanded = sequence_output.unsqueeze(1).expand(-1, N, -1, -1).contiguous()
#         seq_out_expanded[:, :, 0, :] = seq_out_expanded[:, :, 0, :].clone() + desc_fc_output

#         # ---- Transformer + head ----
#         seq_out_flat = seq_out_expanded.reshape(B*N, seq_len, hidden_size)
#         encoder_output = self.transformer_encoder(seq_out_flat)
#         logits = self.classifier(encoder_output)
#         logits = self.activation(logits)
#         self._stat_print("logits", logits)

#         # ---- Loss: only CLS ----
#         loss = None
#         cls_loss = None
#         other_loss = None
#         labels_reshaped = None
#         labels_mask_reshaped = None
#         mean_loss = None
#         diviation_loss = None

#         if labels is not None and labels_mask is not None:
#             # (B, seq_len, N) -> (B*N, seq_len, 1)
#             labels_reshaped = labels.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)
#             labels_mask_reshaped = labels_mask.permute(0, 2, 1).reshape(B*N, seq_len, 1).to(logits.device)

#             # --- Печати, что в labels/labels_mask значения только в CLS ---
#             # Сколько единиц в маске на CLS и на остальных позициях
#             cls_mask = labels_mask_reshaped[:, 0:1, :]     # (B*N,1,1)
#             other_mask = labels_mask_reshaped[:, 1:, :]    # (B*N,seq_len-1,1)
#             print(f"[CHK] cls_mask.sum()={int(cls_mask.sum().item())}, other_mask.sum()={int(other_mask.sum().item())}")

#             # Проверка: есть ли ненулевые значения label вне CLS и внутри маски
#             if (other_mask.sum() > 0):
#                 print("[WARN] other_mask имеет ненулевые позиции — метрика должна быть только по CLS!")

#             nonzero_outside_cls = (labels_reshaped[:, 1:, :] != 0) & other_mask.bool()
#             num_nonzero_outside = int(nonzero_outside_cls.sum().item())
#             if num_nonzero_outside > 0:
#                 print(f"[WARN] Найдены НЕ нулевые labels вне CLS и внутри маски: {num_nonzero_outside}. Прим принт нескольких значений...")
#                 # выведем до 5 примеров
#                 idx = nonzero_outside_cls.nonzero(as_tuple=False)
#                 for k in range(min(5, idx.shape[0])):
#                     bni, tni, _ = idx[k].tolist()
#                     print(f"   outside_cls example -> batch_idx={bni}, tok={tni+1}, label={float(labels_reshaped[bni, tni+1, 0].item())}")

#             # Доп.проверка: вне маски, но label не ноль?
#             outside_mask_positions = ~labels_mask_reshaped.bool()
#             suspicious = (labels_reshaped != 0) & outside_mask_positions
#             if suspicious.any():
#                 cnt = int(suspicious.sum().item())
#                 print(f"[WARN] Есть ненулевые labels вне маски: {cnt}. Они будут занулены для безопасности.")

#             # --- Санитизация лейблов (чтобы NaN не попали в лосс) ---
#             non_finite = ~torch.isfinite(labels_reshaped)
#             if non_finite.any():
#                 tot = int(non_finite.sum().item())
#                 print(f"[WARN] labels_reshaped имеет не-финитные значения: {tot}")
#                 outside_mask_nf = non_finite & (~labels_mask_reshaped.bool())
#                 if outside_mask_nf.any():
#                     print(f"[INFO] Обнуляю не-финитные labels вне маски: {int(outside_mask_nf.sum().item())}")
#                     labels_reshaped[outside_mask_nf] = 0.0
#                 inside_mask_nf = non_finite & labels_mask_reshaped.bool()
#                 if inside_mask_nf.any():
#                     print(f"[WARN] Не-финитные labels ВНУТРИ маски: {int(inside_mask_nf.sum().item())} -> принудительно 0.0")
#                     labels_reshaped[inside_mask_nf] = 0.0  # либо можно: labels_mask_reshaped[inside_mask_nf]=0

#             # На всякий случай: занулим всё вне CLS по маске (если где-то «портится» раньше)
#             if other_mask.sum() > 0:
#                 # занулим labels вне CLS там, где маска 1 (мы не хотим учиться по ним)
#                 labels_reshaped[:, 1:, :] = torch.where(
#                     other_mask.bool(),
#                     torch.zeros_like(labels_reshaped[:, 1:, :]),
#                     labels_reshaped[:, 1:, :]
#                 )

#             # Базовые статы
#             self._stat_print("labels_reshaped", labels_reshaped)
#             self._stat_print("labels_mask_reshaped.float()", labels_mask_reshaped.float())

#             # --- MSE только по маске (без NaN-пропагации) ---
#             diff = (logits - labels_reshaped) * labels_mask_reshaped
#             unreduced_loss = diff * diff
#             self._stat_print("unreduced_loss", unreduced_loss)

#             # --- Разделение на CLS и остальные ---
#             cls_loss = None
#             if cls_mask.sum() > 0:
#                 if self.cell_type_specific_loss_fn is not None:
#                     if dataset_mean is not None and dataset_mean.device != logits.device:
#                         dataset_mean = dataset_mean.to(logits.device)
#                     if dataset_deviation is not None and dataset_deviation.device != logits.device:
#                         dataset_deviation = dataset_deviation.to(logits.device)

#                     cls_loss, mean_loss, diviation_loss = self.cell_type_specific_loss_fn(
#                         cls_targets=labels_reshaped[:, 0:1, :].reshape(B, N),
#                         cls_preds=logits[:, 0:1, :].reshape(B, N),
#                         cls_mask=cls_mask.reshape(B, N),
#                         dataset_mean=dataset_mean,
#                         dataset_deviation=dataset_deviation
#                     )
#                     print(f"[LOSS] cls_loss(ct-specific)={float(cls_loss.item())}")
#                 else:
#                     cls_loss = (unreduced_loss[:, 0:1, :].sum() / cls_mask.sum())
#                     print(f"[LOSS] cls_loss={float(cls_loss.item())}")

#             if other_mask.sum() > 0:
#                 other_loss = (unreduced_loss[:, 1:, :].sum() / other_mask.sum())
#                 print(f"[LOSS] other_loss={float(other_loss.item())}")
#             else:
#                 other_loss = None
#                 print("[LOSS] other_loss is None (как и ожидается, т.к. учимся только по CLS)")

#             if cls_loss is not None and other_loss is not None:
#                 loss = cls_loss + self.weight * other_loss
#             elif cls_loss is not None:
#                 loss = cls_loss
#             elif other_loss is not None:
#                 loss = self.weight * other_loss

#             if loss is not None and not torch.isfinite(loss):
#                 print("[ERR] loss стал не-финитным! Проверь логи выше (labels/logits/маски).")

#         if not return_dict:
#             return (loss, logits)

#         return ExpressionModelOutput(
#             loss=loss,
#             logits=logits,
#             hidden_states=bert_outputs.hidden_states,
#             attentions=bert_outputs.attentions,
#             labels_reshaped=labels_reshaped,
#             labels_mask_reshaped=labels_mask_reshaped,
#             cls_loss=cls_loss,
#             mean_loss=mean_loss,
#             diviation_loss=diviation_loss,
#             other_loss=other_loss
#         )

# class cell_type_specific_loss_fn(nn.Module):
#     def __init__(self,
#                  weight_mean,
#                  loss_fct_mean=nn.MSELoss(reduction="none"),
#                  loss_fct_diviation=nn.MSELoss(reduction="none"),
#                  normalize_by_mean=True,
#                  ):
#         super().__init__()
#         assert 0 <= weight_mean <= 1, "weight_mean must be between 0 and 1"
#         self.loss_fct_mean = loss_fct_mean
#         self.loss_fct_diviation = loss_fct_diviation
#         self.weight_mean = weight_mean
#         self.weight_diviation = 1 - weight_mean
#         self.normalize_by_mean = normalize_by_mean

#     def forward(self, cls_targets, cls_preds, cls_mask,
#                 dataset_mean, dataset_deviation):
#         # Проверка на пустые маски по батчу
#         if cls_mask.sum(dim=1).eq(0).any():
#             raise ValueError("cls_mask.sum(dim=1) is 0 for some samples.")

#         # Среднее по клеточным типам (только там, где mask=1)
#         cls_targets_mean = (cls_targets * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)
#         cls_preds_mean   = (cls_preds   * cls_mask).sum(dim=1) / cls_mask.sum(dim=1)

#         # Контроль соответствия dataset_mean
#         if (dataset_mean is not None) and (not torch.allclose(dataset_mean, cls_targets_mean.squeeze(), atol=1e-6, rtol=1e-5)):
#             diff = torch.max(torch.abs(dataset_mean - cls_targets_mean.squeeze())).item()
#             raise ValueError(f"dataset_mean != cls_targets_mean (max|diff|={diff:.6g})")

#         if self.normalize_by_mean:
#             cls_targets_div = (cls_targets - cls_targets_mean.unsqueeze(1)) / cls_targets_mean.unsqueeze(1)
#             cls_preds_div   = (cls_preds   - cls_preds_mean.unsqueeze(1))   / cls_preds_mean.unsqueeze(1)
#             debug_targets_div = cls_targets_div
#         else:
#             cls_targets_div = (cls_targets - cls_targets_mean.unsqueeze(1))
#             cls_preds_div   = (cls_preds   - cls_preds_mean .unsqueeze(1))
#             debug_targets_div = (cls_targets - cls_targets_mean.unsqueeze(1)) / cls_targets_mean.unsqueeze(1)

#         if (dataset_deviation is not None) and (not torch.allclose(dataset_deviation[cls_mask.bool()], debug_targets_div[cls_mask.bool()], atol=1e-6, rtol=1e-5)):
#             maxdiff = torch.max(torch.abs(dataset_deviation[cls_mask.bool()] - debug_targets_div[cls_mask.bool()])).item()
#             raise ValueError(f"dataset_deviation != cls_targets_deviation (max|diff|={maxdiff:.6g})")

#         # Лоссы (mask как веса)
#         cls_loss_mean = (self.loss_fct_mean(cls_preds_mean, cls_targets_mean) * 1.0).mean()  # уже свернули по cell types
#         # Для дивиаций даём mask как веса элементам
#         elem_loss_div = self.loss_fct_diviation(cls_preds_div, cls_targets_div)  # (B, N)
#         cls_loss_diviation = (elem_loss_div * cls_mask).sum() / cls_mask.sum()

#         full_loss = self.weight_mean * cls_loss_mean + self.weight_diviation * cls_loss_diviation
#         return full_loss, cls_loss_mean, cls_loss_diviation


