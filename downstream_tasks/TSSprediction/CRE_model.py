import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional
from dataclasses import dataclass
from transformers import AutoModel, BertConfig, ModernBertModel, ModernBertConfig
from transformers.utils import cached_file
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_info()
import sys
from pathlib import Path

@dataclass
class CreModelOutput(TokenClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    predicts: Optional[torch.FloatTensor] = None

class CreModel(nn.Module):

    def __init__(
        self,
        loss_fct=nn.BCEWithLogitsLoss(reduction='none'),
        activation = nn.Identity(),
        bert_cpt = None,
        hf: bool = False,
        hf_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
        num_taxons: int|None = None
    ):
        super().__init__()
        

        updated_state_dict = None

        # 1) DNA model (GENA) 
        if hf:
            if "modernbert" in hf_model_name.lower():
                print(f"Using ModernBERT from {hf_model_name}")
                self.bert, info  = ModernBertModel.from_pretrained(
                Path(hf_model_name),
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                attention_dropout=0.1,
                embedding_dropout=0.1,
                mlp_dropout=0.1,
                output_loading_info=True,
                reference_compile=False
            )
                config = self.bert.config
                print("missing:", len(info["missing_keys"]), info["missing_keys"][:10])
                print("unexpected:", len(info["unexpected_keys"]), info["unexpected_keys"][:10])
                print("mismatched:", info.get("mismatched_keys", [])[:5])
                print(
                    "dropouts:",
                    {
                        "attention_dropout": config.attention_dropout,
                        "embedding_dropout": config.embedding_dropout,
                        "mlp_dropout": config.mlp_dropout,
                    }
                )
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
            print('hf was set to False, loading local model')
            config = ModernBertConfig.from_pretrained(hf_model_name)
            config.attn_implementation = "flash_attention_2"
            self.bert, info = ModernBertModel.from_pretrained(
                hf_model_name,
                config=config,
                output_loading_info=True
            )
            print("missing:", len(info["missing_keys"]), info["missing_keys"][:10])
            print("unexpected:", len(info["unexpected_keys"]), info["unexpected_keys"][:10])
            print("mismatched:", info.get("mismatched_keys", [])[:5])
            if bert_cpt is not None:
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

        # 6) Loss
        self.activation = activation
        self.loss_fct = loss_fct

        dtype = next(self.bert.parameters()).dtype
        device = next(self.bert.parameters()).device


        # 5) Classifier
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1, device=device, dtype=dtype)
        
        if num_taxons is not None:
            self.taxon_embeddings = nn.Embedding(num_embeddings=num_taxons, embedding_dim=self.bert.config.hidden_size)

    def forward(
        self,
        input_ids=None,              # (B, L)
        attention_mask=None,         # (B, L) or None
        labels_mask=None,            # (B, L, 1)
        labels=None,                 # (B, L, 1)
        return_dict=None,
        taxon=None                  # (B, 1)
    ):
        
        #print(f'''
        #      input_ids shape: {input_ids.shape}
        #      attention_mask shape: {attention_mask.shape}
        #      labels_mask shape: {labels_mask.shape}
        #      labels shape: {labels.shape}
        #      taxon shape: {taxon.shape}
        #      ''')
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        bert_outputs = self.bert(
                input_ids=input_ids,                         
                attention_mask=attention_mask,
                return_dict=True,
            )
        sequence_output = bert_outputs.last_hidden_state # (B, L, H)
        
        taxon_embedding = self.taxon_embeddings(taxon)  # (B, H)
        
        taxon_sequence_output = sequence_output + taxon_embedding.unsqueeze(1)
        
        logits = self.activation(self.classifier(taxon_sequence_output))  # (B, L, 1)
        # 5) Loss
        loss = None
        labels_reshaped = labels_mask_reshaped = cls_loss = None
        if labels is not None:
            labels_reshaped = labels.to(logits.device)
            labels_mask_reshaped = labels_mask.to(logits.device) if labels_mask is not None else None
            unreduced_loss = self.loss_fct(logits, labels_reshaped)  # (B, L, 1)
            if labels_mask_reshaped is not None and labels_mask_reshaped.sum().item() > 0:
                cls_mask = labels_mask_reshaped[:, 0:1, :]           # (B, 1, 1)

                if cls_mask.sum().item() > 0:
                    cls_loss = (unreduced_loss[:, 0:1, :] * cls_mask).sum() / (cls_mask.sum() + 1e-8)
                if cls_loss is not None:
                    loss = cls_loss

        if not return_dict:
            return (loss, logits)
        hidden_states_out = (sequence_output,)
        return CreModelOutput(
            loss=loss,
            logits=logits,
            predicts = self.classifier(taxon_sequence_output)
        )

    