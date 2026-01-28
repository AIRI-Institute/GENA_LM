import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from src.gena_lm.modeling_bert import BertPreTrainedModel, BertModel
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from transformers import AutoModel, BertConfig
from transformers.utils import cached_file, logging as hf_logging
from typing import Optional, Dict, List, Tuple, Any

# Set logging verbosity
hf_logging.set_verbosity_info()

logger = hf_logging.get_logger(__name__)


@dataclass
class ExpressionModelOutput(TokenClassifierOutput):
    """Extended output class for expression counting model."""
    labels_reshaped: Optional[torch.FloatTensor] = None
    labels_mask_reshaped: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    other_loss: Optional[torch.FloatTensor] = None


class DescriptionEncoder(nn.Module):
    """Handles description encoding with optional freezing."""
    
    def __init__(
        self,
        desc_model_name: str = "intfloat/multilingual-e5-large-instruct",
        unfreeze_last_blocks: int = 4,
        target_hidden_size: Optional[int] = None,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        
        self.desc_model_name = desc_model_name
        self.desc_model = AutoModel.from_pretrained(
            desc_model_name,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype
        )
        
        # Freeze all parameters initially
        for param in self.desc_model.parameters():
            param.requires_grad = False
        
        # Unfreeze specified blocks
        self._unfreeze_blocks(unfreeze_last_blocks)
        
        # Projection if dimensions don't match
        self.desc_hidden_size = self.desc_model.config.hidden_size
        self.target_hidden_size = target_hidden_size or self.desc_hidden_size
        
        if self.desc_hidden_size != self.target_hidden_size:
            self.desc_proj = nn.Linear(self.desc_hidden_size, self.target_hidden_size)
        else:
            self.desc_proj = nn.Identity()
        
        self._log_trainable_params()
    
    def _unfreeze_blocks(self, unfreeze_last_blocks: int):
        """Unfreeze the last N transformer blocks."""
        backbone = getattr(self.desc_model, "model", self.desc_model)
        layers = getattr(backbone, "layers", getattr(backbone, "encoder", None))
        
        if layers is None:
            raise RuntimeError(
                f"Could not find transformer layers in {self.desc_model_name}. "
                f"Expected .model.layers or .encoder.layers"
            )
        
        # Unfreeze last N blocks
        k = min(unfreeze_last_blocks, len(layers))
        
        for block in layers[-k:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Unfreeze normalization layer if it exists
        if hasattr(backbone, "norm") and backbone.norm is not None:
            for param in backbone.norm.parameters():
                param.requires_grad = True
        
        # Set training modes
        for block in layers[:-k]:
            block.eval()
        for block in layers[-k:]:
            block.train()
        
        if hasattr(backbone, "norm") and backbone.norm is not None:
            backbone.norm.train()
    
    def _log_trainable_params(self):
        """Log trainable parameters information."""
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        
        # Find unfrozen blocks
        backbone = getattr(self.desc_model, "model", self.desc_model)
        layers = getattr(backbone, "layers", getattr(backbone, "encoder", None))
        
        unfrozen_blocks = []
        if layers is not None:
            for i, block in enumerate(layers):
                if any(param.requires_grad for param in block.parameters()):
                    unfrozen_blocks.append(i)
        
        logger.info(f"[DescriptionEncoder] Unfrozen transformer blocks: {unfrozen_blocks}")
        
        # Count parameters
        total_params = sum(param.numel() for param in self.desc_model.parameters())
        trainable_params = sum(param.numel() for param in self.desc_model.parameters() 
                              if param.requires_grad)
        
        logger.info(f"[DescriptionEncoder] Trainable params: {trainable_params:,} / {total_params:,} "
                   f"({trainable_params/total_params*100:.1f}%)")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        desc_vectors: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for description encoder."""
        if desc_vectors is not None:
            # Use pre-computed embeddings (frozen embeddings mode)
            return desc_vectors
        
        # Use transformer model
        outputs = self.desc_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use last token representation (often used for sentence embeddings)
        last_hidden_state = outputs.last_hidden_state
        pooled = last_hidden_state[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Project if necessary
        return self.desc_proj(pooled)


class ExpressionCounts(nn.Module):
    """
    Expression counting model with DNA sequence encoder and description encoder.
    
    Expected shapes:
      - input_ids:      (B*N, L)
      - attention_mask: (B*N, L)
      - desc_vectors:   (B, N, D) or None
      - dataset_flag:   (B, N) where 1 -> INPUT duplicates, 0 -> DESC duplicates
      - labels:         (B*N, L, 1)
      - labels_mask:    (B*N, L, 1)
    """
    
    def __init__(
        self,
        config: Optional[BertConfig] = None,
        hf_model_name_decoder: str = "AIRI-Institute/gena-lm-bert-large-t2t",
        loss_fct: nn.Module = nn.MSELoss(reduction="none"),
        activation: nn.Module = nn.Identity(),
        nhead: int = 8,
        weight: float = 1.0,
        hidden_size_desc: int = 768,
        bert_checkpoint: Optional[str] = None,
        use_hf_dna_model: bool = False,
        hf_dna_model_name: str = "AIRI-Institute/gena-lm-bert-large-t2t",
        use_frozen_embeds: bool = True,
        desc_model_name: str = "intfloat/multilingual-e5-large-instruct",
        unfreeze_last_blocks: int = 4,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        
        self.use_frozen_embeds = use_frozen_embeds
        self.weight = weight
        
        self.config = config
        # ===== 1. DNA Sequence Encoder =====
        self.dna_encoder = self._init_dna_encoder(
            config=config,
            use_hf_dna_model=use_hf_dna_model,
            hf_dna_model_name=hf_dna_model_name,
            bert_checkpoint=bert_checkpoint,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype
        )
        
        # ===== 2. Description Encoder =====
        if not use_frozen_embeds:
            self.desc_encoder = DescriptionEncoder(
                desc_model_name=desc_model_name,
                unfreeze_last_blocks=unfreeze_last_blocks,
                target_hidden_size=self.dna_encoder.config.hidden_size,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype
            )
            self.desc_hidden_size = self.desc_encoder.desc_hidden_size
        else:
            # Simple MLP for frozen embeddings
            self.desc_hidden_size = hidden_size_desc
            self.dna_hidden_size = self.dna_encoder.config.hidden_size
            
            self.desc_fc = nn.Sequential(
                nn.Linear(self.desc_hidden_size, self.dna_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.dna_hidden_size, self.dna_hidden_size),
            )
            
            # Projection layer to ensure matching dimensions
            self.desc_proj = nn.Linear(self.dna_hidden_size, self.dna_hidden_size)
        
        # ===== 3. Decoder =====
        self.decoder = self._init_decoder(
            hf_model_name_decoder=hf_model_name_decoder,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype
        )
        
        # ===== 4. Classifier =====
        self.classifier = nn.Linear(
            self.decoder.config.hidden_size, 
            1,
            device=next(self.dna_encoder.parameters()).device,
            dtype=torch_dtype
        )
        
        # Freeze decoder embeddings if they exist
        if hasattr(self.decoder, "embeddings"):
            if hasattr(self.decoder.embeddings, "word_embeddings"):
                self.decoder.embeddings.word_embeddings.weight.requires_grad_(False)
            elif hasattr(self.decoder.embeddings, "tok_embeddings"):
                self.decoder.embeddings.tok_embeddings.weight.requires_grad_(False)
        
        # ===== 5. Loss and Activation =====
        self.loss_fct = loss_fct
        self.activation = activation
    
    def _init_dna_encoder(
        self,
        config: Optional[BertConfig],
        use_hf_dna_model: bool,
        hf_dna_model_name: str,
        bert_checkpoint: Optional[str],
        attn_implementation: str,
        torch_dtype: torch.dtype
    ) -> nn.Module:
        """Initialize the DNA sequence encoder."""
        try:
            # Try to import ModernBertModel
            from transformers import ModernBertModel
            if "modernbert" in hf_dna_model_name.lower():
                logger.info(f"Using ModernBERT from {hf_dna_model_name}")
                model, info = ModernBertModel.from_pretrained(
                    hf_dna_model_name,
                    trust_remote_code=True,
                    attn_implementation=attn_implementation,
                    output_loading_info=True,
                    torch_dtype=torch_dtype
                )
                config = model.config
                self._log_loading_info("DNA Encoder", info)
                return model
        except ImportError:
            pass
        
        # Fall back to standard BERT
        if use_hf_dna_model:
            hf_config = BertConfig.from_pretrained(hf_dna_model_name)
            model = BertModel(hf_config, add_pooling_layer=False)
            
            # Load weights
            weights_path = cached_file(hf_dna_model_name, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu")
            
            # Update state dict keys
            updated_state_dict = {
                k.replace("bert.", ""): v for k, v in state_dict.items()
                if k.startswith("bert.")
            }
            
            missing, unexpected = model.load_state_dict(updated_state_dict, strict=False)
            self._log_missing_unexpected("DNA Encoder", missing, unexpected)
            
            config = hf_config
        else:
            #if config is None:
            #    raise ValueError("Config must be provided when not using HF model")
            config = BertConfig.from_pretrained(hf_dna_model_name)
            model = BertModel(config, add_pooling_layer=False)
            
            if bert_checkpoint:
                checkpoint = torch.load(bert_checkpoint, map_location="cpu")
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                
                # Update state dict keys
                updated_state_dict = {
                    k.replace("bert.", ""): v for k, v in state_dict.items()
                }
                
                missing, unexpected = model.load_state_dict(updated_state_dict, strict=False)
                self._log_missing_unexpected("DNA Encoder", missing, unexpected)
        
        return model
    
    def _init_decoder(
        self,
        hf_model_name_decoder: str,
        attn_implementation: str,
        torch_dtype: torch.dtype
    ) -> nn.Module:
        """Initialize the decoder."""
        try:
            from transformers import ModernBertModel
            logger.info(f"Using ModernBERT for decoder from {hf_model_name_decoder}")
            model, info = ModernBertModel.from_pretrained(
                hf_model_name_decoder,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                output_loading_info=True,
                torch_dtype=torch_dtype
            )
            self._log_loading_info("Decoder", info)
            return model
        except ImportError:
            # Fall back to standard BERT
            logger.info(f"Using standard BERT for decoder from {hf_model_name_decoder}")
            model = AutoModel.from_pretrained(
                hf_model_name_decoder,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype
            )
            return model
    
    def _log_loading_info(self, component: str, info: Dict[str, Any]):
        """Log model loading information."""
        logger.info(f"[{component}] Missing keys: {len(info.get('missing_keys', []))}")
        if info.get('missing_keys'):
            logger.info(f"  First 10: {info['missing_keys'][:10]}")
        
        logger.info(f"[{component}] Unexpected keys: {len(info.get('unexpected_keys', []))}")
        if info.get('unexpected_keys'):
            logger.info(f"  First 10: {info['unexpected_keys'][:10]}")
        
        logger.info(f"[{component}] Mismatched keys: {len(info.get('mismatched_keys', []))}")
        if info.get('mismatched_keys'):
            logger.info(f"  First 5: {info['mismatched_keys'][:5]}")
    
    def _log_missing_unexpected(self, component: str, missing: list, unexpected: list):
        """Log missing and unexpected keys."""
        if missing:
            logger.warning(f"[{component}] Missing keys: {len(missing)}")
            logger.warning(f"  First 10: {missing[:10]}")
        if unexpected:
            logger.warning(f"[{component}] Unexpected keys: {len(unexpected)}")
            logger.warning(f"  First 10: {unexpected[:10]}")
    
    def _compute_unique_indices(
        self, 
        dataset_flag: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute indices for unique DNA sequences and descriptions.
        
        Args:
            dataset_flag: (B, N) tensor where 1 -> INPUT duplicates, 0 -> DESC duplicates
        
        Returns:
            Tuple of (unique_dna_idx, unique_desc_idx, dna_mapping, desc_mapping)
        """
        B, N = dataset_flag.shape
        BxN = B * N
        
        # Create index grid
        idx_all = torch.arange(BxN, device=device)
        idx_grid = idx_all.view(B, N)
        
        # ===== DNA indices (unique sequences) =====
        # When dataset_flag == 1 (INPUT duplicates), all N sequences in that batch are the same
        # We need to deduplicate based on the flag
        flag = dataset_flag.bool()
        
        # Get indices for unique DNA sequences
        unique_dna_indices = []
        dna_mapping = torch.zeros(BxN, dtype=torch.long, device=device)
        
        current_unique_idx = 0
        for b in range(B):
            if flag[b, 0]:  # INPUT duplicates - all N are the same
                # First one is unique
                unique_dna_indices.append(idx_grid[b, 0])
                # Map all N to the same unique index
                dna_mapping[idx_grid[b]] = current_unique_idx
                current_unique_idx += 1
            else:  # DESC duplicates - each is unique (different descriptions)
                for n in range(N):
                    unique_dna_indices.append(idx_grid[b, n])
                    dna_mapping[idx_grid[b, n]] = current_unique_idx
                    current_unique_idx += 1
        
        unique_dna_idx = torch.tensor(unique_dna_indices, device=device)
        
        # ===== Description indices (unique descriptions) =====
        # When dataset_flag == 0 (DESC duplicates), all N descriptions in that batch are the same
        unique_desc_indices = []
        desc_mapping = torch.zeros(BxN, dtype=torch.long, device=device)
        
        current_unique_idx = 0
        for b in range(B):
            if not flag[b, 0]:  # DESC duplicates - all N descriptions are the same
                # First one is unique
                unique_desc_indices.append(idx_grid[b, 0])
                # Map all N to the same unique index
                desc_mapping[idx_grid[b]] = current_unique_idx
                current_unique_idx += 1
            else:  # INPUT duplicates - each description is unique
                for n in range(N):
                    unique_desc_indices.append(idx_grid[b, n])
                    desc_mapping[idx_grid[b, n]] = current_unique_idx
                    current_unique_idx += 1
        
        unique_desc_idx = torch.tensor(unique_desc_indices, device=device)
        
        return unique_dna_idx, unique_desc_idx, dna_mapping, desc_mapping
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        desc_input_ids: Optional[torch.Tensor] = None,
        desc_attention_mask: Optional[torch.Tensor] = None,
        desc_vectors: Optional[torch.Tensor] = None,
        dataset_flag: Optional[torch.Tensor] = None,
    ) -> ExpressionModelOutput:
        # Validate inputs
        if dataset_flag is None:
            raise ValueError("dataset_flag must be provided with shape (B, N)")
        
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        B, N = dataset_flag.shape
        device = input_ids.device
        
        # Reshape inputs if needed
        if input_ids.dim() == 3:  # (B, N, L) -> (B*N, L)
            input_ids = input_ids.reshape(B * N, -1)
        
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask.reshape(B * N, -1)
        
        if desc_input_ids is not None and desc_input_ids.dim() == 3:
            desc_input_ids = desc_input_ids.reshape(B * N, -1)
        
        if desc_attention_mask is not None and desc_attention_mask.dim() == 3:
            desc_attention_mask = desc_attention_mask.reshape(B * N, -1)
        
        if labels is not None and labels.dim() == 4:
            labels = labels.reshape(B * N, labels.shape[-2], labels.shape[-1])
        
        if labels_mask is not None and labels_mask.dim() == 4:
            labels_mask = labels_mask.reshape(B * N, labels_mask.shape[-2], labels_mask.shape[-1])
        
        # ===== 1. Compute unique indices =====
        unique_dna_idx, unique_desc_idx, dna_mapping, desc_mapping = self._compute_unique_indices(
            dataset_flag, device
        )
        
        # ===== 2. Encode DNA sequences (with deduplication) =====
        dna_embeddings = self.dna_encoder(
            input_ids=input_ids[unique_dna_idx],
            attention_mask=attention_mask[unique_dna_idx] if attention_mask is not None else None,
            return_dict=True,
        ).last_hidden_state
        
        # Map back to original order
        dna_embeddings = dna_embeddings[dna_mapping]  # (B*N, L, H)
        
        # ===== 3. Encode descriptions (with deduplication) =====
        if not self.use_frozen_embeds:
            # Use transformer-based encoder
            desc_embeddings = self.desc_encoder(
                input_ids=desc_input_ids[unique_desc_idx] if desc_input_ids is not None else None,
                attention_mask=desc_attention_mask[unique_desc_idx] if desc_attention_mask is not None else None,
                desc_vectors=None
            )
            desc_embeddings = desc_embeddings[desc_mapping]
        else:
            # Use frozen embeddings with MLP
            if desc_vectors is None:
                raise ValueError("desc_vectors must be provided when use_frozen_embeds=True")
            
            B_orig = desc_vectors.shape[0]
            desc_vectors_2d = desc_vectors.reshape(B_orig * N, -1)
            desc_embeddings = self.desc_fc(desc_vectors_2d)
            desc_embeddings = desc_embeddings.reshape(B_orig, N, -1)
            
            # Project and map
            desc_embeddings = self.desc_proj(desc_embeddings)
            desc_embeddings = desc_embeddings.reshape(B * N, -1)
            desc_embeddings = desc_embeddings[desc_mapping]
        
        # Ensure same dtype
        desc_embeddings = desc_embeddings.to(dna_embeddings.dtype)
        
        # ===== 4. Combine DNA and description embeddings =====
        # Add description information to each token position
        combined_embeddings = dna_embeddings + desc_embeddings.unsqueeze(1)
        
        # Optionally mask using attention_mask
        if attention_mask is not None:
            combined_embeddings = combined_embeddings * attention_mask.unsqueeze(-1).to(combined_embeddings.dtype)
        
        # ===== 5. Decode combined embeddings =====
        decoder_outputs = self.decoder(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        decoder_hidden = decoder_outputs.last_hidden_state
        
        # ===== 6. Classification =====
        logits = self.activation(self.classifier(decoder_hidden))  # (B*N, L, 1)
        
        # ===== 7. Compute loss if labels provided =====
        loss = None
        cls_loss = None
        other_loss = None
        
        if labels is not None:
            labels = labels.to(logits.device)
            
            # Compute per-element loss
            unreduced_loss = self.loss_fct(logits, labels)  # (B*N, L, 1)
            
            if labels_mask is not None:
                labels_mask = labels_mask.to(logits.device)
                
                # Separate CLS token (position 0) and other tokens
                cls_mask = labels_mask[:, 0:1, :]  # (B*N, 1, 1)
                other_mask = labels_mask[:, 1:, :]  # (B*N, L-1, 1)
                
                # Compute CLS loss
                if cls_mask.sum().item() > 0:
                    cls_loss = (unreduced_loss[:, 0:1, :] * cls_mask).sum() / (cls_mask.sum() + 1e-8)
                
                # Compute other tokens loss
                if other_mask.sum().item() > 0:
                    other_loss = (unreduced_loss[:, 1:, :] * other_mask).sum() / (other_mask.sum() + 1e-8)
                
                # Combine losses
                if cls_loss is not None and other_loss is not None:
                    loss = cls_loss + self.weight * other_loss
                elif cls_loss is not None:
                    loss = cls_loss
                elif other_loss is not None:
                    loss = self.weight * other_loss
        
        # ===== 8. Return outputs =====
        return_dict = return_dict if return_dict is not None else True
        
        if not return_dict:
            return (loss, logits)
        
        return ExpressionModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=(decoder_hidden,),
            attentions=decoder_outputs.attentions,
            labels_reshaped=labels,
            labels_mask_reshaped=labels_mask,
            cls_loss=cls_loss,
            other_loss=other_loss
        )