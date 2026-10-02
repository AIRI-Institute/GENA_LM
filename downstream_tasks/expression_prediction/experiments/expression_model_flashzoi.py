"""Expression model with a pretrained Flashzoi/Borzoi DNA encoder.

Data flow::

    dna_codes (B, S)
      -> FlashzoiEncoder                  (B, out_bins, 1536)
      -> Linear(1536 -> H) + LayerNorm    (B, out_bins, H)
      -> prepend CLS, append SEP          (B, out_bins + 2, H)
      -> [optional ModernGENA tower]
      -> + broadcast description embedding
      -> ModernBERT decoder
      -> Linear(H, 1)                     (B, out_bins + 2, 1)

Relative to :class:`~expression_model_cnn.ExpressionCountsCNN` exactly two things
change: the DNA encoder is Flashzoi instead of the AlphaGenome CNN, and the
decoder's sequence length comes from ``out_bins`` instead of
``S // cnn_total_stride`` (Flashzoi crops its output, so the two are not equal).
Everything below the DNA input -- the frozen-except-last-k description encoder,
the additive LayerNorm fusion, the ModernBERT decoder, the linear head, and
position 0 carrying the gene-level target -- is unchanged, so the loss and metric
code is untouched.

``use_tower`` decides whether ModernGENA still sits between the encoder and the
fusion. The default is ``False``: Flashzoi already contains 8 transformer layers
over a 524 kb receptive field, and stacking a BPE-pretrained tower on top of
32 bp bins is both a mismatch and ~400M extra parameters. Set it to ``True`` to
measure that claim rather than assume it.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, ModernBertModel
from transformers.utils import logging as hf_logging

from downstream_tasks.expression_prediction.flashzoi_encoder import FlashzoiEncoder
from downstream_tasks.expression_prediction.expression_model_final import (
    ExpActivation,  # noqa: F401  (re-exported for configs that reference it)
    ExpressionModelOutput,
    cls_deviation_from_mean_loss,
    cls_multinomial_loss,
)

hf_logging.set_verbosity_warning()


def _is_main_process() -> bool:
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


class ExpressionCountsFlashzoi(nn.Module):
    """Expected shapes:

    - dna_codes:           (B, S)                  uint8/long nucleotide codes
    - attention_mask:      (B, out_bins + 2)
    - desc_input_ids:      (B, N, D)
    - desc_attention_mask: (B, N, D)
    - desc_index:          (B, N)                  batch-local description ids
    - labels:              (B, N, out_bins + 2, 1) [:, :, 0] = gene-level TPM/CPM
    - labels_mask:         (B, N, out_bins + 2, 1)

    The DNA window is shared by every track of a gene, so the encoder runs once
    per gene and its output is broadcast over the ``N`` rows.
    """

    def __init__(
        self,
        hf_model_name_decoder,
        flashzoi_model_name: str = "johahi/flashzoi-replicate-0",
        loss_fct=nn.MSELoss(reduction="none"),
        activation=nn.Identity(),
        weight: float = 1.0,
        desc_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        dropout_prob: float = 0.1,
        desc_unfrozen_blocks: int = 4,
        use_deviation_loss: bool = False,
        use_multinomial_loss: bool = False,
        weight_deviation_loss: float = 1.0,
        weight_multinomial_loss: float = 1.0,
        # --- Flashzoi encoder ---
        out_bins: int = 1022,
        pool: int = 1,
        tap: str = "unet",
        # False = same architecture, no Borzoi weights. The control that tells
        # apart "the architecture helps" from "the pretraining helps".
        flashzoi_pretrained: bool = True,
        freeze_conv_tower: bool = True,
        freeze_transformer_blocks: int = 0,
        freeze_batchnorm: bool = True,
        gradient_checkpointing: bool = True,
        flashzoi_attn_dropout: Optional[float] = None,
        flashzoi_dropout_rate: Optional[float] = None,
        # --- optional ModernGENA tower between encoder and fusion ---
        use_tower: bool = False,
        hf_model_name: Optional[str] = None,
        pretrained_tower: bool = True,
    ):
        super().__init__()

        # 1) DNA encoder
        self.dna_encoder = FlashzoiEncoder(
            flashzoi_model_name,
            out_bins=out_bins,
            pool=pool,
            tap=tap,
            pretrained=flashzoi_pretrained,
            freeze_conv_tower=freeze_conv_tower,
            freeze_transformer_blocks=freeze_transformer_blocks,
            freeze_batchnorm=freeze_batchnorm,
            gradient_checkpointing=gradient_checkpointing,
            attn_dropout=flashzoi_attn_dropout,
            dropout_rate=flashzoi_dropout_rate,
        )
        self.out_bins = self.dna_encoder.out_bins
        self.seq_len = self.out_bins + 2

        # 2) Decoder (always present -- it is where the cell-type conditioning lands)
        if _is_main_process():
            print(f"Using ModernBERT decoder from {hf_model_name_decoder}")
        self.decoder, info = ModernBertModel.from_pretrained(
            hf_model_name_decoder,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            attention_dropout=dropout_prob,
            embedding_dropout=dropout_prob,
            mlp_dropout=dropout_prob,
            output_loading_info=True,
        )
        if _is_main_process():
            print("missing:", len(info["missing_keys"]), info["missing_keys"][:10])
            print("unexpected:", len(info["unexpected_keys"]), info["unexpected_keys"][:10])
        self._freeze_token_embeddings(self.decoder, "decoder")

        max_pos = getattr(self.decoder.config, "max_position_embeddings", None)
        if max_pos is not None and self.seq_len > max_pos:
            raise ValueError(
                f"decoder {hf_model_name_decoder} accepts at most {max_pos} positions, but "
                f"out_bins={self.out_bins} needs {self.seq_len} (CLS + bins + SEP). "
                f"Lower out_bins to {max_pos - 2}, raise `pool`, or use a decoder with a "
                "longer position budget."
            )

        # 3) Optional ModernGENA tower
        self.use_tower = bool(use_tower)
        if self.use_tower:
            if hf_model_name is None:
                raise ValueError("use_tower=True requires hf_model_name")
            if _is_main_process():
                print(f"Using ModernGENA tower from {hf_model_name}")
            if pretrained_tower:
                self.bert, info2 = ModernBertModel.from_pretrained(
                    hf_model_name,
                    trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                    attention_dropout=dropout_prob,
                    embedding_dropout=dropout_prob,
                    mlp_dropout=dropout_prob,
                    output_loading_info=True,
                )
                if _is_main_process():
                    print("missing:", len(info2["missing_keys"]), info2["missing_keys"][:10])
                    print("unexpected:", len(info2["unexpected_keys"]), info2["unexpected_keys"][:10])
            else:
                tower_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
                tower_config.attention_dropout = dropout_prob
                tower_config.embedding_dropout = dropout_prob
                tower_config.mlp_dropout = dropout_prob
                self.bert = ModernBertModel(tower_config)
                if _is_main_process():
                    print("Tower initialised from scratch (pretrained_tower=False)")
            self._freeze_token_embeddings(self.bert, "tower")
            self.gen_hidden_size = self.bert.config.hidden_size
            self.config = self.bert.config
        else:
            self.bert = None
            self.gen_hidden_size = self.decoder.config.hidden_size
            self.config = self.decoder.config

        if self.gen_hidden_size != self.decoder.config.hidden_size:
            raise ValueError(
                f"tower hidden size {self.gen_hidden_size} != decoder hidden size "
                f"{self.decoder.config.hidden_size}; the fusion feeds the tower output "
                "straight into the decoder as inputs_embeds."
            )

        # 4) Description model
        self.desc_model_name = desc_model_name
        self.desc_model = AutoModel.from_pretrained(
            self.desc_model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            attention_dropout=dropout_prob,
        )
        for p in self.desc_model.parameters():
            p.requires_grad = False

        backbone = getattr(self.desc_model, "model", None) or self.desc_model
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise RuntimeError(
                "Could not find layers in desc_model (expected .model.layers). "
                "Check model architecture."
            )
        k = max(0, min(desc_unfrozen_blocks, len(layers)))
        if k > 0:
            for block in layers[-k:]:
                for p in block.parameters():
                    p.requires_grad = True
            if getattr(backbone, "norm", None) is not None:
                for p in backbone.norm.parameters():
                    p.requires_grad = True
        self.desc_unfrozen_blocks = k
        self.desc_hidden_size = self.desc_model.config.hidden_size

        if _is_main_process():
            trainable = sum(p.numel() for p in self.desc_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.desc_model.parameters())
            state = f"unfrozen last {k} blocks" if k > 0 else "fully frozen"
            print(f"[desc_model] {state}; trainable {trainable:,} / {total:,}")

        dtype = next(self.decoder.parameters()).dtype
        device = next(self.decoder.parameters()).device

        # 5) Projections, CLS/SEP slots, head
        self.cnn_proj = nn.Linear(
            self.dna_encoder.output_channels, self.gen_hidden_size, device=device, dtype=dtype
        )
        self.cnn_ln = nn.LayerNorm(self.gen_hidden_size, device=device, dtype=dtype)
        # Learned stand-ins for the tokenizer's CLS/SEP: the tower and the decoder
        # are driven by inputs_embeds, so there is no table to look them up in.
        # Keeping both reproduces the token model's [CLS] + content + [SEP] layout
        # and with it the label layout the loss and metric code expects.
        self.cls_embedding = nn.Parameter(
            torch.zeros(1, 1, self.gen_hidden_size, device=device, dtype=dtype)
        )
        self.sep_embedding = nn.Parameter(
            torch.zeros(1, 1, self.gen_hidden_size, device=device, dtype=dtype)
        )
        nn.init.normal_(self.cls_embedding, std=0.02)
        nn.init.normal_(self.sep_embedding, std=0.02)

        self.dna_ln = nn.LayerNorm(self.gen_hidden_size, device=device, dtype=dtype)
        self.desc_ln = nn.LayerNorm(self.gen_hidden_size, device=device, dtype=dtype)
        self.desc_proj = nn.Linear(
            self.desc_hidden_size, self.gen_hidden_size, device=device, dtype=dtype
        )
        self.classifier = nn.Linear(
            self.decoder.config.hidden_size, 1, device=device, dtype=dtype
        )

        # 6) Loss
        self.activation = activation
        self.loss_fct = loss_fct
        self.weight = weight
        self.use_deviation_loss = use_deviation_loss
        self.use_multinomial_loss = use_multinomial_loss
        self.weight_deviation_loss = weight_deviation_loss
        self.weight_multinomial_loss = weight_multinomial_loss

        if _is_main_process():
            print(self.dna_encoder.parameter_report())
            print(
                f"[model] decoder sees {self.seq_len} positions "
                f"(CLS + {self.out_bins} bins @ {self.dna_encoder.bin_size} bp + SEP) "
                f"= {self.out_bins * self.dna_encoder.bin_size:,} bp"
            )

    @property
    def cnn(self) -> nn.Module:
        """Alias so the shared runner does not have to know about this class.

        ``run_expression_finetuning_cnn.py`` reads ``model.cnn`` for one log line
        reporting the DNA encoder's parameter count. ``ExpressionCountsCNN`` calls
        it ``cnn``; exposing the same name here keeps that runner byte-identical
        for both models.
        """
        return self.dna_encoder

    @staticmethod
    def _freeze_token_embeddings(model: nn.Module, tag: str) -> None:
        """Driven by inputs_embeds, so the table gets no gradient.

        DDP with find_unused_parameters=False (what the runner uses) crashes on
        parameters that require grad but never receive one.
        """
        embeddings = getattr(model, "embeddings", None)
        if embeddings is None:
            return
        for attr in ("tok_embeddings", "word_embeddings"):
            table = getattr(embeddings, attr, None)
            if table is not None:
                table.weight.requires_grad_(False)
                if _is_main_process():
                    print(f"[{tag}] froze embeddings.{attr} (unused: driven by inputs_embeds)")
                return

    def encode_dna(self, dna_codes: torch.Tensor) -> torch.Tensor:
        """(B, S) codes -> (B, out_bins + 2, H): CLS + bins + SEP."""
        batch_size = dna_codes.shape[0]
        features = self.dna_encoder(dna_codes)  # (B, out_bins, C)
        assert features.shape[:2] == (batch_size, self.out_bins), features.shape

        embeds = self.cnn_ln(self.cnn_proj(features.to(self.cnn_proj.weight.dtype)))
        cls = self.cls_embedding.expand(batch_size, -1, -1).to(embeds.dtype)
        sep = self.sep_embedding.expand(batch_size, -1, -1).to(embeds.dtype)
        embeds = torch.cat([cls, embeds, sep], dim=1)
        assert embeds.shape == (batch_size, self.seq_len, self.gen_hidden_size), embeds.shape
        return embeds

    def forward(
        self,
        dna_codes=None,              # (B, S)          one window per gene
        attention_mask=None,         # (B, seq_len)
        labels_mask=None,            # (B, N, seq_len, 1)
        labels=None,                 # (B, N, seq_len, 1)
        return_dict=None,
        desc_input_ids=None,         # (B, N, D)
        desc_attention_mask=None,    # (B, N, D)
        desc_index=None,             # (B, N)          batch-local description ids
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if dna_codes is None:
            raise ValueError("dna_codes must be provided")
        if dna_codes.dim() != 2:
            raise ValueError(f"dna_codes must be (B, S), got {tuple(dna_codes.shape)}")
        if desc_input_ids is None or desc_input_ids.dim() != 3:
            raise ValueError("desc_input_ids must be (B, N, D); one row per track of the gene")

        batch_size = dna_codes.shape[0]
        n_keys = desc_input_ids.shape[1]
        rows = batch_size * n_keys
        seq_len = self.seq_len

        if attention_mask is None:
            attention_mask = dna_codes.new_ones((batch_size, seq_len), dtype=torch.long)
        if attention_mask.shape != (batch_size, seq_len):
            raise ValueError(
                f"attention_mask must be {(batch_size, seq_len)} "
                f"(CLS + {self.out_bins} bins + SEP), got {tuple(attention_mask.shape)}"
            )

        # 1) DNA: shared by every track of the gene, so encoder (+ tower) run once
        #    per gene and the result is broadcast over the n_keys rows.
        inputs_embeds = self.encode_dna(dna_codes)
        if self.use_tower:
            bert_outputs = self.bert(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )
            sequence_output = bert_outputs.last_hidden_state
            attentions = bert_outputs.attentions
        else:
            sequence_output = inputs_embeds
            attentions = None

        sequence_output = sequence_output.repeat_interleave(n_keys, dim=0)
        attention_mask = attention_mask.repeat_interleave(n_keys, dim=0)
        assert sequence_output.shape[:2] == (rows, seq_len), sequence_output.shape

        # 2) Description: only a handful of distinct tracks per batch, so encode
        #    each once and scatter the result back over the rows.
        flat_ids = desc_input_ids.reshape(rows, -1)
        flat_mask = desc_attention_mask.reshape(rows, -1)
        if desc_index is None:
            unique_rows = torch.arange(rows, device=flat_ids.device)
            inverse = unique_rows
        else:
            _, inverse = torch.unique(desc_index.reshape(-1), return_inverse=True)
            unique_rows = torch.zeros(
                int(inverse.max()) + 1, dtype=torch.long, device=inverse.device
            )
            unique_rows[inverse] = torch.arange(rows, device=inverse.device)

        desc_out = self.desc_model(
            input_ids=flat_ids[unique_rows],
            attention_mask=flat_mask[unique_rows],
            return_dict=True,
        )
        # desc_model is bf16 while desc_proj holds fp32 weights, so the cast is
        # required outside autocast (inference, profiling); under it, a no-op.
        desc_pooled = desc_out.last_hidden_state[:, -1]
        desc_pooled = self.desc_proj(desc_pooled.to(self.desc_proj.weight.dtype))
        desc_pooled = desc_pooled.to(sequence_output.dtype)[inverse]
        assert desc_pooled.shape[0] == rows, desc_pooled.shape

        # 3) Fusion (unchanged from the token model)
        sequence_output = self.dna_ln(sequence_output)
        desc_output = self.desc_ln(desc_pooled)
        desc_broadcast = desc_output[:, None, :] * attention_mask[:, :, None].to(
            sequence_output.dtype
        )
        sequence_output = sequence_output + desc_broadcast

        # 4) Decoder + head
        dec_out = self.decoder(
            inputs_embeds=sequence_output,
            attention_mask=attention_mask,
            return_dict=True,
        )
        decoder_output = dec_out.last_hidden_state
        logits = self.activation(self.classifier(decoder_output))  # (rows, seq_len, 1)

        # 5) Loss: CLS position carries the gene-level target, the rest the bins.
        loss = None
        labels_reshaped = labels_mask_reshaped = None
        cls_loss = other_loss = deviation_loss = multinomial_loss = None

        if labels is not None:
            labels_reshaped = labels.reshape(rows, seq_len, 1).to(logits.device)
            labels_mask_reshaped = (
                labels_mask.reshape(rows, seq_len, 1).to(logits.device)
                if labels_mask is not None
                else None
            )

            unreduced_loss = self.loss_fct(logits, labels_reshaped)

            if labels_mask_reshaped is not None and labels_mask_reshaped.sum() > 0:
                cls_mask = labels_mask_reshaped[:, 0:1, :]
                other_mask = labels_mask_reshaped[:, 1:, :]

                if cls_mask.sum() > 0:
                    cls_loss = (unreduced_loss[:, 0:1, :] * cls_mask).sum() / (
                        cls_mask.sum() + 1e-8
                    )
                    if self.use_deviation_loss and n_keys > 1:
                        deviation_loss = cls_deviation_from_mean_loss(
                            logits, labels_reshaped, labels_mask_reshaped, n_keys=n_keys
                        )
                    if self.use_multinomial_loss and n_keys > 1:
                        multinomial_loss = cls_multinomial_loss(
                            logits, labels_reshaped, labels_mask_reshaped, n_keys=n_keys
                        )
                if other_mask.sum() > 0:
                    other_loss = (unreduced_loss[:, 1:, :] * other_mask).sum() / (
                        other_mask.sum() + 1e-8
                    )

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

        return ExpressionModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=(decoder_output,),
            attentions=attentions,
            labels_reshaped=labels_reshaped,
            labels_mask_reshaped=labels_mask_reshaped,
            cls_loss=cls_loss,
            other_loss=other_loss,
            deviation_loss=deviation_loss,
            multinomial_loss=multinomial_loss,
        )
