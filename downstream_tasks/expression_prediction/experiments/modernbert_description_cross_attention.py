from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.modernbert.modeling_modernbert import (
    _pad_modernbert_output,
    _unpad_modernbert_input,
)


class ResidualDescriptionCrossAttention(nn.Module):
    """One small residual cross-attention adapter.

    DNA tokens are the queries.  Description tokens are the keys and values.
    Consequently, every DNA token can choose which parts of the experiment
    description are relevant to its current representation.

    Parameters
    ----------
    hidden_size:
        Hidden size of the DNA ModernBERT.  Description states must already be
        projected to this size before entering this block.
    num_heads:
        Number of cross-attention heads. ``hidden_size`` must be divisible by
        this value.
    dropout:
        Dropout applied by multi-head attention and to its output.
    initial_residual_scale:
        Initial contribution of the new branch.  A small value protects the
        pretrained DNA representation at the beginning of fine-tuning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        initial_residual_scale: float = 0.1,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by "
                f"num_heads={num_heads}"
            )

        factory_kwargs = {"device": device, "dtype": dtype}

        # Pre-normalization makes the new branch insensitive to scale
        # differences between the two independently pretrained encoders.
        self.query_norm = nn.LayerNorm(hidden_size, **factory_kwargs)
        self.context_norm = nn.LayerNorm(hidden_size, **factory_kwargs)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            **factory_kwargs,
        )
        self.dropout = nn.Dropout(dropout)

        # tanh keeps the residual multiplier bounded in [-1, 1].  Initializing
        # it to 0.1 makes the adapter visible from the first step, while keeping
        # the initial model close to the pretrained DNA model.
        self.residual_scale = nn.Parameter(
            torch.tensor(initial_residual_scale, **factory_kwargs)
        )

    def forward(
        self,
        dna_states: torch.Tensor,
        description_states: torch.Tensor,
        dna_attention_mask: torch.Tensor,
        description_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Condition padded DNA states on padded description states.

        Shapes
        ------
        dna_states:
            ``(batch, dna_length, hidden_size)``
        description_states:
            ``(batch, description_length, hidden_size)``
        dna_attention_mask:
            ``(batch, dna_length)``; nonzero values mark real DNA tokens.
        description_attention_mask:
            ``(batch, description_length)``; nonzero values mark real
            description tokens.
        """
        if dna_states.ndim != 3 or description_states.ndim != 3:
            raise ValueError("DNA and description states must both be rank-3 tensors")
        if dna_states.shape[0] != description_states.shape[0]:
            raise ValueError(
                "DNA and description batch sizes must match; got "
                f"{dna_states.shape[0]} and {description_states.shape[0]}"
            )
        if dna_attention_mask.shape != dna_states.shape[:2]:
            raise ValueError("dna_attention_mask does not match dna_states")
        if description_attention_mask.shape != description_states.shape[:2]:
            raise ValueError(
                "description_attention_mask does not match description_states"
            )

        description_mask = description_attention_mask.bool()
        if not description_mask.any(dim=1).all():
            raise ValueError("Every description must contain at least one real token")

        normalized_description = self.context_norm(description_states)
        attended, _ = self.attention(
            query=self.query_norm(dna_states),
            key=normalized_description,
            value=normalized_description,
            # PyTorch uses True to mean "ignore this key".
            key_padding_mask=~description_mask,
            # Attention maps are not needed for training and can be very large.
            need_weights=False,
        )

        output = dna_states + torch.tanh(self.residual_scale) * self.dropout(attended)

        # Keep padding states exactly zero.  They must not leak into later
        # layers when a padded batch is used instead of FlashAttention unpadding.
        return output * dna_attention_mask.unsqueeze(-1).to(output.dtype)


class DescriptionConditionedModernBert(nn.Module):
    """Combine a DNA ModernBERT and a description encoder using cross-attention.

    The constructor takes model *objects*, not model names.  Loading, freezing,
    checkpoint selection, and dropout configuration therefore remain under the
    control of the original training code.

    Parameters
    ----------
    dna_model:
        A Hugging Face ``ModernBertModel`` instance.  Structurally it must have
        ``embeddings``, ``layers``, ``final_norm``, and ``config`` attributes.
    description_model:
        Any Hugging Face encoder whose output has ``last_hidden_state`` and
        whose config exposes ``hidden_size``.
    cross_attention_after_layers:
        One-based DNA layer numbers after which cross-attention is applied.
        ``None`` chooses ``(3, 14, 28)`` for a 28-layer model.  One-based values
        make configuration files easier to read than zero-based Python indices.
    cross_attention_heads:
        Number of heads in each new adapter.  By default, reuse the DNA model's
        number of self-attention heads.
    dropout:
        Cross-attention dropout only.  Existing model dropout is unchanged.
    initial_residual_scale:
        Initial strength of every new residual cross-attention branch.
    freeze_description_model:
        If true, description-model parameters are frozen and it is kept in eval
        mode.  The shared projection and cross-attention adapters still train.
    enable_gradient_checkpointing:
        Ask both supplied models to enable their native gradient checkpointing
        and checkpoint the three new adapters.  This reduces activation memory
        but increases backward computation.
    disable_reference_compile:
        Disable ModernBERT's internal compiled-MLP path.  This should remain
        true when gradient checkpointing is used; some PyTorch/Transformers
        combinations otherwise fail during FakeTensor tracing.

    Forward inputs
    --------------
    ``input_ids`` and ``attention_mask`` describe DNA. ``desc_input_ids`` and
    ``desc_attention_mask`` describe the experiment.  Both sides must already
    have the same flattened batch size.

    The returned object is the same ``BaseModelOutput`` type returned by
    ``ModernBertModel``, so downstream code can continue to use
    ``output.last_hidden_state``.
    """

    def __init__(
        self,
        dna_model: nn.Module,
        description_model: nn.Module,
        cross_attention_after_layers: Optional[Sequence[int]] = None,
        cross_attention_heads: Optional[int] = None,
        dropout: float = 0.1,
        initial_residual_scale: float = 0.1,
        freeze_description_model: bool = False,
        enable_gradient_checkpointing: bool = False,
        disable_reference_compile: bool = True,
    ) -> None:
        super().__init__()

        for attribute in ("config", "embeddings", "layers", "final_norm"):
            if not hasattr(dna_model, attribute):
                raise TypeError(f"dna_model is missing required attribute {attribute!r}")
        if not hasattr(description_model, "config"):
            raise TypeError("description_model must expose a Hugging Face config")

        self.dna_model = dna_model
        self.description_model = description_model
        self.config = dna_model.config
        self.freeze_description_model = freeze_description_model
        self.gradient_checkpointing_enabled = enable_gradient_checkpointing

        dna_hidden_size = getattr(dna_model.config, "hidden_size", None)
        description_hidden_size = getattr(
            description_model.config, "hidden_size", None
        )
        if dna_hidden_size is None or description_hidden_size is None:
            raise ValueError("Both model configs must define hidden_size")

        number_of_layers = len(dna_model.layers)
        if cross_attention_after_layers is None:
            # For ModernBERT-large (28 layers), this is exactly (3, 14, 28).
            cross_attention_after_layers = (
                min(3, number_of_layers),
                max(1, number_of_layers // 2),
                number_of_layers,
            )

        layer_numbers = tuple(int(value) for value in cross_attention_after_layers)
        if not layer_numbers:
            raise ValueError("At least one cross-attention position is required")
        if len(set(layer_numbers)) != len(layer_numbers):
            raise ValueError(f"Cross-attention layer numbers must be unique: {layer_numbers}")
        invalid = [value for value in layer_numbers if not 1 <= value <= number_of_layers]
        if invalid:
            raise ValueError(
                f"Cross-attention layer numbers {invalid} are outside "
                f"1..{number_of_layers}"
            )
        self.cross_attention_after_layers = tuple(sorted(layer_numbers))

        first_dna_parameter = next(dna_model.parameters())
        factory_kwargs = {
            "device": first_dna_parameter.device,
            "dtype": first_dna_parameter.dtype,
        }
        number_of_heads = cross_attention_heads or getattr(
            dna_model.config, "num_attention_heads", None
        )
        if number_of_heads is None:
            raise ValueError(
                "cross_attention_heads must be provided when the DNA config "
                "does not define num_attention_heads"
            )

        # One projection is shared by all three adapters.  It is the only
        # operation needed when the description and DNA hidden sizes differ.
        self.description_projection = nn.Linear(
            description_hidden_size,
            dna_hidden_size,
            **factory_kwargs,
        )

        # ModuleDict keys are strings because state-dict paths must be strings.
        # Human-readable one-based layer numbers also make checkpoints clearer.
        self.cross_attention = nn.ModuleDict(
            {
                str(layer_number): ResidualDescriptionCrossAttention(
                    hidden_size=dna_hidden_size,
                    num_heads=number_of_heads,
                    dropout=dropout,
                    initial_residual_scale=initial_residual_scale,
                    **factory_kwargs,
                )
                for layer_number in self.cross_attention_after_layers
            }
        )

        if freeze_description_model:
            self.description_model.requires_grad_(False)
            self.description_model.eval()

        if disable_reference_compile and hasattr(
            self.dna_model.config, "reference_compile"
        ):
            self.dna_model.config.reference_compile = False
            for layer in self.dna_model.layers:
                if hasattr(layer, "config"):
                    layer.config.reference_compile = False

        if enable_gradient_checkpointing:
            self._enable_gradient_checkpointing(self.dna_model)
            if not freeze_description_model:
                self._enable_gradient_checkpointing(self.description_model)

    @staticmethod
    def _enable_gradient_checkpointing(model: nn.Module) -> None:
        """Enable a Hugging Face model's native checkpointing when available."""
        enable = getattr(model, "gradient_checkpointing_enable", None)
        if enable is None:
            return
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # Compatibility with older Transformers versions.
            enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    def train(self, mode: bool = True):
        """Keep a frozen description model in eval mode during outer training."""
        super().train(mode)
        if self.freeze_description_model:
            self.description_model.eval()
        return self

    def _encode_description(
        self,
        desc_input_ids: torch.Tensor,
        desc_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return all projected description-token embeddings, without pooling."""
        context = self.description_model(
            input_ids=desc_input_ids,
            attention_mask=desc_attention_mask,
            return_dict=True,
        ).last_hidden_state
        context = context.to(self.description_projection.weight.dtype)
        return self.description_projection(context)

    def _apply_cross_attention(
        self,
        hidden_states: torch.Tensor,
        layer_number: int,
        description_states: torch.Tensor,
        dna_attention_mask: torch.Tensor,
        description_attention_mask: torch.Tensor,
        *,
        flash_attention: bool,
        unpadded_indices: Optional[torch.Tensor],
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        """Apply one adapter, repadding only when FlashAttention requires it."""
        if flash_attention:
            padded_states = _pad_modernbert_output(
                inputs=hidden_states,
                indices=unpadded_indices,
                batch=batch_size,
                seqlen=sequence_length,
            )
        else:
            padded_states = hidden_states

        adapter = self.cross_attention[str(layer_number)]
        adapter_inputs = (
            padded_states,
            description_states,
            dna_attention_mask,
            description_attention_mask,
        )
        if (
            self.gradient_checkpointing_enabled
            and self.training
            and torch.is_grad_enabled()
        ):
            conditioned_states = checkpoint(
                adapter,
                *adapter_inputs,
                use_reentrant=False,
            )
        else:
            conditioned_states = adapter(*adapter_inputs)

        if flash_attention:
            # Use the same indices ModernBERT created at the start of the
            # forward pass; token ordering has not changed.
            conditioned_states = conditioned_states.reshape(
                batch_size * sequence_length, conditioned_states.shape[-1]
            )[unpadded_indices]
        return conditioned_states

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        desc_input_ids: Optional[torch.LongTensor] = None,
        desc_attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """Encode DNA while injecting description context three times.

        ``input_ids``/``inputs_embeds`` and ``desc_input_ids`` must have the
        same first dimension.  This method deliberately does not know about
        ``dataset_flag``; flattening and grouped losses remain responsibilities
        of the surrounding expression model.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if desc_input_ids is None:
            raise ValueError("desc_input_ids must be provided")

        output_attentions = (
            self.config.output_attentions
            if output_attentions is None
            else output_attentions
        )
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict

        dna_source = input_ids if input_ids is not None else inputs_embeds
        batch_size, sequence_length = dna_source.shape[:2]
        device = dna_source.device
        if desc_input_ids.shape[0] != batch_size:
            raise ValueError(
                "DNA and description rows must match after flattening; got "
                f"{batch_size} and {desc_input_ids.shape[0]}"
            )

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=device,
            )
        if desc_attention_mask is None:
            desc_attention_mask = torch.ones_like(desc_input_ids, dtype=torch.bool)

        description_states = self._encode_description(
            desc_input_ids=desc_input_ids,
            desc_attention_mask=desc_attention_mask,
        )

        flash_attention = (
            self.dna_model.config._attn_implementation == "flash_attention_2"
        )
        repad_output = False

        if flash_attention:
            repad_output = True
            if input_ids is not None:
                # Token IDs never require gradients, so their indexing metadata
                # can be prepared without building an autograd graph.
                with torch.no_grad():
                    (
                        input_ids,
                        unpadded_indices,
                        cu_seqlens,
                        max_seqlen,
                        _,
                        _,
                    ) = _unpad_modernbert_input(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                    )
            else:
                (
                    inputs_embeds,
                    unpadded_indices,
                    cu_seqlens,
                    max_seqlen,
                    _,
                    _,
                ) = _unpad_modernbert_input(
                    inputs=inputs_embeds,
                    attention_mask=attention_mask,
                )
            position_ids = None
            sliding_window_mask = None
            layer_attention_mask = attention_mask
        else:
            unpadded_indices = None
            cu_seqlens = None
            max_seqlen = None
            position_ids = torch.arange(sequence_length, device=device).unsqueeze(0)
            (
                layer_attention_mask,
                sliding_window_mask,
            ) = self.dna_model._update_attention_mask(
                attention_mask,
                output_attentions=output_attentions,
            )

        hidden_states = self.dna_model.embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for zero_based_index, encoder_layer in enumerate(self.dna_model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_self_attentions += (layer_outputs[1],)

            layer_number = zero_based_index + 1
            if layer_number in self.cross_attention_after_layers:
                hidden_states = self._apply_cross_attention(
                    hidden_states=hidden_states,
                    layer_number=layer_number,
                    description_states=description_states,
                    dna_attention_mask=attention_mask,
                    description_attention_mask=desc_attention_mask,
                    flash_attention=flash_attention,
                    unpadded_indices=unpadded_indices,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        hidden_states = self.dna_model.final_norm(hidden_states)

        if repad_output:
            hidden_states = _pad_modernbert_output(
                inputs=hidden_states,
                indices=unpadded_indices,
                batch=batch_size,
                seqlen=sequence_length,
            )
            if all_hidden_states is not None:
                all_hidden_states = tuple(
                    _pad_modernbert_output(
                        inputs=state,
                        indices=unpadded_indices,
                        batch=batch_size,
                        seqlen=sequence_length,
                    )
                    for state in all_hidden_states
                )

        if not return_dict:
            return tuple(
                value
                for value in (
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                )
                if value is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
