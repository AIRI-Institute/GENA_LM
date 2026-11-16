"""Independent short DNA encoding, center-out recurrence only in the combining decoder."""

import re
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoModel, ModernBertModel

from downstream_tasks.expression_prediction.armt_hf_encoder_vendor import InnerLoopAssociativeLayerWrapper
from downstream_tasks.expression_prediction.expression_model_final import (
    ExpressionModelOutput, cls_deviation_from_mean_loss, cls_multinomial_loss,
)


@dataclass
class ARMTExpressionModelOutput(ExpressionModelOutput):
    # DDP reconstructs ModelOutput through its constructor. Keep variable-length
    # diagnostics in a declared field instead of adding undeclared mapping keys.
    armt_metrics: Optional[Dict[str, torch.Tensor]] = None


def center_out_offsets(count):
    if count < 1 or count % 2 != 1:
        raise ValueError("num_segments must be a positive odd integer")
    return [0] + [offset for radius in range(1, count // 2 + 1) for offset in (-radius, radius)]


def segment_indices(tss_index, offset, payload_size, attention_mask):
    """Gather DNA positions only; preserve transcription order inside each segment."""
    start = tss_index[:, None] - payload_size // 2 + offset * payload_size
    indices = start + torch.arange(payload_size, device=tss_index.device)[None]
    lengths = attention_mask.long().sum(-1)
    valid = (indices >= 1) & (indices < lengths[:, None] - 1) & (tss_index[:, None] >= 1)
    return indices.clamp(0, attention_mask.shape[1] - 1), valid


def pack_segment(input_ids, indices, valid, cls_id, sep_id, pad_id):
    """Pack the already-selected DNA, without shifting the genomic window.

    Return source indices in packed order so supervision and scattered signal
    predictions still refer to the original cached token positions.
    """
    rows, payload_size = indices.shape
    positions = torch.arange(payload_size, device=indices.device)[None]
    order = (positions + (~valid).long() * payload_size).argsort(dim=1)
    indices, valid = indices.gather(1, order), valid.gather(1, order)
    ids = input_ids.new_full((rows, payload_size + 2), pad_id)
    ids[:, 0] = cls_id
    ids[:, 1:-1] = input_ids.gather(1, indices).masked_fill(~valid, pad_id)
    lengths = valid.long().sum(-1)
    ids.scatter_(1, (lengths + 1)[:, None], sep_id)
    mask = torch.arange(payload_size + 2, device=ids.device)[None] < lengths[:, None] + 2
    return ids, mask, indices, valid


def flash_memory_attention(qkv, cu_seqlens, max_seqlen, num_mem_tokens,
                           local_attention, dropout_p=0., deterministic=False, kernel=None):
    """FA2 with the authors' mask: DNA cannot read memory-token keys;
    memory queries read all DNA and causally read the memory-token prefix.

    Two varlen calls preserve this asymmetric mask without a dense attention
    matrix. FA2 >=2.1 uses bottom-right causal alignment for unequal Q/K lengths.
    A supplied kernel is only used by CPU/reference tests.
    """
    if kernel is None:
        from flash_attn import flash_attn_varlen_func
        kernel = flash_attn_varlen_func
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    batch = len(lengths)
    row = torch.repeat_interleave(torch.arange(batch, device=qkv.device), lengths.long())
    position = torch.arange(qkv.shape[0], device=qkv.device) - cu_seqlens[row]
    is_real = position < lengths[row] - num_mem_tokens
    real_indices = is_real.nonzero(as_tuple=True)[0]
    memory_indices = (~is_real).nonzero(as_tuple=True)[0]
    real_cu = torch.cat([cu_seqlens.new_zeros(1), (lengths-num_mem_tokens).cumsum(0)]).to(torch.int32)
    memory_cu = torch.arange(batch+1, device=qkv.device, dtype=torch.int32) * num_mem_tokens
    real = qkv[real_indices]
    real_output = kernel(real[:,0], real[:,1], real[:,2],
        cu_seqlens_q=real_cu, cu_seqlens_k=real_cu,
        max_seqlen_q=max_seqlen-num_mem_tokens, max_seqlen_k=max_seqlen-num_mem_tokens,
        dropout_p=dropout_p, causal=False, window_size=local_attention, deterministic=deterministic)
    memory_output = kernel(qkv[memory_indices,0], qkv[:,1], qkv[:,2],
        cu_seqlens_q=memory_cu, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=num_mem_tokens, max_seqlen_k=max_seqlen,
        dropout_p=dropout_p, causal=True, window_size=(-1,-1), deterministic=deterministic)
    return torch.zeros_like(qkv[:,0]).index_copy(0, real_indices, real_output).index_copy(
        0, memory_indices, memory_output)


class CenterFlashAttention(nn.Module):
    """ModernBERT FA2 projections/RoPE, retaining original checkpoint key names."""
    def __init__(self, original, num_mem_tokens):
        super().__init__()
        self.Wqkv, self.Wo = original.Wqkv, original.Wo
        self.rotary_emb, self.out_drop = original.rotary_emb, original.out_drop
        self.num_heads, self.head_dim = original.num_heads, original.head_dim
        self.all_head_size = original.all_head_size
        self.local_attention = original.local_attention
        self.attention_dropout = original.attention_dropout
        self.deterministic_flash_attn = original.deterministic_flash_attn
        self.num_mem_tokens = num_mem_tokens

    def forward(self, hidden_states, cu_seqlens=None, max_seqlen=None,
                output_attentions=False, **kwargs):
        if hidden_states.ndim != 2 or cu_seqlens is None:
            raise ValueError('ARMT FA2 expects ModernBERT unpadded tokens and cu_seqlens')
        if output_attentions:
            raise ValueError('FA2 does not return dense attention weights')
        qkv = self.Wqkv(hidden_states).view(-1,3,self.num_heads,self.head_dim)
        qkv = self.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        original_dtype = qkv.dtype
        if qkv.dtype not in (torch.float16, torch.bfloat16):
            qkv = qkv.to(torch.bfloat16)
        output = flash_memory_attention(qkv, cu_seqlens, max_seqlen, self.num_mem_tokens,
            self.local_attention, self.attention_dropout if self.training else 0.,
            self.deterministic_flash_attn).reshape(-1,self.all_head_size).to(original_dtype)
        return (self.out_drop(self.Wo(output)),)


class CenterMemoryLayer(InnerLoopAssociativeLayerWrapper):
    """HF read/write equations, with one ModernBERT segment per call.

    ModernBERT supplies a second local-attention mask and can unpad FlashAttention
    inputs to 2D. The FA2 path preserves sequence boundaries for memory reads and
    writes; the SDPA/eager reference path augments BOTH attention masks.
    Memory state remains attached to autograd until the complete example finishes.
    """

    active_rows = None

    def _memory_mask(self, mask, hidden_states):
        real_length = hidden_states.shape[1] - self.num_mem_tokens
        if mask is None:
            raise ValueError("ModernBERT must provide an explicit SDPA/eager mask")
        real = mask[..., :real_length, :real_length]
        allowed = real if real.dtype == torch.bool else real == 0
        allowed = self.pad_attention_mask(allowed, hidden_states.dtype).bool()
        # Padding queries can have no local keys. A private self-edge prevents
        # all-masked SDPA rows (NaN in torch 2.4/bf16); real queries still cannot
        # attend to padding keys. This does not expose fake DNA to real tokens.
        empty = ~allowed.any(dim=-1, keepdim=True)
        diagonal = torch.eye(allowed.shape[-1], device=allowed.device, dtype=torch.bool)
        allowed = allowed | (empty & diagonal)
        return torch.zeros_like(allowed, dtype=hidden_states.dtype).masked_fill(
            ~allowed, torch.finfo(hidden_states.dtype).min)

    def forward(self, hidden_states, attention_mask=None, sliding_window_mask=None,
                position_ids=None, cu_seqlens=None, max_seqlen=None, output_attentions=False):
        packed = cu_seqlens is not None
        if packed:
            if hidden_states.ndim != 2:
                raise ValueError('FA2 memory wrapper expects unpadded 2D hidden states')
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            row = torch.repeat_interleave(torch.arange(len(lengths), device=hidden_states.device), lengths.long())
            position = torch.arange(hidden_states.shape[0],device=hidden_states.device) - cu_seqlens[row]
            padded = hidden_states.new_zeros(len(lengths), max_seqlen, hidden_states.shape[-1])
            hidden_states = padded.index_put((row,position.long()), hidden_states)
        elif hidden_states.ndim != 3:
            raise ValueError('SDPA/eager memory wrapper expects padded 3D hidden states')
        first = self.memory_state is None
        old_memory, old_z = (self._alloc_initial_mem(hidden_states.device, self.memory_dtype)
                             if first else self.memory_state)
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            association = None if first else self._associate_with_mem(hidden_states, old_memory, old_z)
        current = hidden_states if first else hidden_states + association
        if packed:
            result = self.layer(current[row,position.long()], cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen, output_attentions=output_attentions)
            memory_indices = cu_seqlens[1:,None].long()-self.num_mem_tokens + torch.arange(
                self.num_mem_tokens,device=cu_seqlens.device)[None]
            memory_tokens = result[0][memory_indices]
        else:
            result = self.layer(
                current, attention_mask=self._memory_mask(attention_mask, current),
                sliding_window_mask=self._memory_mask(sliding_window_mask, current),
                position_ids=position_ids, output_attentions=output_attentions)
            memory_tokens = result[0][:, -self.num_mem_tokens:]
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            new_memory, new_z, _ = self._update_mem_with_mem(
                memory_tokens, old_memory, old_z, first)
        active = self.active_rows
        if active is not None:
            new_memory = torch.where(active[:, None, None, None], new_memory, old_memory)
            if new_z is not None:
                new_z = torch.where(active[:, None, None], new_z, old_z)
        self.memory_state = new_memory, new_z
        return result


class ExpressionCountsARMT(nn.Module):
    def __init__(self, config, hf_model_name_decoder, hf_model_name,
                 desc_model_name="Qwen/Qwen3-Embedding-0.6B", hf=True,
                 loss_fct=None, activation=None, weight=0.05,
                 num_segments=1, max_segments=17, segment_size=1024,
                 num_mem_tokens=32, d_mem=64, n_heads=1,
                 memory_dtype="float32", dropout_prob=0.1,
                 freeze_dna_encoder=False, desc_trainable_layers=4,
                 use_deviation_loss=False, use_multinomial_loss=False,
                 weight_deviation_loss=1.0, weight_multinomial_loss=1.0,
                 dna_checkpointing=False, attn_implementation="flash_attention_2",
                 dna_attn_implementation="flash_attention_2"):
        super().__init__()
        self.offsets = center_out_offsets(num_segments)
        center_out_offsets(max_segments)
        if num_segments > max_segments or segment_size < 4 or num_mem_tokens < 1:
            raise ValueError("Invalid segment/memory sizes")
        if segment_size % 2:
            raise ValueError("segment_size must be even so that the DNA payload centers on TSS")
        if not hf or attn_implementation not in {"flash_attention_2", "sdpa", "eager"}:
            raise ValueError("This implementation supports HF ModernBERT with FA2/SDPA/eager")
        if dna_attn_implementation not in {"flash_attention_2", "sdpa", "eager"}:
            raise ValueError("Unsupported DNA attention implementation")
        if attn_implementation == 'flash_attention_2':
            import flash_attn
            from packaging.version import Version
            if Version(flash_attn.__version__) < Version('2.1'):
                raise ValueError('ARMT FA2 requires flash-attn>=2.1 for bottom-right causal alignment')
        if memory_dtype != "float32":
            raise ValueError("Associative memory accumulation currently requires float32")
        self.num_segments, self.max_segments = num_segments, max_segments
        self.segment_size, self.payload_size = segment_size, segment_size - 2
        self.num_mem_tokens = num_mem_tokens
        self.freeze_dna_encoder = freeze_dna_encoder
        self.bert = ModernBertModel.from_pretrained(
            hf_model_name, attn_implementation=dna_attn_implementation,
            attention_dropout=dropout_prob, embedding_dropout=dropout_prob,
            mlp_dropout=dropout_prob, reference_compile=False)
        self.config = self.bert.config
        self.desc_model_name = desc_model_name
        self.desc_model = AutoModel.from_pretrained(
            desc_model_name, attn_implementation="sdpa", torch_dtype=torch.float32)
        for parameter in self.desc_model.parameters():
            parameter.requires_grad_(False)
        desc_backbone = getattr(self.desc_model, "model", self.desc_model)
        desc_layers = desc_backbone.layers
        count = min(max(0, desc_trainable_layers), len(desc_layers))
        self._frozen_desc_layers = list(desc_layers[:len(desc_layers) - count])
        if count:
            for layer in desc_layers[-count:]:
                layer.requires_grad_(True)
            if getattr(desc_backbone, "norm", None) is not None:
                desc_backbone.norm.requires_grad_(True)
        self.decoder = ModernBertModel.from_pretrained(
            hf_model_name_decoder, attn_implementation=attn_implementation,
            # Only the combining backbone sees the appended memory tokens.
            max_position_embeddings=segment_size + num_mem_tokens,
            attention_dropout=dropout_prob, embedding_dropout=dropout_prob,
            mlp_dropout=dropout_prob, reference_compile=False)
        self.gen_hidden_size = self.config.hidden_size
        if self.decoder.config.hidden_size != self.gen_hidden_size:
            raise ValueError("DNA and combining decoder hidden sizes must match")
        self.desc_hidden_size = self.desc_model.config.hidden_size
        self.dna_ln = nn.LayerNorm(self.gen_hidden_size)
        self.desc_ln = nn.LayerNorm(self.gen_hidden_size)
        self.desc_proj = nn.Linear(self.desc_hidden_size, self.gen_hidden_size)
        self.classifier = nn.Linear(self.gen_hidden_size, 1)
        self.armt_memory = nn.Parameter(torch.empty(num_mem_tokens, self.gen_hidden_size))
        nn.init.normal_(self.armt_memory, std=0.02)
        self.segment_position = nn.Embedding(max_segments, self.gen_hidden_size)
        nn.init.normal_(self.segment_position.weight, std=0.02)
        for index, layer in enumerate(self.decoder.layers):
            if attn_implementation == 'flash_attention_2':
                layer.attn = CenterFlashAttention(layer.attn, num_mem_tokens)
            self.decoder.layers[index] = CenterMemoryLayer(
                layer, self.gen_hidden_size, num_mem_tokens, d_mem, segment_size,
                n_heads=n_heads, is_causal=False, attn_implementation=attn_implementation,
                memory_dtype=torch.float32, info={"layer": index})
        self.decoder.embeddings.tok_embeddings.weight.requires_grad_(False)
        if freeze_dna_encoder:
            self.bert.requires_grad_(False)
        if dna_checkpointing:
            self.bert.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.activation = activation if activation is not None else nn.Softplus()
        self.loss_fct = loss_fct if loss_fct is not None else nn.MSELoss(reduction="none")
        self.weight = weight
        self.use_deviation_loss, self.use_multinomial_loss = use_deviation_loss, use_multinomial_loss
        self.weight_deviation_loss, self.weight_multinomial_loss = weight_deviation_loss, weight_multinomial_loss
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_dna_encoder:
            self.bert.eval()
        for layer in self._frozen_desc_layers:
            layer.eval()
        return self

    def reset_memory(self):
        for layer in self.decoder.layers:
            layer.zero_mem()
            layer.active_rows = None

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Plain expression checkpoints store decoder.layers.N.*, wrapped stages
        # store decoder.layers.N.layer.*. DNA/Qwen/head names are preserved.
        targets = self.state_dict()
        mapped = {}
        for key, value in state_dict.items():
            new_key = key
            if key not in targets and re.match(r"decoder\.layers\.\d+\.", key):
                new_key = re.sub(r"^(decoder\.layers\.\d+)\.", r"\1.layer.", key)
            mapped[new_key] = value
        result = super().load_state_dict(mapped, strict=strict, assign=assign)
        if not strict:
            allowed_new = ("armt_memory", "segment_position.")
            bad = [key for key in result.missing_keys if not (
                key.startswith(allowed_new) or re.match(r"decoder\.layers\.\d+\.W_m", key))]
            if bad or result.unexpected_keys:
                raise RuntimeError(f"Incomplete expression checkpoint: missing={bad}, unexpected={result.unexpected_keys}")
        return result

    def forward(self, input_ids=None, attention_mask=None, labels_mask=None, labels=None,
                return_dict=None, desc_input_ids=None, desc_attention_mask=None,
                dataset_flag=None, tss_token_index=None):
        if dataset_flag is None or tss_token_index is None:
            raise ValueError("dataset_flag and tss_token_index are required")
        batch, n_keys = dataset_flag.shape
        rows = batch * n_keys
        input_ids = input_ids.reshape(rows, -1)
        attention_mask = attention_mask.reshape(rows, -1)
        tss_token_index = tss_token_index.reshape(rows).to(input_ids.device)
        labels = labels.reshape(rows, -1, 1) if labels is not None else None
        labels_mask = labels_mask.reshape(rows, -1, 1).bool() if labels_mask is not None else None
        if labels is not None and labels_mask is None:
            raise ValueError("labels_mask is required when labels are supplied")
        if self.decoder.is_gradient_checkpointing:
            raise RuntimeError("Decoder checkpointing cannot replay mutable ARMT memory; use DNA checkpointing only")
        grid = torch.arange(rows, device=input_ids.device).reshape(batch, n_keys)
        flag = dataset_flag[:, :1].bool().expand(batch, n_keys)
        dna_source = torch.where(flag, grid[:, :1].expand_as(grid), grid).reshape(-1)
        dna_unique, dna_map = torch.unique(dna_source, sorted=True, return_inverse=True)
        desc_source = torch.where(flag, grid, grid[:, :1].expand_as(grid)).reshape(-1)
        desc_unique, desc_map = torch.unique(desc_source, sorted=True, return_inverse=True)
        desc_ids = desc_input_ids.reshape(rows, -1)[desc_unique]
        desc_mask = desc_attention_mask.reshape(rows, -1)[desc_unique]
        desc_hidden = self.desc_model(input_ids=desc_ids, attention_mask=desc_mask, return_dict=True).last_hidden_state
        positions = torch.arange(desc_mask.shape[1], device=desc_mask.device)[None]
        last = positions.masked_fill(~desc_mask.bool(), -1).max(-1).values.clamp_min(0)
        desc = self.desc_ln(self.desc_proj(desc_hidden[torch.arange(len(last), device=last.device), last]))[desc_map]
        length = input_ids.shape[1]
        output_logits = desc.new_zeros(rows, length, 1)
        processed = torch.zeros_like(attention_mask, dtype=torch.bool)
        final_expression = desc.new_zeros(rows, 1)
        cls_sum = desc.new_zeros((), dtype=torch.float32)
        signal_sum = cls_sum.clone()
        cls_count = cls_sum.clone()
        signal_count = cls_sum.clone()
        deviation_sum, multinomial_sum = cls_sum.clone(), cls_sum.clone()
        regularizer_count = 0
        diagnostics = {}
        cls_id, sep_id = self.config.cls_token_id, self.config.sep_token_id
        if cls_id is None or sep_id is None:
            raise ValueError("DNA config must specify CLS and SEP token IDs")
        self.reset_memory()
        try:
            for step, offset in enumerate(self.offsets, 1):
                indices, valid = segment_indices(tss_token_index, offset, self.payload_size, attention_mask)
                active = valid.any(-1)
                local_ids, local_mask, indices, valid = pack_segment(
                    input_ids, indices, valid, cls_id, sep_id, self.config.pad_token_id)
                dna = self.bert(input_ids=local_ids[dna_unique], attention_mask=local_mask[dna_unique],
                                return_dict=True).last_hidden_state[dna_map]
                combined = self.dna_ln(dna) + desc[:, None].to(dna.dtype)
                position = self.segment_position.weight[offset + self.max_segments // 2]
                combined = combined + position.to(combined.dtype)
                memory = self.armt_memory.to(combined.dtype)[None].expand(rows, -1, -1)
                augmented = torch.cat([combined, memory], dim=1)
                augmented_mask = torch.cat([local_mask, torch.ones(rows, self.num_mem_tokens,
                                            device=valid.device, dtype=torch.bool)], dim=1)
                for layer in self.decoder.layers:
                    layer.active_rows = active
                decoded = self.decoder(inputs_embeds=augmented, attention_mask=augmented_mask,
                                       return_dict=True).last_hidden_state[:, :self.segment_size]
                predictions = self.activation(self.classifier(decoded))
                final_expression = torch.where(active[:, None], predictions[:, 0], final_expression)
                signal_prediction = predictions[:, 1:-1] * valid[:, :, None]
                output_logits = output_logits.scatter_add(1, indices[:, :, None], signal_prediction)
                visited = torch.zeros_like(attention_mask, dtype=torch.long).scatter_add_(1, indices, valid.long())
                processed |= visited.bool()
                if labels is not None:
                    expr_valid = labels_mask[:, 0, 0] & active
                    error = self.loss_fct(predictions[:, 0].float(), labels[:, 0].float()).reshape(rows)
                    step_sum = error.masked_fill(~expr_valid, 0).sum()
                    count = expr_valid.sum()
                    cls_sum, cls_count = cls_sum + step_sum, cls_count + count
                    signal_labels = labels.gather(1, indices[:, :, None])
                    signal_valid = labels_mask.gather(1, indices[:, :, None]) & valid[:, :, None]
                    errors = self.loss_fct(predictions[:, 1:-1].float(), signal_labels.float())
                    step_signal = errors.masked_fill(~signal_valid, 0).sum()
                    count_signal = signal_valid.sum()
                    signal_sum, signal_count = signal_sum + step_signal, signal_count + count_signal
                    diagnostics[f"armt_expr_loss_step_{step}"] = (step_sum / count.clamp_min(1)).detach()
                    diagnostics[f"armt_signal_loss_step_{step}"] = (step_signal / count_signal.clamp_min(1)).detach()
                    if expr_valid.any():
                        step_labels = labels[:, :1]
                        step_mask = expr_valid[:, None, None]
                        if self.use_deviation_loss:
                            deviation_sum = deviation_sum + cls_deviation_from_mean_loss(
                                predictions[:, :1], step_labels, step_mask, n_keys, dataset_flag[:, 0].bool())
                        if self.use_multinomial_loss:
                            multinomial_sum = multinomial_sum + cls_multinomial_loss(
                                predictions[:, :1], step_labels, step_mask, n_keys,
                                group_mask=dataset_flag[:, 0].bool())
                        regularizer_count += 1
            output_logits = torch.cat([final_expression[:, None], output_logits[:, 1:]], dim=1)
            output_mask = None
            cls_loss = other_loss = loss = deviation_loss = multinomial_loss = None
            if labels is not None:
                output_mask = labels_mask & processed[:, :, None]
                output_mask[:, 0] = labels_mask[:, 0] & processed.any(-1)[:, None]
                cls_loss = cls_sum / cls_count.clamp_min(1)
                other_loss = signal_sum / signal_count.clamp_min(1)
                loss = cls_loss + self.weight * other_loss
                if self.use_deviation_loss:
                    deviation_loss = deviation_sum / max(1, regularizer_count)
                    loss = loss + self.weight_deviation_loss * deviation_loss
                if self.use_multinomial_loss:
                    multinomial_loss = multinomial_sum / max(1, regularizer_count)
                    loss = loss + self.weight_multinomial_loss * multinomial_loss
            output = ARMTExpressionModelOutput(
                loss=loss, logits=output_logits, labels_reshaped=labels,
                labels_mask_reshaped=output_mask, cls_loss=cls_loss, other_loss=other_loss,
                deviation_loss=deviation_loss, multinomial_loss=multinomial_loss,
                armt_metrics=diagnostics or None)
            return output if return_dict is not False else (loss, output_logits)
        finally:
            self.reset_memory()
