# Vendored from irodkin/ARMT-Gemma-3-1B-IT-8x1024-portable-encoder
# Revision: f7aa87af40da1495920e8554fde48da3c2c99a4d
# Source: https://huggingface.co/irodkin/ARMT-Gemma-3-1B-IT-8x1024-portable-encoder/blob/f7aa87af40da1495920e8554fde48da3c2c99a4d/modeling_armt.py
# Only the unchanged inner-loop layer and its utility functions are included.
import inspect
import warnings
from typing import Optional, Tuple, Callable
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

import torch
from torch.nn.functional import relu as r
import os

def dpfp(x, nu=1):
  x = torch.cat([r(x), r(-x)], dim=-1)
  x_rolled = torch.cat([x.roll(shifts=j, dims=-1)
           for j in range(1,nu+1)], dim=-1)
  x_repeat = torch.cat([x] * nu, dim=-1)
  return x_repeat * x_rolled

class DPFP:
    def __init__(self, nu):
        self.nu = nu
    
    def __call__(self, x):
        nu = self.nu
        x = torch.cat([r(x), r(-x)], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1,nu+1)], dim=-1)
        x_repeat = torch.cat([x] * nu, dim=-1)
        return x_repeat * x_rolled
def attn_mask_to_4d(attn_mask, upper, query_len):
    if attn_mask is None:
        return None
    seg_len = attn_mask.size(-1)
    if upper:
        tri = torch.triu(torch.ones(query_len, seg_len, dtype=attn_mask.dtype, device=attn_mask.device))
    else:
        tri = torch.tril(torch.ones(query_len, seg_len, dtype=attn_mask.dtype, device=attn_mask.device))

    mask = torch.einsum('bj,ij->bij', attn_mask, tri)
    mask = mask.unsqueeze(1)
    return mask

def invert_attn_mask(attn_mask, dtype):
        if os.environ.get("NOT_INVERT_ATTN_MASK"):
            return attn_mask
        min_dtype = torch.finfo(dtype).min
        # Use the same dtype as attn_mask to avoid dtype conversion
        one = torch.tensor(1.0, dtype=torch.long, device=attn_mask.device)
        new_mask = (one - attn_mask.long()) * min_dtype
        return new_mask

def reverse_invert_attn_mask(mask: torch.Tensor) -> torch.Tensor:
    if os.environ.get("NOT_INVERT_ATTN_MASK"):
        return mask
    # Boolean masks (used by recent transformers SDPA/eager paths) mark allowed positions with True.
    if mask.dtype == torch.bool:
        return mask.long()
    # Compare in floating point. Casting finfo.min to an integer is undefined behaviour and
    # silently unmasks future tokens on some backends (for example Apple MPS).
    return (mask > -1).long()

def attn_mask_to_2d(mask: torch.Tensor) -> torch.Tensor:
    mask = reverse_invert_attn_mask(mask)
    mask = torch.any(mask, dim=-2)
    mask = torch.any(mask, dim=1)
    return mask.long()

def is_empty_past_key_values(past_key_values: Optional[DynamicCache], layer_idx: int) -> bool:
    if past_key_values is None:
        return True
    if len(past_key_values.layers) == 0:
        return True
    if len(past_key_values.layers) <= layer_idx:
        return True
    if past_key_values.layers[layer_idx].keys is None:
        return True
    return False

def segment_tensor(t, start_idx: int, end_idx: int, seq_len: int):
    # Handle tuples of tensors (like position_embeddings which are (cos, sin))
    if isinstance(t, tuple):
        return tuple(segment_tensor(item, start_idx, end_idx, seq_len) for item in t)
    if not isinstance(t, torch.Tensor):
        return t
    # Handle 1D tensors (like cache_position)
    if t.dim() == 1 and t.size(0) == seq_len:
        return t[start_idx:end_idx]
    # common cases: (bsz, seq_len, ...), (bsz, seq_len), (seq_len, ...)
    if t.dim() >= 2 and t.size(1) == seq_len:
        return t[:, start_idx:end_idx, ...]
    return t

class InnerLoopAssociativeLayerWrapper(nn.Module):
    """
    A per-layer wrapper that performs associative read/write within the layer by
    splitting the incoming full sequence into fixed-size segments on the fly.

    Unlike the outer-loop design (which segments inputs before the model), this
    module receives the full, unsplit hidden sequence and internally iterates
    over segments:
      1) Optional associative READ is applied to the segment's hidden states
         based on the current associative memory (W_mem, z).
      2) Memory tokens are appended to the segment and the underlying transformer
         layer is executed only on this augmented segment.
      3) The resulting memory token outputs are used to WRITE/update the
         associative memory.
      4) The transformed real-token outputs replace the corresponding slice in
         the layer output for the full sequence.

    This preserves identical behavior w.r.t. memory math while avoiding any
    outer recurrent wrapper.
    """

    def __init__(
        self,
        layer: nn.Module,
        d_model: int,
        num_mem_tokens: int,
        d_mem: int,
        segment_size: int,
        n_heads: int = 1,
        correction: bool = True,
        use_denom: bool = True,
        gating: bool = False,
        use_sink: bool = False,
        sliding_window: bool = False,
        attn_implementation: str = "flash_attention_2",
        is_causal: bool = True,
        memory_dtype: Optional[torch.dtype] = None,
        get_memory_fn: Optional[Callable[[], torch.Tensor]] = None,
        get_sink_fn: Optional[Callable[[], Optional[torch.Tensor]]] = None,
        rotary_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        read_prev_states_fn: Optional[Callable[[int, int, torch.device, torch.dtype], Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None,
        write_states_fn: Optional[Callable[[int, torch.Tensor, Optional[torch.Tensor]], None]] = None,
        info: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.info = info
        self.layer = layer
        self.d_model = d_model
        self.num_mem_tokens = int(num_mem_tokens or 0)
        self.d_mem = d_mem
        self.segment_size = int(segment_size)
        self.n_heads = n_heads
        self.gating = gating
        self.use_denom = use_denom
        self.correction = correction
        self.use_sink = bool(use_sink)
        self.sliding_window_enabled = bool(sliding_window)
        self.attn_implementation = attn_implementation
        self.is_causal = bool(is_causal)

        # Parameters accepted by the wrapped block; None if it takes **kwargs (accepts anything)
        try:
            sig_params = inspect.signature(self.layer.forward).parameters.values()
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_params):
                self._layer_params = None
            else:
                self._layer_params = {p.name for p in sig_params}
        except (TypeError, ValueError):
            self._layer_params = None

        # DPFP feature map dimensions
        nu = 3
        self.d_key = 2 * nu * d_mem

        assert self.d_mem % n_heads == 0 and self.d_model % n_heads == 0

        # Match the dtype to the wrapped layer
        layer_dtype = next(self.layer.parameters()).dtype
        self.memory_dtype = memory_dtype if memory_dtype is not None else layer_dtype

        # Readout/query/key/value projections for associative memory
        self.W_mq = nn.Linear(d_model, d_mem, bias=False, dtype=self.memory_dtype)
        self.W_mk = nn.Linear(d_model, d_mem, bias=False, dtype=self.memory_dtype)
        self.W_mv = nn.Linear(d_model, d_model, bias=False, dtype=self.memory_dtype)
        if gating:
            self.W_mb = nn.Linear(d_model, d_model, dtype=self.memory_dtype)
        else:
            self.W_mb = nn.Linear(d_model, n_heads, dtype=self.memory_dtype)
        torch.nn.init.zeros_(self.W_mv.weight)

        self.phi = DPFP(nu)

        # Runtime flags/counters
        self.generate_mode = False
        self.seg_num = 0

        # Lightweight accessors to shared trainable memory tensors owned by the top-level model.
        # These are callables, not Modules/Parameters stored as attributes, to avoid submodule cycles.
        self._get_memory = get_memory_fn
        self._get_sink = get_sink_fn
        self._rotary_fn = rotary_fn
        self._read_prev_states = read_prev_states_fn
        self._write_states = write_states_fn

        self.memory_state = None

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped layer if not found in wrapper."""
        # First let nn.Module try to find the attribute (in _modules, _parameters, _buffers)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # Then delegate to the wrapped layer
        try:
            layer = super().__getattr__('layer')
            return getattr(layer, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ----- helpers for heads reshaping -----
    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        x = x.reshape(bsz, seq_len, self.n_heads, d_model // self.n_heads)
        x = x.permute(0, 2, 1, 3)
        return x

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, seq_len, d_head = x.shape
        x = x.permute(0, 2, 1, 3).reshape(bsz, seq_len, n_heads * d_head)
        return x

    # ----- associative read -----
    def associate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("associate() is unused in inner-loop; uses local memory helpers instead")

    # ----- associative write -----
    def update_mem(self, mem_tokens: torch.Tensor) -> None:
        raise NotImplementedError("update_mem() is unused in inner-loop; uses local memory helpers instead")

    # ----- memory state management -----
    def zero_mem(self) -> None:
        self.memory_state = None

    def detach_mem(self) -> None:
        self.memory_state = (self.memory_state[0].detach(), self.memory_state[1].detach()) if self.memory_state is not None else None

    def freeze_mem(self) -> None:
        self.W_mb.weight.requires_grad = False
        self.W_mb.bias.requires_grad = False
        self.W_mq.weight.requires_grad = False
        self.W_mk.weight.requires_grad = False
        self.W_mv.weight.requires_grad = False

    # ----- utilities -----
    def _get_segment_positions(
        self, position_ids: Optional[torch.LongTensor], start: int, end: int, device: torch.device
    ) -> torch.LongTensor:
        # Match old SW behavior: positions are local to each segment.
        seg_len = end - start
        return torch.arange(0, seg_len, device=device).long().unsqueeze(0)


    def pad_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype):
        if self.num_mem_tokens in {0, None} and not self.use_sink:
            return attention_mask
        shape = list(attention_mask.shape)
        if len(shape) == 4:
            shape[-1] += self.num_mem_tokens + int(self.use_sink)
            shape[-2] += self.num_mem_tokens + int(self.use_sink)
            mask = torch.ones(*shape, dtype=dtype).to(attention_mask.device)
            mask[..., int(self.use_sink):-self.num_mem_tokens, int(self.use_sink):-self.num_mem_tokens] = attention_mask
            # Memory tokens read every real token of the segment, but not padding: a key column is
            # valid if at least one real query may attend to it.
            key_valid = attention_mask.bool().any(dim=-2, keepdim=True)
            mask[..., -self.num_mem_tokens:, int(self.use_sink):-self.num_mem_tokens] = key_valid.to(dtype)
            if self.use_sink:
                mask[..., 0, 1:] = 0
            mask[..., :-self.num_mem_tokens, -self.num_mem_tokens:] = 0
            # Make memory→memory attention causal (matching SW behavior)
            mem_causal = torch.tril(torch.ones(self.num_mem_tokens, self.num_mem_tokens, dtype=dtype, device=attention_mask.device))
            mask[..., -self.num_mem_tokens:, -self.num_mem_tokens:] = mem_causal
        elif len(shape) == 2:
            shape[-1] += self.num_mem_tokens + int(self.use_sink)
            mask = torch.ones(*shape, dtype=dtype).to(attention_mask.device)
            mask[..., int(self.use_sink):-self.num_mem_tokens] = attention_mask
        else:
            raise ValueError("Attention mask must be 2D or 4D")
        return mask.to(dtype)

    def pad_prev_seg_attn_mask(self, prev_seg_attn_mask: torch.Tensor, dtype: torch.dtype):
        if self.num_mem_tokens in {0, None} and not self.use_sink:
            return prev_seg_attn_mask
        shape = list(prev_seg_attn_mask.shape)
        if len(shape) == 4:
            shape[-2] += self.num_mem_tokens + int(self.use_sink)
            mask = torch.ones(*shape, dtype=dtype, device=prev_seg_attn_mask.device)
            mask[..., int(self.use_sink):-self.num_mem_tokens, :] = prev_seg_attn_mask
            if self.use_sink:
                mask[..., 0, :] = 0
            return mask.to(dtype)
        return prev_seg_attn_mask.to(dtype)


    def _get_memory_tokens(self, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._get_memory is None or self.num_mem_tokens == 0:
            return None, None
        memory = self._get_memory()
        sink = self._get_sink() if self.use_sink and self._get_sink is not None else None
        mem = memory.unsqueeze(0).expand(batch_size, -1, -1)
        if sink is not None:
            sink = sink.unsqueeze(0).expand(batch_size, -1, -1)
        return mem, sink

    # ----- helpers operating on provided memory tensors (no buffers) -----
    def _alloc_initial_mem(self, device: torch.device, dtype: Optional[torch.dtype] = None):
        dtype = self.memory_dtype if dtype is None else dtype
        W_mem = torch.zeros(
            1,
            self.n_heads,
            self.d_key // self.n_heads,
            self.d_model // self.n_heads,
            device=device,
            dtype=dtype,
        )
        z = torch.zeros(1, self.n_heads, self.d_key // self.n_heads, device=device, dtype=dtype) if self.use_denom else None
        return W_mem, z

    def _associate_with_mem(self, hidden_states: torch.Tensor, W_mem: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.memory_dtype)
        q = self._to_heads(self.W_mq(hidden_states))
        mq = self.phi(q)
        mq = F.normalize(mq, dim=-1, p=2.0)
        num = torch.einsum("ihjk,ihkt->ihjt", mq, W_mem)
        if self.use_denom and z is not None:
            denom = torch.einsum("ihk,ihjk->ihj", z, mq)[..., None] + 1e-5
            hs = num / denom
        else:
            hs = num
        return self._from_heads(hs).to(original_dtype)

    def _update_mem_with_mem(
        self,
        mem_tokens: torch.Tensor,
        W_mem: torch.Tensor,
        z: Optional[torch.Tensor],
        first_seg: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        mem_tokens = mem_tokens.to(self.memory_dtype)
        k = self._to_heads(self.W_mk(mem_tokens))
        mk = self.phi(k)
        mk = F.normalize(mk, dim=-1, p=2.0)

        new_mv = self._to_heads(self.W_mv(mem_tokens))
        if not first_seg:
            num = torch.einsum("ihjk,ihkt->ihjt", mk, W_mem)
            if self.use_denom and z is not None:
                denom = torch.einsum("ihj,ihkj->ihk", z, mk)[..., None] + 1e-5
                prev_mv = num / denom
                if self.correction:
                    new_info_coef = (
                        1 - denom / (torch.linalg.norm(mk, dim=-1) ** 2)[..., None]
                    )
                    new_info_coef = torch.clip(new_info_coef, 0, 1).detach()
                else:
                    new_info_coef = 1
            else:
                prev_mv = num
                new_info_coef = 1
        else:
            prev_mv = torch.zeros_like(new_mv, device=new_mv.device)
            new_info_coef = 1

        mv = new_mv - prev_mv
        mb = self._to_heads(torch.sigmoid(self.W_mb(mem_tokens)))
        einop = f"ihjk,ihjt,ihj{'t' if self.gating else 'x'}->ihkt"
        associations = torch.einsum(einop, mk, mv, mb)
        W_mem = W_mem + associations
        if self.use_denom and z is not None:
            z = z + (new_info_coef * mk).sum(dim=-2)
        return W_mem, z, False

    
    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Convert positional args of the wrapped HF block into keyword args by
        introspecting the block's forward signature. This prevents accidental
        misplacement (e.g., a cache object being treated as attention_mask).
        """
        # Map positional args to their parameter names (excluding self & hidden_states)
        try:
            sig = inspect.signature(self.layer.forward)
            params = list(sig.parameters.values())
            # Drop the first param which should be 'self' for bound method
            param_names = [p.name for p in params[1:]]
            # If the next parameter is hidden_states, drop it as well
            if len(param_names) > 0 and param_names[0] in {"hidden_states", "x"}:
                param_names = param_names[1:]
        except Exception:
            param_names = []

        for idx, arg in enumerate(args):
            if idx >= len(param_names):
                break
            name = param_names[idx]
            if name not in kwargs:
                kwargs[name] = arg

        # Normalize cache kwarg name to 'past_key_values'
        if "layer_past" in kwargs and "past_key_values" not in kwargs:
            layer_past = kwargs.pop("layer_past")
            try:
                if isinstance(layer_past, DynamicCache):
                    kwargs["past_key_values"] = layer_past
                else:
                    kwargs["past_key_values"] = DynamicCache.from_legacy_cache(layer_past)
            except Exception:
                kwargs["past_key_values"] = layer_past

        # Extract attention mask (avoid passing both positional & kwarg duplicates)
        attention_mask = kwargs.pop("attention_mask", None)

        return self.forward_horizontal(hidden_states, attention_mask, **kwargs)
    
    # ----- main forward (inner-loop segmentation) -----
    def forward_horizontal(self, hidden_states: torch.Tensor, attention_mask=None, *args, **kwargs):
        assert not self.generate_mode, "Generate mode is not supported for horizontal forward"
        if self.sliding_window_enabled:
            assert attention_mask is None or attention_mask.dim() == 4, "Attention mask must be 4D when sliding window is enabled"
        using_cache = not is_empty_past_key_values(kwargs.get("past_key_values"), self.info['layer'])
        assert not using_cache or (kwargs.get('past_attn_mask') is not None and kwargs.get('past_attn_mask').shape[-1] == self.segment_size), "When using cache, past_attn_mask must be provided and have the same length as the segment size"

        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        bsz, seq_len, _ = hidden_states.shape

        if attention_mask is None:
            # The base model may drop an all-ones mask (e.g. SDPA). Rebuild it with the base model's
            # causality: lower-triangular for decoders, full for bidirectional encoders.
            attention_mask = torch.ones(bsz, seq_len, device=hidden_states.device, dtype=hidden_states.dtype)
            if self.sliding_window_enabled or self.attn_implementation != "flash_attention_2":
                if self.is_causal:
                    attention_mask = attn_mask_to_4d(attention_mask, upper=False, query_len=seq_len)
                else:
                    attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
                attention_mask = invert_attn_mask(attention_mask, hidden_states.dtype)
        elif attention_mask.dim() == 4 and attention_mask.size(-2) == 1 and seq_len > 1:
            # Broadcastable key-padding mask (bsz, 1, 1, seq_len), e.g. BERT's eager extended mask:
            # materialise the query dimension so that it can be sliced per segment.
            attention_mask = attention_mask.expand(*attention_mask.shape[:-2], seq_len, attention_mask.size(-1))
        out_full = []

        # Initialize associative memory from persisted state if available
        if self.memory_state is not None:
            W_mem, z = self.memory_state
            W_mem = W_mem.to(device=hidden_states.device, dtype=self.memory_dtype)
            if z is not None:
                z = z.to(device=hidden_states.device, dtype=self.memory_dtype)
            first_seg = False
        else:
            W_mem, z = self._alloc_initial_mem(hidden_states.device, self.memory_dtype)
            first_seg = True


        # Always use provided cache object if present, even if currently empty,
        # so upstream callers can observe in-place mutations across segments.
        provided_cache = kwargs.get("past_key_values")
        past_key_values = provided_cache if provided_cache is not None else DynamicCache()
        past_attn_mask = kwargs.get('past_attn_mask') if using_cache else None
        present_kv = None

        # helper to segment arbitrary tensor-like by time dim
        
        seg_num = 0
        for start in range(0, seq_len, self.segment_size+self.num_mem_tokens+int(self.use_sink)):
            real_start = start+int(self.use_sink)
            real_end = min(real_start + self.segment_size, seq_len-self.num_mem_tokens)
            end = real_end+self.num_mem_tokens
            seg_aug = hidden_states[:, start:end, :]
            seg_len = real_end - real_start

            if attention_mask.dim() == 4:
                attn_mask = attention_mask[:, :, real_start:real_end, real_start:real_end]
            else:
                attn_mask = attention_mask[:, real_start:real_end]

            # print("attn_mask", attn_mask[0][0])

            # Check if this is the last segment and we're in generate mode
            is_last_segment = (end >= seq_len)


            if not first_seg:
                assoc = self._associate_with_mem(seg_aug, W_mem, z)
                seg_aug = assoc + seg_aug

            # Build attention mask for this augmented segment
            seg_aug_len = seg_aug.size(1)
            
            if self.sliding_window_enabled:
                # print(attn_mask.shape, "attn_mask", "*"*100)
                # print(base_cur4d.shape, "base_cur4d", "*"*100)
                base_cur4d = reverse_invert_attn_mask(attn_mask)
                seg_mask = self.pad_attention_mask(base_cur4d, dtype=seg_aug.dtype)
                seg_mask = invert_attn_mask(seg_mask, seg_aug.dtype)

                if past_attn_mask is not None:

                    base_past4d = attn_mask_to_4d(attn_mask_to_2d(past_attn_mask), upper=True, query_len=seg_len)
                    base_past4d = self.pad_prev_seg_attn_mask(base_past4d, dtype=seg_aug.dtype)
                    base_past4d = invert_attn_mask(base_past4d, seg_aug.dtype)

                    # print(base_past4d.shape, "base_past4d", "*"*100)
                    # print(seg_mask.shape, "seg_mask", "*"*100)
                    seg_mask = torch.cat([base_past4d, seg_mask], dim=-1)
                if os.environ.get("ARMT_DEBUG_SW"):
                    print(f"[H-SEG] L{self.info['layer']} seg_len={seg_len} seg_aug_len={seg_aug_len} mask={tuple(seg_mask.shape)}")
            else:
                if attn_mask.dim() == 4:
                    if self.attn_implementation == "flash_attention_2":
                        attn_mask = attn_mask_to_2d(attn_mask)
                        seg_mask = self.pad_attention_mask(attn_mask, dtype=seg_aug.dtype)
                    else:
                        base_cur4d = reverse_invert_attn_mask(attn_mask)
                        seg_mask = self.pad_attention_mask(base_cur4d, dtype=seg_aug.dtype)
                        seg_mask = invert_attn_mask(seg_mask, seg_aug.dtype)
                else:
                    seg_mask = self.pad_attention_mask(attn_mask, dtype=seg_aug.dtype)
            # print("seg_mask", reverse_invert_attn_mask(seg_mask)[0][0])
            # print("seg_mask", seg_mask.shape)
            seg_pos_ids = self._get_segment_positions(kwargs.get("position_ids", None), start, end, seg_aug.device)

            # Segment incoming args/kwargs by time where applicable
            seg_args = tuple(segment_tensor(a, start, end, seq_len) if isinstance(a, torch.Tensor) else a for a in args)
            seg_kwargs = {k: segment_tensor(v, start, end, seq_len) for k, v in kwargs.items()}


            
            # Override with our computed fields
            seg_kwargs["attention_mask"] = seg_mask.to(seg_aug.dtype)
            if seg_pos_ids is not None:
                seg_kwargs["position_ids"] = seg_pos_ids
            seg_kwargs["use_cache"] = self.sliding_window_enabled
            
            if self.sliding_window_enabled:
                seg_kwargs["past_key_values"] = past_key_values
            else:
                # When not using sliding window, clear cache-related params to avoid shape mismatches
                seg_kwargs["past_key_values"] = None
                seg_kwargs["use_cache"] = False

            if self._rotary_fn is not None and seg_pos_ids is not None:
                cos, sin = self._rotary_fn(seg_aug, seg_pos_ids)
                seg_kwargs["position_embeddings"] = (cos, sin)

            if self._layer_params is not None:
                # Blocks without **kwargs (e.g. BertLayer in transformers 4.x) reject extra arguments
                seg_kwargs = {k: v for k, v in seg_kwargs.items() if k in self._layer_params}

            layer_out = self.layer(seg_aug, *seg_args, **seg_kwargs)
            if self.sliding_window_enabled:
                assert past_key_values is not None, "Past key values object must be provided"
                # In-place update & trim so outer references observe changes
                if os.environ.get("ARMT_DEBUG_SW"):
                    k = past_key_values.layers[self.info['layer']].keys
                    v = past_key_values.layers[self.info['layer']].values
                    print(f"[H-CACHE:pre] L{self.info['layer']} K={tuple(k.shape) if k is not None else None} V={tuple(v.shape) if v is not None else None}")
                past_key_values = self.update_past_key_values_sw(past_key_values, self.segment_size)
                if os.environ.get("ARMT_DEBUG_SW"):
                    k = past_key_values.layers[self.info['layer']].keys
                    v = past_key_values.layers[self.info['layer']].values
                    print(f"[H-CACHE:post] L{self.info['layer']} K={tuple(k.shape) if k is not None else None} V={tuple(v.shape) if v is not None else None}")
            if isinstance(layer_out, tuple):
                seg_out = layer_out[0]
            else:
                seg_out = layer_out

            seg_mem_out = seg_out[:, -self.num_mem_tokens:, :]
            W_mem, z, first_seg = self._update_mem_with_mem(
                seg_mem_out, W_mem, z, first_seg
            )
            first_seg = False

            out_full.append(seg_out)

            past_attn_mask = attn_mask
            seg_num += 1

        merged = torch.cat(out_full, dim=1)

        # Persist updated memory state for vertical mode to reuse across segments
        self.memory_state = (W_mem, z)

        if isinstance(layer_out, tuple):
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            if len(layer_out) == 1:
                return (merged,)
            elif len(layer_out) == 2:
                warnings.warn(f"{YELLOW}Last attention was not tested for horizontal forward{RESET}")
                return (merged, None)
            elif len(layer_out) == 3:
                warnings.warn(f"{YELLOW}Last attention and kv states were not tested for horizontal forward{RESET}")
                return (merged, None, present_kv)
            else:
                raise ValueError(f"Expected 1, 2 or 3 elements in layer output, got {len(layer_out)}")
        else:
            return merged

    def update_past_key_values_sw(self, past_key_values, window_size):
        """
        Update past key values for sliding window attention.
        This keeps only the most recent tokens within the window size.
        """
        if is_empty_past_key_values(past_key_values, self.info['layer']):
            return None
            
        # Convert to legacy cache format for easier manipulation
        if hasattr(past_key_values, 'to_legacy_cache'):
            legacy = past_key_values.to_legacy_cache()
        else:
            legacy = past_key_values
        
        # Keep only the most recent real tokens within the window size
        k, v = legacy[self.info['layer']]
        k = k[..., -window_size-self.num_mem_tokens:-self.num_mem_tokens, :]
        v = v[..., -window_size-self.num_mem_tokens:-self.num_mem_tokens, :]
        
        past_key_values.layers[self.info['layer']].keys = k
        past_key_values.layers[self.info['layer']].values = v
        return past_key_values



