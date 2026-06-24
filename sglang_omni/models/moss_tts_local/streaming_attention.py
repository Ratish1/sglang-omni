# SPDX-License-Identifier: Apache-2.0
"""SGLang attention patch for the MOSS-TTS Local streaming codec decoder."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any

import torch
from torch import nn

try:
    from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

logger = logging.getLogger(__name__)

_ORIGINAL_STREAMING_SDPA_ATTR = "_sglang_omni_original_forward_streaming_sdpa"


def moss_local_window_size(
    context: int | None, *, causal: bool = True
) -> tuple[int, int]:
    if context is None or not causal:
        return (-1, -1)
    return (max(int(context) - 1, 0), 0)


def moss_decoder_attention_modules(codec: nn.Module) -> list[Any]:
    modules_by_id: dict[int, Any] = {}
    decoder = getattr(codec, "decoder", ())
    for decoder_module in decoder:
        modules = decoder_module.modules() if hasattr(decoder_module, "modules") else ()
        for module in modules:
            if _is_moss_decoder_attention(module):
                modules_by_id.setdefault(id(module), module)
    return list(modules_by_id.values())


def patch_codec_streaming_attention(codec: nn.Module) -> int:
    """Route MOSS decoder streaming attention through SGLang FlashAttention."""
    if flash_attn_with_kvcache is None:
        logger.debug(
            "MOSS-TTS Local streaming SGLang attention unavailable; using codec SDPA"
        )
        return 0

    patched = 0
    for module in moss_decoder_attention_modules(codec):
        forward = getattr(module, "_forward_streaming_sdpa")
        if hasattr(module, _ORIGINAL_STREAMING_SDPA_ATTR):
            continue
        setattr(module, _ORIGINAL_STREAMING_SDPA_ATTR, forward)
        module._forward_streaming_sdpa = MethodType(_forward_streaming_sglang, module)
        patched += 1
    return patched


def streaming_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    pos_k: torch.Tensor,
    context: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run MOSS streaming local attention over already-built K/V cache+chunk tensors."""
    if flash_attn_with_kvcache is None:
        raise RuntimeError("SGLang flash_attn_with_kvcache is unavailable")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("MOSS streaming attention expects q/k/v rank-4 tensors")
    batch, heads, query_len, head_dim = q.shape
    if k.shape[:2] != (batch, heads) or v.shape[:2] != (batch, heads):
        raise ValueError(
            f"MOSS streaming q/k/v batch-head mismatch: q={tuple(q.shape)} "
            f"k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    key_len = int(k.shape[2])
    if v.shape[2] != key_len or v.shape[3] != head_dim:
        raise ValueError(
            f"MOSS streaming k/v shape mismatch: k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if key_len != int(context) + query_len:
        raise ValueError(
            f"MOSS streaming attention expects K=context+Q, got K={key_len}, "
            f"context={context}, Q={query_len}"
        )
    if pos_k.shape != (batch, key_len):
        raise ValueError(
            f"MOSS streaming pos_k shape mismatch: {tuple(pos_k.shape)} "
            f"for batch={batch}, K={key_len}"
        )

    cache_leftpad = _derive_cache_leftpad(pos_k, context=int(context))
    cache_seqlens = torch.full_like(cache_leftpad, key_len, dtype=torch.int32)
    out = flash_attn_with_kvcache(
        q.transpose(1, 2).contiguous(),
        k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(),
        cache_seqlens=cache_seqlens,
        cache_leftpad=cache_leftpad,
        causal=True,
        window_size=moss_local_window_size(int(context)),
        softmax_scale=softmax_scale,
    )
    return out.transpose(1, 2)


def _is_moss_decoder_attention(module: Any) -> bool:
    return (
        hasattr(module, "attention_implementation")
        and callable(getattr(module, "_forward_streaming_sdpa", None))
        and callable(getattr(module, "_update_streaming_cache", None))
    )


def _can_run_sglang_attention(x: torch.Tensor, module: Any) -> bool:
    if flash_attn_with_kvcache is None:
        return False
    if x.device.type != "cuda":
        return False
    check_dtype = module._get_backend_check_dtype(x)
    return check_dtype == torch.bfloat16


def _forward_streaming_sglang(self: Any, x: torch.Tensor, state: Any) -> torch.Tensor:
    original = getattr(self, _ORIGINAL_STREAMING_SDPA_ATTR)
    context = getattr(self, "context", None)
    if context is None or not _can_run_sglang_attention(x, self):
        return original(x, state)

    batch_size, chunk_length, _ = x.shape
    q, k_cur, v_cur = self._project_qkv(x)
    if self.rope is not None:
        q, k_cur = self.rope(q, k_cur, state.offset, time_before_heads=False)
    pos_q = state.offset.view(-1, 1) + torch.arange(
        chunk_length, device=x.device, dtype=torch.long
    ).view(1, -1)
    cached_k, cached_v, cached_pos = self._ensure_streaming_cache(
        state, batch_size, k_cur.device, k_cur.dtype
    )
    k_all, v_all, pos_k = self._build_streaming_kv(
        cached_k, cached_v, cached_pos, k_cur, v_cur, pos_q
    )
    out = streaming_local_attention(
        q,
        k_all,
        v_all,
        pos_k=pos_k,
        context=int(context),
    )
    out = out.transpose(1, 2).reshape(batch_size, chunk_length, self.embed_dim)

    self._update_streaming_cache(
        state, cached_k, cached_v, cached_pos, k_all, v_all, pos_k
    )
    state.offset[:] = torch.where(
        state.exec_mask, state.offset + chunk_length, state.offset
    )
    return out


def _derive_cache_leftpad(pos_k: torch.Tensor, *, context: int) -> torch.Tensor:
    if context < 0:
        raise ValueError(f"context must be non-negative, got {context}")
    valid_cache = pos_k[:, :context] >= 0
    history = valid_cache.sum(dim=1).to(torch.int32)
    return torch.full_like(history, context, dtype=torch.int32) - history


__all__ = [
    "moss_decoder_attention_modules",
    "moss_local_window_size",
    "patch_codec_streaming_attention",
    "streaming_local_attention",
]
