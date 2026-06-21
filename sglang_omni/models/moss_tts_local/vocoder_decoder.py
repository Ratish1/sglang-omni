# SPDX-License-Identifier: Apache-2.0
"""Patched MOSS-TTS Local vocoder decoder stages.

This module mirrors the remote MOSS-Audio-Tokenizer-v2 decoder stage mechanics
without changing the scheduler, codec embeddings, or waveform projection code.
It is intentionally MOSS-specific: the interception point is the decoder's
projected transformer stages, not the codec embeddings or waveform projection.
"""

from __future__ import annotations

import functools
import importlib
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _load_sglang_flash_attn_varlen_func() -> Any | None:
    try:
        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
    except Exception as exc:
        logger.debug("SGLang flash attention unavailable for MOSS vocoder: %s", exc)
        return None
    return flash_attn_varlen_func


def _module_list(value: Any) -> list[nn.Module]:
    if isinstance(value, nn.ModuleList):
        return list(value)
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, nn.Module) for item in value
    ):
        return list(value)
    return []


def _pack_padded_sequence(
    x: torch.Tensor,
    input_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, max_seqlen, _ = x.shape
    positions = torch.arange(max_seqlen, device=x.device, dtype=torch.long)
    valid_mask = positions.view(1, max_seqlen) < input_lengths.view(batch_size, 1)
    packed_x = x[valid_mask]
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
    cu_seqlens[1:] = torch.cumsum(input_lengths.to(torch.int32), dim=0)
    position_ids = positions.view(1, max_seqlen).expand(batch_size, -1)[valid_mask]
    return packed_x, valid_mask, cu_seqlens, position_ids


def _pack_single_unpadded_sequence(
    x: torch.Tensor,
    metadata_cache: "_SingleUnpaddedMetadataCache | None" = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, max_seqlen, _ = x.shape
    packed_x = x.reshape(max_seqlen, x.shape[-1])
    if metadata_cache is None:
        cu_seqlens = torch.tensor([0, max_seqlen], dtype=torch.int32, device=x.device)
        position_ids = torch.arange(max_seqlen, device=x.device, dtype=torch.long)
    else:
        cu_seqlens, position_ids = metadata_cache.get(
            device=x.device,
            max_seqlen=max_seqlen,
        )
    return packed_x, cu_seqlens, position_ids


class _SingleUnpaddedMetadataCache:
    def __init__(self) -> None:
        self._items: dict[
            tuple[str, int | None, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def get(
        self,
        *,
        device: torch.device,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_seqlen <= 0:
            raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")
        key = (device.type, device.index, max_seqlen)
        cached = self._items.get(key)
        if cached is not None:
            return cached
        cu_seqlens = torch.tensor([0, max_seqlen], dtype=torch.int32, device=device)
        position_ids = torch.arange(max_seqlen, device=device, dtype=torch.long)
        self._items[key] = (cu_seqlens, position_ids)
        return cu_seqlens, position_ids


def _unpack_packed_sequence(
    packed_x: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
) -> torch.Tensor:
    x = packed_x.new_zeros(batch_size, max_seqlen, packed_x.shape[-1])
    x[valid_mask] = packed_x
    return x


def _unpack_single_unpadded_sequence(
    packed_x: torch.Tensor,
) -> torch.Tensor:
    return packed_x.reshape(1, packed_x.shape[0], packed_x.shape[-1])


class _MossPackedRopeCache:
    def __init__(self, *, max_period: float) -> None:
        self.max_period = float(max_period)
        self._device: torch.device | None = None
        self._head_dim = 0
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def get(
        self,
        *,
        device: torch.device,
        head_dim: int,
        max_positions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {max_positions}")
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        if (
            self._cos is not None
            and self._sin is not None
            and self._device == device
            and self._head_dim == head_dim
            and self._cos.shape[0] >= max_positions
        ):
            return self._cos[:max_positions], self._sin[:max_positions]

        half_dim = head_dim // 2
        ds = torch.arange(half_dim, device=device, dtype=torch.float32)
        freqs = torch.exp(ds * (-math.log(self.max_period) * 2 / head_dim))
        positions = torch.arange(
            max_positions, device=device, dtype=torch.float32
        ).view(-1, 1)
        phase = positions * freqs.view(1, -1)
        self._device = device
        self._head_dim = head_dim
        self._cos = torch.cos(phase)
        self._sin = torch.sin(phase)
        return self._cos, self._sin


def _apply_cached_packed_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    max_positions: int,
    cache: _MossPackedRopeCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k.shape != q.shape:
        raise ValueError(
            f"Expected k.shape == q.shape, got k={tuple(k.shape)} q={tuple(q.shape)}"
        )
    if q.dim() != 3:
        raise ValueError(
            f"packed RoPE expects [tokens, heads, dim], got {tuple(q.shape)}"
        )
    _, _, head_dim = q.shape
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    cos_cache, sin_cache = cache.get(
        device=q.device,
        head_dim=head_dim,
        max_positions=max_positions,
    )
    if position_ids.numel() == max_positions:
        cos = cos_cache.view(max_positions, 1, head_dim // 2)
        sin = sin_cache.view(max_positions, 1, head_dim // 2)
    else:
        cos = cos_cache.index_select(0, position_ids).view(
            position_ids.numel(), 1, head_dim // 2
        )
        sin = sin_cache.index_select(0, position_ids).view(
            position_ids.numel(), 1, head_dim // 2
        )

    dims = q.shape[:-1]
    q_pair = q.view(*dims, head_dim // 2, 2)
    k_pair = k.view(*dims, head_dim // 2, 2)
    qr, qi = q_pair[..., 0].float(), q_pair[..., 1].float()
    kr, ki = k_pair[..., 0].float(), k_pair[..., 1].float()

    qor = qr * cos - qi * sin
    qoi = qr * sin + qi * cos
    kor = kr * cos - ki * sin
    koi = kr * sin + ki * cos

    q_out = torch.stack([qor.to(q.dtype), qoi.to(q.dtype)], dim=-1).view(
        *dims, head_dim
    )
    k_out = torch.stack([kor.to(k.dtype), koi.to(k.dtype)], dim=-1).view(
        *dims, head_dim
    )
    return q_out, k_out


class MossTTSLocalAttention(nn.Module):
    """MOSS local-causal self attention over dense or packed decoder frames."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.in_proj = getattr(source, "in_proj", None)
        self.out_proj = getattr(source, "out_proj", None)
        if self.in_proj is None or self.out_proj is None:
            raise ValueError("MOSS vocoder attention requires in_proj and out_proj")
        self.embed_dim = int(getattr(source, "embed_dim"))
        self.num_heads = int(getattr(source, "num_heads"))
        self.head_dim = int(
            getattr(source, "head_dim", self.embed_dim // self.num_heads)
        )
        if self.embed_dim != self.num_heads * self.head_dim:
            raise ValueError(
                f"invalid attention shape: embed_dim={self.embed_dim}, "
                f"num_heads={self.num_heads}, head_dim={self.head_dim}"
            )
        self.causal = bool(getattr(source, "causal", True))
        self.context = getattr(source, "context", None)
        self.rope = getattr(source, "rope", None)
        self._sglang_flash_attn_varlen_func = _load_sglang_flash_attn_varlen_func()
        max_period = getattr(self.rope, "max_period", 10000.0)
        self._packed_rope_cache = _MossPackedRopeCache(max_period=max_period)

    def resolve_attention_implementation(
        self,
        x: torch.Tensor,
        *,
        is_streaming: bool = False,
    ) -> str:
        backend = self.source.resolve_attention_implementation(
            x,
            is_streaming=is_streaming,
        )
        if backend == "flash_attention_2" and self._supports_sglang_flash_attention(x):
            return backend
        if getattr(
            self.source, "attention_implementation", None
        ) == "flash_attention_2" and self._supports_sglang_flash_attention(x):
            return "flash_attention_2"
        return backend

    def _supports_sglang_flash_attention(self, x: torch.Tensor) -> bool:
        if self._sglang_flash_attn_varlen_func is None or x.device.type != "cuda":
            return False
        return self._backend_check_dtype(x) == torch.bfloat16

    def _backend_check_dtype(self, x: torch.Tensor) -> torch.dtype:
        get_backend_check_dtype = getattr(self.source, "_get_backend_check_dtype", None)
        if callable(get_backend_check_dtype):
            return get_backend_check_dtype(x)
        if x.device.type != "cuda":
            return x.dtype
        try:
            autocast_enabled = torch.is_autocast_enabled("cuda")
        except TypeError:
            autocast_enabled = torch.is_autocast_enabled()
        if not autocast_enabled:
            return x.dtype
        try:
            return torch.get_autocast_dtype("cuda")
        except TypeError:
            return torch.get_autocast_gpu_dtype()

    def forward(
        self,
        query: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        position_ids: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        streaming_state = getattr(self.source, "_streaming_state", None)
        backend = self.resolve_attention_implementation(
            query,
            is_streaming=streaming_state is not None,
        )
        if streaming_state is not None:
            if query.dim() != 3:
                raise ValueError(
                    f"streaming attention expects a 3D tensor, got {tuple(query.shape)}"
                )
            out = self._forward_streaming_sdpa(query, streaming_state)
            return self.out_proj(out)
        if backend == "flash_attention_2":
            if query.dim() != 2:
                raise ValueError(
                    "packed flash attention expects a 2D tensor, "
                    f"got {tuple(query.shape)}"
                )
            if cu_seqlens is None or max_seqlen is None or position_ids is None:
                raise ValueError(
                    "packed flash attention requires cu_seqlens, max_seqlen, "
                    "and position_ids"
                )
            return self._forward_packed_flash(
                query,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_ids=position_ids,
            )
        if query.dim() != 3:
            raise ValueError(
                f"dense attention expects a 3D tensor, got {tuple(query.shape)}"
            )
        if input_lengths is None:
            raise ValueError("dense attention requires input_lengths")
        return self.source(
            query,
            input_lengths=input_lengths,
        )

    def _forward_streaming_sdpa(
        self, query: torch.Tensor, streaming_state: Any
    ) -> torch.Tensor:
        return self.source._forward_streaming_sdpa(query, streaming_state)

    def _project_qkv(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.in_proj(x)
        if x.dim() == 3:
            projected = projected.reshape(
                x.shape[0], x.shape[1], 3, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            return projected[0], projected[1], projected[2]
        if x.dim() == 2:
            projected = projected.view(x.shape[0], 3, self.num_heads, self.head_dim)
            return projected[:, 0], projected[:, 1], projected[:, 2]
        raise ValueError(f"expected a 2D or 3D tensor, got {tuple(x.shape)}")

    def _apply_packed_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        max_positions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        return _apply_cached_packed_rope(
            q,
            k,
            position_ids,
            max_positions=max_positions,
            cache=self._packed_rope_cache,
        )

    def _forward_packed_flash(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self._project_qkv(x)
        q, k = self._apply_packed_rope(
            q,
            k,
            position_ids,
            max_positions=max_seqlen,
        )
        out = self._run_flash_attention(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
        )
        return self.out_proj(out.reshape(x.shape[0], self.embed_dim))

    def _run_flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        flash_attn_varlen_func = self._sglang_flash_attn_varlen_func
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "SGLang flash attention is not available for MOSS vocoder"
            )
        window_size = (
            (int(self.context), 0)
            if self.context is not None and self.causal
            else (-1, -1)
        )
        return flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=self.causal,
            window_size=window_size,
        )


class MossTTSLocalTransformerLayer(nn.Module):
    """One MOSS vocoder transformer layer."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.norm1 = getattr(source, "norm1", None)
        self.norm2 = getattr(source, "norm2", None)
        self.layer_scale_1 = getattr(source, "layer_scale_1", None)
        self.layer_scale_2 = getattr(source, "layer_scale_2", None)
        self.ffn = getattr(source, "ffn", None)
        self.self_attn = MossTTSLocalAttention(getattr(source, "self_attn"))
        if self.norm1 is None or self.norm2 is None:
            raise ValueError("MOSS vocoder transformer layer requires norm1/norm2")
        if self.layer_scale_1 is None or self.layer_scale_2 is None:
            raise ValueError("MOSS vocoder transformer layer requires layer scales")
        if not isinstance(self.ffn, nn.Sequential) or len(self.ffn) < 3:
            raise ValueError(
                "MOSS vocoder transformer layer requires Linear-GELU-Linear FFN"
            )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = residual.to(x) + self.layer_scale_1(self.self_attn(x, **kwargs))
        residual = x
        x = self.norm2(x)
        x = residual.to(x) + self.layer_scale_2(self.ffn(x))
        return x


class MossTTSLocalTransformer(nn.Module):
    """MOSS vocoder transformer body."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.layers = nn.ModuleList(
            [
                MossTTSLocalTransformerLayer(layer)
                for layer in _module_list(source.layers)
            ]
        )
        self.positional_embedding = getattr(source, "positional_embedding", None)
        self.positional_scale = float(getattr(source, "positional_scale", 1.0))
        self.max_period = getattr(source, "max_period", None)
        self._remote_module = importlib.import_module(source.__class__.__module__)
        self._create_sin_embedding = getattr(
            self._remote_module, "create_sin_embedding", None
        )

    def resolve_attention_implementation(self, x: torch.Tensor) -> str:
        if len(self.layers) == 0:
            return "sdpa"
        first_layer = self.layers[0]
        if not isinstance(first_layer, MossTTSLocalTransformerLayer):
            return self.source.resolve_attention_implementation(x)
        return first_layer.self_attn.resolve_attention_implementation(
            x,
            is_streaming=getattr(self.source, "_streaming_state", None) is not None,
        )

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        streaming_state = getattr(self.source, "_streaming_state", None)
        if self.positional_embedding in {"sin", "sin_rope"}:
            if self._create_sin_embedding is None:
                raise RuntimeError(
                    "MOSS vocoder transformer cannot create sin embeddings"
                )
            if x.dim() == 3:
                offsets = (
                    streaming_state.offsets
                    if streaming_state is not None
                    else torch.zeros(1, dtype=torch.long, device=x.device)
                )
                positions = torch.arange(x.shape[1], device=x.device).view(
                    1, -1
                ) + offsets.view(-1, 1)
            else:
                positions = kwargs.get("position_ids")
                if positions is None:
                    raise ValueError(
                        "packed transformer inputs require position_ids for "
                        "sinusoidal embeddings"
                    )
            pos_emb = self._create_sin_embedding(
                positions,
                x.shape[-1],
                max_period=self.max_period,
                dtype=x.dtype,
            )
            x = x + self.positional_scale * pos_emb
        for layer in self.layers:
            x = layer(x, **kwargs)
        if streaming_state is not None and x.dim() == 3:
            streaming_state.offsets[:] = torch.where(
                streaming_state.exec_mask,
                streaming_state.offsets + x.shape[1],
                streaming_state.offsets,
            )
        return x


class MossTTSLocalProjectedTransformer(nn.Module):
    """Projected transformer decoder stage with the MOSS input/output layout."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.input_proj = getattr(source, "input_proj", None)
        self.output_proj = getattr(source, "output_proj", None)
        self.transformer = MossTTSLocalTransformer(getattr(source, "transformer"))
        self.is_streaming = bool(getattr(source, "is_streaming", False))
        self._single_unpadded_metadata_cache = _SingleUnpaddedMetadataCache()
        if self.input_proj is None or self.output_proj is None:
            raise ValueError("MOSS vocoder projected transformer requires projections")

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x.transpose(1, 2))
        backend = self.transformer.resolve_attention_implementation(x)
        if not self.source.is_streaming and backend == "flash_attention_2":
            batch_size, max_seqlen, _ = x.shape
            if max_seqlen > 0 and batch_size == 1:
                max_valid_seqlen = int(input_lengths[0].item())
                pack_mode = (
                    "single_unpadded" if max_valid_seqlen == max_seqlen else "masked"
                )
            elif max_seqlen > 0 and bool(input_lengths.any().item()):
                max_valid_seqlen = int(input_lengths.max().item())
                pack_mode = "masked"
            else:
                max_valid_seqlen = 0
                pack_mode = ""
            if max_valid_seqlen > 0:
                if pack_mode == "single_unpadded":
                    packed_x, cu_seqlens, position_ids = _pack_single_unpadded_sequence(
                        x,
                        self._single_unpadded_metadata_cache,
                    )
                    valid_mask = None
                else:
                    (
                        packed_x,
                        valid_mask,
                        cu_seqlens,
                        position_ids,
                    ) = _pack_padded_sequence(
                        x,
                        input_lengths,
                    )
                packed_x = self.transformer(
                    packed_x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_valid_seqlen,
                    position_ids=position_ids,
                    input_lengths=input_lengths,
                    **kwargs,
                )
                if valid_mask is None:
                    x = _unpack_single_unpadded_sequence(packed_x)
                else:
                    x = _unpack_packed_sequence(
                        packed_x,
                        valid_mask,
                        batch_size,
                        max_seqlen,
                    )
            else:
                x = x.new_zeros(x.shape)
        else:
            x = self.transformer(x, input_lengths=input_lengths, **kwargs)
        return self.output_proj(x).transpose(1, 2), input_lengths


class MossTTSLocalVocoderDecoder(nn.Module):
    """Iterable MOSS vocoder decoder with patched projected transformers."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        source_stages = _module_list(source)
        if not source_stages:
            raise ValueError("MOSS vocoder decoder must be a non-empty stage list")
        self.stages = nn.ModuleList(
            [self._wrap_stage(stage) for stage in source_stages]
        )

    @staticmethod
    def _wrap_stage(stage: nn.Module) -> nn.Module:
        if hasattr(stage, "transformer"):
            return MossTTSLocalProjectedTransformer(stage)
        if hasattr(stage, "patch_size"):
            return stage
        raise ValueError(
            f"unsupported MOSS vocoder decoder stage {stage.__class__.__name__}"
        )

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    def __getitem__(self, index: int) -> nn.Module:
        return self.stages[index]

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for stage in self.stages:
            x, input_lengths = stage(x, input_lengths)
        return x, input_lengths


def build_moss_tts_local_vocoder_decoder(codec: Any) -> MossTTSLocalVocoderDecoder:
    decoder = getattr(codec, "decoder", None)
    if decoder is None:
        raise RuntimeError("MOSS vocoder codec is missing decoder")
    return MossTTSLocalVocoderDecoder(decoder)


@contextmanager
def use_moss_tts_local_vocoder_decoder(
    codec: Any,
    decoder: MossTTSLocalVocoderDecoder,
):
    original_decoder = getattr(codec, "decoder", None)
    if original_decoder is None:
        raise RuntimeError("MOSS vocoder codec is missing decoder")
    codec.decoder = decoder
    try:
        yield
    finally:
        codec.decoder = original_decoder


__all__ = [
    "MossTTSLocalAttention",
    "MossTTSLocalProjectedTransformer",
    "MossTTSLocalTransformer",
    "MossTTSLocalTransformerLayer",
    "MossTTSLocalVocoderDecoder",
    "build_moss_tts_local_vocoder_decoder",
    "use_moss_tts_local_vocoder_decoder",
]
