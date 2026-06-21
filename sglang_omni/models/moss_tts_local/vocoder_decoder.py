# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS Local non-streaming vocoder decoder with packed attention.

The wrapper keeps the upstream codec embeddings, pretransform stages, streaming
state, and waveform projection. It replaces only the non-streaming projected
transformer attention path so decoder frames can run through SGLang's packed
varlen FlashAttention.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterator
from typing import Any

import torch
from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
from torch import nn


class _UnpaddedMetadataCache:
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


def _pack_unpadded_sequence(
    x: torch.Tensor,
    metadata_cache: "_UnpaddedMetadataCache",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.shape[0] == 1, f"expected a single unpadded sequence, got {x.shape[0]}"
    _, max_seqlen, _ = x.shape
    packed_x = x.reshape(max_seqlen, x.shape[-1])
    cu_seqlens, position_ids = metadata_cache.get(
        device=x.device,
        max_seqlen=max_seqlen,
    )
    return packed_x, cu_seqlens, position_ids


def _unpack_packed_sequence(
    packed_x: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
) -> torch.Tensor:
    x = packed_x.new_zeros(batch_size, max_seqlen, packed_x.shape[-1])
    x[valid_mask] = packed_x
    return x


def _unpack_unpadded_sequence(
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
        self.in_proj = source.in_proj
        self.out_proj = source.out_proj
        self.embed_dim = int(source.embed_dim)
        self.num_heads = int(source.num_heads)
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
        self._flash_attn_varlen = flash_attn_varlen_func
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
        if self._can_run_packed_flash(x) and (
            backend == "flash_attention_2"
            or self.source.attention_implementation == "flash_attention_2"
        ):
            return "flash_attention_2"
        return backend

    def _can_run_packed_flash(self, x: torch.Tensor) -> bool:
        if x.device.type != "cuda":
            return False
        return self.source._get_backend_check_dtype(x) == torch.bfloat16

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
            out = self.source._forward_streaming_sdpa(query, streaming_state)
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
        out = self._flash_attn_varlen(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=self.causal,
            window_size=self._flash_window_size(),
        )
        return self.out_proj(out.reshape(x.shape[0], self.embed_dim))

    def _flash_window_size(self) -> tuple[int, int]:
        if self.context is None or not self.causal:
            return (-1, -1)
        # MOSS's SDPA local mask keeps `context` total tokens including the current
        # query token. FlashAttention's left-window argument counts prior keys.
        return (max(int(self.context) - 1, 0), 0)


class MossTTSLocalTransformerLayer(nn.Module):
    """One MOSS vocoder transformer layer."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.norm1 = source.norm1
        self.norm2 = source.norm2
        self.layer_scale_1 = source.layer_scale_1
        self.layer_scale_2 = source.layer_scale_2
        self.ffn = source.ffn
        self.self_attn = MossTTSLocalAttention(source.self_attn)
        assert (
            isinstance(self.ffn, nn.Sequential) and len(self.ffn) >= 3
        ), "MOSS vocoder transformer layer requires Linear-GELU-Linear FFN"

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
            [MossTTSLocalTransformerLayer(layer) for layer in source.layers]
        )
        self.positional_embedding = getattr(source, "positional_embedding", None)
        self.positional_scale = float(getattr(source, "positional_scale", 1.0))
        self.max_period = getattr(source, "max_period", None)
        self._remote_module = importlib.import_module(source.__class__.__module__)
        self._create_sin_embedding = getattr(
            self._remote_module, "create_sin_embedding", None
        )

    def resolve_attention_implementation(self, x: torch.Tensor) -> str:
        assert len(self.layers) > 0, "MOSS vocoder transformer must have layers"
        return self.layers[0].self_attn.resolve_attention_implementation(
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
        self.input_proj = source.input_proj
        self.output_proj = source.output_proj
        self.transformer = MossTTSLocalTransformer(source.transformer)
        self.is_streaming = bool(source.is_streaming)
        self._unpadded_metadata_cache = _UnpaddedMetadataCache()

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x.transpose(1, 2))
        backend = self.transformer.resolve_attention_implementation(x)
        if not self.is_streaming and backend == "flash_attention_2":
            batch_size, max_seqlen, _ = x.shape
            if max_seqlen == 0 or not bool(input_lengths.any().item()):
                x = x.new_zeros(x.shape)
            else:
                max_valid_seqlen = int(input_lengths.max().item())
                is_unpadded_single = (
                    batch_size == 1 and int(input_lengths[0].item()) == max_seqlen
                )
                if is_unpadded_single:
                    packed_x, cu_seqlens, position_ids = _pack_unpadded_sequence(
                        x,
                        self._unpadded_metadata_cache,
                    )
                    valid_mask = None
                else:
                    packed_x, valid_mask, cu_seqlens, position_ids = (
                        _pack_padded_sequence(x, input_lengths)
                    )
                packed_x = self.transformer(
                    packed_x,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_valid_seqlen,
                    position_ids=position_ids,
                    input_lengths=input_lengths,
                    **kwargs,
                )
                x = (
                    _unpack_unpadded_sequence(packed_x)
                    if valid_mask is None
                    else _unpack_packed_sequence(
                        packed_x,
                        valid_mask,
                        batch_size,
                        max_seqlen,
                    )
                )
        else:
            x = self.transformer(x, input_lengths=input_lengths, **kwargs)
        return self.output_proj(x).transpose(1, 2), input_lengths


class MossTTSLocalVocoderDecoder(nn.Module):
    """Iterable MOSS vocoder decoder with patched projected transformers."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        source_stages = list(source)
        assert source_stages, "MOSS vocoder decoder must be a non-empty stage list"
        self.stages = nn.ModuleList(
            [self._wrap_stage(stage) for stage in source_stages]
        )

    @staticmethod
    def _wrap_stage(stage: nn.Module) -> nn.Module:
        module_type = stage.module_type
        if module_type == "Transformer":
            return MossTTSLocalProjectedTransformer(stage)
        if module_type == "PatchedPretransform":
            return stage
        raise ValueError(
            f"unsupported MOSS vocoder decoder stage {stage.__class__.__name__} "
            f"with module_type={module_type!r}"
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
