# SPDX-License-Identifier: Apache-2.0
"""Owned MOSS-TTS Local vocoder decoder stages.

This module mirrors the remote MOSS-Audio-Tokenizer-v2 decoder stage mechanics
without changing the scheduler, codec embeddings, or waveform projection code.
It is intentionally MOSS-specific: the decoder is a chain of projected
transformers and patch transforms, not a generic LLM model runner.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _module_list(value: Any) -> list[nn.Module]:
    if isinstance(value, nn.ModuleList):
        return list(value)
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, nn.Module) for item in value
    ):
        return list(value)
    return []


def _first_attr(obj: Any, *names: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


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


def _unpack_packed_sequence(
    packed_x: torch.Tensor,
    valid_mask: torch.Tensor,
    batch_size: int,
    max_seqlen: int,
) -> torch.Tensor:
    x = packed_x.new_zeros(batch_size, max_seqlen, packed_x.shape[-1])
    x[valid_mask] = packed_x
    return x


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
        self._remote_module = importlib.import_module(source.__class__.__module__)
        self._apply_rope_with_positions = getattr(
            self._remote_module, "apply_rope_with_positions", None
        )
        self._flash_attn_varlen_func = getattr(
            self._remote_module, "flash_attn_varlen_func", None
        )

    def forward(
        self,
        query: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        position_ids: torch.Tensor | None = None,
        input_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        backend = self.source.resolve_attention_implementation(
            query,
            is_streaming=False,
        )
        if backend == "flash_attention_2":
            if query.dim() != 2:
                raise ValueError(
                    f"packed flash attention expects a 2D tensor, got {tuple(query.shape)}"
                )
            if cu_seqlens is None or max_seqlen is None or position_ids is None:
                raise ValueError(
                    "packed flash attention requires cu_seqlens, max_seqlen, and position_ids"
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        if self._apply_rope_with_positions is None:
            return self.source._apply_packed_rope(q, k, position_ids)
        max_period = getattr(self.rope, "max_period", None)
        return self._apply_rope_with_positions(
            q,
            k,
            position_ids,
            max_period=max_period,
        )

    def _forward_packed_flash(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        flash_attn_varlen_func = self._flash_attn_varlen_func
        if flash_attn_varlen_func is None:
            raise RuntimeError("flash attention is not available for MOSS vocoder")
        q, k, v = self._project_qkv(x)
        q, k = self._apply_packed_rope(q, k, position_ids)
        window_size = (
            (int(self.context), 0)
            if self.context is not None and self.causal
            else (-1, -1)
        )
        out = flash_attn_varlen_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=self.causal,
            window_size=window_size,
        )
        return self.out_proj(out.reshape(x.shape[0], self.embed_dim))


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
        return self.source.resolve_attention_implementation(x)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if self.positional_embedding in {"sin", "sin_rope"}:
            if self._create_sin_embedding is None:
                raise RuntimeError(
                    "MOSS vocoder transformer cannot create sin embeddings"
                )
            if x.dim() == 3:
                positions = torch.arange(x.shape[1], device=x.device).view(1, -1)
            else:
                positions = kwargs.get("position_ids")
                if positions is None:
                    raise ValueError(
                        "packed transformer inputs require position_ids for sinusoidal embeddings"
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
        if self.input_proj is None or self.output_proj is None:
            raise ValueError("MOSS vocoder projected transformer requires projections")

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_x = x
        x = self.input_proj(x.transpose(1, 2))
        if (
            not self.is_streaming
            and self.transformer.resolve_attention_implementation(x)
            == "flash_attention_2"
        ):
            batch_size, max_seqlen, _ = x.shape
            if max_seqlen > 0 and bool(input_lengths.any().item()):
                max_valid_seqlen = int(input_lengths.max().item())
                packed_x, valid_mask, cu_seqlens, position_ids = _pack_padded_sequence(
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
                x = _unpack_packed_sequence(
                    packed_x,
                    valid_mask,
                    batch_size,
                    max_seqlen,
                )
            else:
                x = x.new_zeros(x.shape)
        else:
            return self.source(source_x, input_lengths, **kwargs)
        return self.output_proj(x).transpose(1, 2), input_lengths


class MossTTSLocalPatchTransform(nn.Module):
    """MOSS codec patch reshape stage."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.patch_size = int(getattr(source, "patch_size"))
        self.downsample_ratio = getattr(source, "downsample_ratio", None)
        self.is_downsample = bool(getattr(source, "is_downsample", False))
        self.module_type = getattr(source, "module_type", "PatchedPretransform")

    def encode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, dim, _ = x.shape
        patch = self.patch_size
        x = x.reshape(batch_size, dim, -1, patch)
        x = x.permute(0, 1, 3, 2).reshape(batch_size, dim * patch, -1)
        return x, input_lengths // patch

    def decode(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, patched_dim, length = x.shape
        patch = self.patch_size
        dim = patched_dim // patch
        x = x.reshape(batch_size, dim, patch, length)
        x = x.permute(0, 1, 3, 2).reshape(batch_size, dim, length * patch)
        return x, input_lengths * patch

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_downsample:
            return self.encode(x, input_lengths)
        return self.decode(x, input_lengths)


class MossTTSLocalVocoderDecoder(nn.Module):
    """Iterable MOSS vocoder decoder replacement."""

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
            return MossTTSLocalPatchTransform(stage)
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


def install_moss_tts_local_vocoder_decoder(codec: Any) -> nn.Module:
    """Install the owned decoder and return the previous decoder."""
    original_decoder = getattr(codec, "decoder", None)
    if original_decoder is None:
        raise RuntimeError("MOSS vocoder codec is missing decoder")
    if isinstance(original_decoder, MossTTSLocalVocoderDecoder):
        return original_decoder
    owned_decoder = MossTTSLocalVocoderDecoder(original_decoder)
    codec.decoder = owned_decoder
    logger.info(
        "MOSS-TTS Local vocoder decoder=owned-pytorch stages=%d",
        len(owned_decoder),
    )
    return original_decoder


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
    "MossTTSLocalPatchTransform",
    "MossTTSLocalProjectedTransformer",
    "MossTTSLocalTransformer",
    "MossTTSLocalTransformerLayer",
    "MossTTSLocalVocoderDecoder",
    "build_moss_tts_local_vocoder_decoder",
    "install_moss_tts_local_vocoder_decoder",
    "use_moss_tts_local_vocoder_decoder",
]
