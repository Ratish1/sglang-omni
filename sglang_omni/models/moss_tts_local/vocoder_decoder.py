# SPDX-License-Identifier: Apache-2.0
"""Patched MOSS-TTS Local vocoder decoder stages.

This module mirrors the remote MOSS-Audio-Tokenizer-v2 decoder stage mechanics
without changing the scheduler, codec embeddings, or waveform projection code.
It is intentionally MOSS-specific: the interception point is the decoder's
projected transformer stages, not the codec embeddings or waveform projection.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from sglang.jit_kernel.chunked_local_attention import (
    LocalCausalVarlenWorkspace,
    local_causal_varlen_attention_with_cache,
)
from sglang.jit_kernel.flash_attention import (
    flash_attn_varlen_func as sglang_flash_attn_varlen_func,
)
from torch import nn

logger = logging.getLogger(__name__)

_SOURCE_ATTENTION = "source"


def _module_list(value: Any) -> list[nn.Module]:
    if isinstance(value, nn.ModuleList):
        return list(value)
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, nn.Module) for item in value
    ):
        return list(value)
    return []


def _accepts_kwarg(fn: Any, name: str) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == name:
            return True
    return False


def _has_parameters(fn: Any, names: set[str]) -> bool:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return names.issubset(signature.parameters)


class MossTTSLocalAttention(nn.Module):
    """MOSS local-causal self attention with optional streaming SGLang FA."""

    def __init__(
        self,
        source: nn.Module,
        *,
        max_batch_size: int = 16,
        max_chunk_frames: int = 100,
    ) -> None:
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
        self._remote_has_flash_attn = bool(
            getattr(self._remote_module, "HAS_FLASH_ATTN", False)
        )
        self._remote_flash_attn_varlen_func = getattr(
            self._remote_module, "flash_attn_varlen_func", None
        )
        self._source_accepts_input_lengths = _accepts_kwarg(
            self.source.forward, "input_lengths"
        )
        self._source_accepts_qkv = _has_parameters(
            self.source.forward, {"key", "value"}
        )
        self._sglang_flash_attn_varlen_func = sglang_flash_attn_varlen_func
        self._workspace_cls = LocalCausalVarlenWorkspace
        self._local_attention_func = local_causal_varlen_attention_with_cache
        self._max_batch_size = int(max_batch_size)
        self._max_chunk_frames = int(max_chunk_frames)
        self._workspace: Any | None = None

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
        if self._can_use_sglang_flash_attention(
            x,
            backend=backend,
            is_streaming=is_streaming,
        ):
            return "flash_attention_2"
        if backend == "flash_attention_2":
            return _SOURCE_ATTENTION
        return backend

    def _can_use_sglang_flash_attention(
        self,
        x: torch.Tensor,
        *,
        backend: str,
        is_streaming: bool,
    ) -> bool:
        if not is_streaming:
            return False
        if (
            backend != "flash_attention_2"
            and getattr(self.source, "attention_implementation", None)
            != "flash_attention_2"
        ):
            return False
        return self._supports_sglang_flash_attention(x)

    def _supports_sglang_flash_attention(self, x: torch.Tensor) -> bool:
        if self.context is None or x.device.type != "cuda":
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
            if backend == _SOURCE_ATTENTION:
                return self._forward_source_attention(query, input_lengths)
            if backend == "flash_attention_2":
                return self.out_proj(
                    self._forward_streaming_flash(query, streaming_state)
                )
            return self.out_proj(self._forward_streaming_sdpa(query, streaming_state))
        if query.dim() != 3:
            raise ValueError(
                f"dense attention expects a 3D tensor, got {tuple(query.shape)}"
            )
        return self._forward_source_attention(query, input_lengths)

    def _forward_source_attention(
        self,
        query: torch.Tensor,
        input_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._source_accepts_qkv and self._source_accepts_input_lengths:
            return self.source(query, query, query, input_lengths=input_lengths)
        if self._source_accepts_qkv:
            return self.source(query, query, query)
        if self._source_accepts_input_lengths:
            return self.source(query, input_lengths=input_lengths)
        return self.source(query, query, query)

    def _forward_streaming_sdpa(
        self, query: torch.Tensor, streaming_state: Any
    ) -> torch.Tensor:
        return self.source._forward_streaming_sdpa(query, streaming_state)

    def _forward_streaming_flash(
        self,
        x: torch.Tensor,
        state: Any,
    ) -> torch.Tensor:
        batch_size, chunk_length, _ = x.shape
        q, k_cur, v_cur = self._project_qkv(x)
        if self.rope is not None:
            q, k_cur = self.rope(q, k_cur, state.offset, time_before_heads=False)

        cached_k, cached_v, cached_pos = self.source._ensure_streaming_cache(
            state,
            batch_size,
            k_cur.device,
            k_cur.dtype,
        )

        workspace = self._workspace_for(q)
        out_bhtd = self._local_attention_func(
            q,
            k_cur,
            v_cur,
            cached_k,
            cached_v,
            cached_pos,
            state.offset,
            state.exec_mask,
            workspace,
            context=int(self.context),
            flash_attn_varlen_func=self._sglang_flash_attn_varlen_func,
            window_size=self._flash_window_size(),
        )
        return out_bhtd.transpose(1, 2).reshape(
            batch_size, chunk_length, self.embed_dim
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

    def _workspace_for(self, x: torch.Tensor) -> Any:
        if self.context is None:
            raise RuntimeError("SGLang streaming attention requires finite context")
        workspace = self._workspace
        if (
            workspace is not None
            and workspace.q_pack.device == x.device
            and workspace.q_pack.dtype == x.dtype
        ):
            return workspace
        workspace = self._workspace_cls.create(
            max_batch_size=self._max_batch_size,
            max_chunk_len=self._max_chunk_frames,
            context=int(self.context),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        self._workspace = workspace
        return workspace

    def _flash_window_size(self) -> tuple[int, int]:
        if self.context is None or not self.causal:
            return (-1, -1)
        context = int(self.context)
        if (
            self._remote_has_flash_attn
            and self._remote_flash_attn_varlen_func is not None
        ):
            return (context, 0)
        return (max(context - 1, 0), 0)


class MossTTSLocalTransformerLayer(nn.Module):
    """One MOSS vocoder transformer layer."""

    def __init__(
        self,
        source: nn.Module,
        *,
        max_batch_size: int = 16,
        max_chunk_frames: int = 100,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.norm1 = getattr(source, "norm1", None)
        self.norm2 = getattr(source, "norm2", None)
        self.layer_scale_1 = getattr(source, "layer_scale_1", None)
        self.layer_scale_2 = getattr(source, "layer_scale_2", None)
        self.ffn = getattr(source, "ffn", None)
        self.self_attn = MossTTSLocalAttention(
            getattr(source, "self_attn"),
            max_batch_size=max_batch_size,
            max_chunk_frames=max_chunk_frames,
        )
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

    def __init__(
        self,
        source: nn.Module,
        *,
        max_batch_size: int = 16,
        max_chunk_frames: int = 100,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.layers = nn.ModuleList(
            [
                MossTTSLocalTransformerLayer(
                    layer,
                    max_batch_size=max_batch_size,
                    max_chunk_frames=max_chunk_frames,
                )
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
            if x.dim() != 3:
                raise ValueError(
                    f"MOSS streaming transformer expects [B, T, C], got {tuple(x.shape)}"
                )
            offsets = (
                streaming_state.offsets
                if streaming_state is not None
                else torch.zeros(1, dtype=torch.long, device=x.device)
            )
            positions = torch.arange(x.shape[1], device=x.device).view(
                1, -1
            ) + offsets.view(-1, 1)
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

    def __init__(
        self,
        source: nn.Module,
        *,
        max_batch_size: int = 16,
        max_chunk_frames: int = 100,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "source", source)
        self.input_proj = getattr(source, "input_proj", None)
        self.output_proj = getattr(source, "output_proj", None)
        self.transformer = MossTTSLocalTransformer(
            getattr(source, "transformer"),
            max_batch_size=max_batch_size,
            max_chunk_frames=max_chunk_frames,
        )
        self.is_streaming = bool(getattr(source, "is_streaming", False))
        if self.input_proj is None or self.output_proj is None:
            raise ValueError("MOSS vocoder projected transformer requires projections")

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        backend = self.transformer.resolve_attention_implementation(x.transpose(1, 2))
        if not self.source.is_streaming and backend != "flash_attention_2":
            return self.source(x, input_lengths, **kwargs)

        x = self.input_proj(x.transpose(1, 2))
        x = self.transformer(x, input_lengths=input_lengths, **kwargs)
        return self.output_proj(x).transpose(1, 2), input_lengths


class MossTTSLocalVocoderDecoder(nn.Module):
    """Iterable MOSS vocoder decoder with patched projected transformers."""

    def __init__(
        self,
        source: nn.Module,
        *,
        max_batch_size: int = 16,
        max_chunk_frames: int = 100,
    ) -> None:
        super().__init__()
        self.source = source
        source_stages = _module_list(source)
        if not source_stages:
            raise ValueError("MOSS vocoder decoder must be a non-empty stage list")
        self.stages = nn.ModuleList(
            [
                self._wrap_stage(
                    stage,
                    max_batch_size=max_batch_size,
                    max_chunk_frames=max_chunk_frames,
                )
                for stage in source_stages
            ]
        )

    @staticmethod
    def _wrap_stage(
        stage: nn.Module,
        *,
        max_batch_size: int,
        max_chunk_frames: int,
    ) -> nn.Module:
        if hasattr(stage, "transformer"):
            return MossTTSLocalProjectedTransformer(
                stage,
                max_batch_size=max_batch_size,
                max_chunk_frames=max_chunk_frames,
            )
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


def build_moss_tts_local_vocoder_decoder(
    codec: Any,
    *,
    max_batch_size: int,
    max_chunk_frames: int,
) -> MossTTSLocalVocoderDecoder:
    decoder = getattr(codec, "decoder", None)
    if decoder is None:
        raise RuntimeError("MOSS vocoder codec is missing decoder")
    return MossTTSLocalVocoderDecoder(
        decoder,
        max_batch_size=max_batch_size,
        max_chunk_frames=max_chunk_frames,
    )


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
