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
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
from torch import nn

from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import get_recorder as _get_event_recorder
from sglang_omni.profiler.ranges import torch_profile_range

logger = logging.getLogger(__name__)

_ATTENTION_KERNEL_ENV = "SGLANG_OMNI_MOSS_LOCAL_VOCODER_ATTENTION_KERNEL"
_ATTENTION_KERNEL_REMOTE = "remote"
_ATTENTION_KERNEL_SGLANG = "sglang"
_SUPPORTED_ATTENTION_KERNELS = {
    _ATTENTION_KERNEL_REMOTE,
    _ATTENTION_KERNEL_SGLANG,
}
_ATTN_PROFILE_REQUEST_ID: ContextVar[str | None] = ContextVar(
    "moss_tts_local_attn_profile_request_id", default=None
)
_ATTN_PROFILE_METADATA: ContextVar[dict[str, Any] | None] = ContextVar(
    "moss_tts_local_attn_profile_metadata", default=None
)


def _attention_kernel_preference() -> str:
    value = os.environ.get(_ATTENTION_KERNEL_ENV, _ATTENTION_KERNEL_REMOTE)
    value = value.strip().lower()
    if value not in _SUPPORTED_ATTENTION_KERNELS:
        raise ValueError(
            f"{_ATTENTION_KERNEL_ENV} must be one of "
            f"{sorted(_SUPPORTED_ATTENTION_KERNELS)}"
        )
    return value


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


@contextmanager
def profile_moss_tts_local_vocoder_attention(
    request_id: str | None,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    token_id = _ATTN_PROFILE_REQUEST_ID.set(request_id)
    token_metadata = _ATTN_PROFILE_METADATA.set(metadata)
    try:
        yield
    finally:
        _ATTN_PROFILE_METADATA.reset(token_metadata)
        _ATTN_PROFILE_REQUEST_ID.reset(token_id)


@contextmanager
def _attention_profile_interval(
    event_base: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> Iterator[None]:
    with torch_profile_range(f"moss.vocoder.attn.{event_base}"):
        request_id = _ATTN_PROFILE_REQUEST_ID.get()
        recorder = _get_event_recorder()
        if request_id is None or not recorder.is_active():
            yield
            return

        merged_metadata = dict(_ATTN_PROFILE_METADATA.get() or {})
        if metadata is not None:
            merged_metadata.update(metadata)
        start_ns = time.time_ns()
        _emit_event(
            request_id=request_id,
            stage=None,
            event_name=f"moss_vocoder_attn_{event_base}_start",
            metadata=merged_metadata,
            timestamp_ns=start_ns,
        )
        try:
            yield
        finally:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name=f"moss_vocoder_attn_{event_base}_end",
                metadata=merged_metadata,
                timestamp_ns=time.time_ns(),
            )


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
        self._attention_kernel = _attention_kernel_preference()
        self._sglang_flash_attn_varlen_func = _load_sglang_flash_attn_varlen_func()
        if (
            self._attention_kernel == _ATTENTION_KERNEL_SGLANG
            and self._sglang_flash_attn_varlen_func is None
        ):
            raise RuntimeError(
                f"{_ATTENTION_KERNEL_ENV}=sglang requires "
                "sglang.jit_kernel.flash_attention.flash_attn_varlen_func"
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
        streaming_state = getattr(self.source, "_streaming_state", None)
        backend = self.source.resolve_attention_implementation(
            query,
            is_streaming=streaming_state is not None,
        )
        if streaming_state is not None:
            if query.dim() != 3:
                raise ValueError(
                    f"streaming attention expects a 3D tensor, got {tuple(query.shape)}"
                )
            out = (
                self._forward_streaming_flash(query, streaming_state)
                if backend == "flash_attention_2"
                else self._forward_streaming_sdpa(query, streaming_state)
            )
            return self.out_proj(out)
        if backend == "flash_attention_2":
            if query.dim() != 2:
                raise ValueError(
                    f"packed flash attention expects a 2D tensor, got {tuple(query.shape)}"
                )
            if cu_seqlens is None or max_seqlen is None or position_ids is None:
                raise ValueError(
                    "packed flash attention requires cu_seqlens, max_seqlen, and position_ids"
                )
            with _attention_profile_interval(
                "packed_flash_path",
                metadata={
                    "attention_kernel": self._attention_kernel,
                    "query_dim": query.dim(),
                    "query_tokens": int(query.shape[0]),
                    "max_seqlen": max_seqlen,
                },
            ):
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
        with _attention_profile_interval(
            "dense_source_path",
            metadata={
                "attention_kernel": self._attention_kernel,
                "query_dim": query.dim(),
                "batch_size": int(query.shape[0]),
                "max_seqlen": int(query.shape[1]),
            },
        ):
            return self.source(
                query,
                input_lengths=input_lengths,
            )

    def _forward_streaming_sdpa(
        self, query: torch.Tensor, streaming_state: Any
    ) -> torch.Tensor:
        with _attention_profile_interval(
            "streaming_sdpa_path",
            metadata={
                "attention_kernel": self._attention_kernel,
                "batch_size": int(query.shape[0]),
                "chunk_length": int(query.shape[1]),
            },
        ):
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
        q, k, v = self._project_qkv(x)
        q, k = self._apply_packed_rope(q, k, position_ids)
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
        flash_attn_varlen_func = (
            self._sglang_flash_attn_varlen_func
            if self._attention_kernel == _ATTENTION_KERNEL_SGLANG
            else None
        )
        if flash_attn_varlen_func is None:
            flash_attn_varlen_func = self._flash_attn_varlen_func
        if flash_attn_varlen_func is None:
            raise RuntimeError("flash attention is not available for MOSS vocoder")
        window_size = (
            (int(self.context), 0)
            if self.context is not None and self.causal
            else (-1, -1)
        )
        event_base = (
            "flash_sglang"
            if self._attention_kernel == _ATTENTION_KERNEL_SGLANG
            else "flash_remote"
        )
        with _attention_profile_interval(
            event_base,
            metadata={
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "window_size": window_size,
            },
        ):
            out = flash_attn_varlen_func(
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
        return out

    def _forward_streaming_flash(
        self,
        x: torch.Tensor,
        state: Any,
    ) -> torch.Tensor:
        batch_size, chunk_length, _ = x.shape
        with _attention_profile_interval(
            "project_qkv",
            metadata={
                "batch_size": batch_size,
                "chunk_length": chunk_length,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "context": self.context,
            },
        ):
            q, k_cur, v_cur = self._project_qkv(x)
        if self.rope is not None:
            with _attention_profile_interval("rope"):
                q, k_cur = self.rope(q, k_cur, state.offset, time_before_heads=False)
        pos_q = state.offset.view(-1, 1) + torch.arange(
            chunk_length,
            device=x.device,
            dtype=torch.long,
        ).view(1, -1)
        with _attention_profile_interval("ensure_cache"):
            cached_k, cached_v, cached_pos = self.source._ensure_streaming_cache(
                state,
                batch_size,
                k_cur.device,
                k_cur.dtype,
            )

        with _attention_profile_interval("build_kv"):
            k_all, v_all, pos_k = self.source._build_streaming_kv(
                cached_k,
                cached_v,
                cached_pos,
                k_cur,
                v_cur,
                pos_q,
            )

        q_chunks = []
        k_chunks = []
        v_chunks = []
        cu_q = [0]
        cu_k = [0]
        max_kv_len = 0

        with _attention_profile_interval("pack_varlen"):
            for batch_idx in range(batch_size):
                valid_k = pos_k[batch_idx] >= 0
                q_i = q[batch_idx].transpose(0, 1).contiguous()
                k_i = k_all[batch_idx, :, valid_k, :].transpose(0, 1).contiguous()
                v_i = v_all[batch_idx, :, valid_k, :].transpose(0, 1).contiguous()
                q_chunks.append(q_i)
                k_chunks.append(k_i)
                v_chunks.append(v_i)
                cu_q.append(cu_q[-1] + q_i.shape[0])
                cu_k.append(cu_k[-1] + k_i.shape[0])
                max_kv_len = max(max_kv_len, int(k_i.shape[0]))
            q_pack = torch.cat(q_chunks, dim=0)
            k_pack = torch.cat(k_chunks, dim=0)
            v_pack = torch.cat(v_chunks, dim=0)
            cu_q_tensor = torch.tensor(cu_q, device=x.device, dtype=torch.int32)
            cu_k_tensor = torch.tensor(cu_k, device=x.device, dtype=torch.int32)

        out_flat = self._run_flash_attention(
            q_pack,
            k_pack,
            v_pack,
            cu_q_tensor,
            cu_k_tensor,
            max_seqlen_q=chunk_length,
            max_seqlen_k=max_kv_len,
        )

        with _attention_profile_interval("unpack_varlen"):
            outputs = []
            start = 0
            for _ in range(batch_size):
                outputs.append(
                    out_flat[start : start + chunk_length].transpose(0, 1).contiguous()
                )
                start += chunk_length
            out = torch.stack(outputs, dim=0)
            out = out.transpose(1, 2).reshape(batch_size, chunk_length, self.embed_dim)

        with _attention_profile_interval("update_cache"):
            self._update_streaming_cache_in_place(
                state,
                cached_k,
                cached_v,
                cached_pos,
                k_all,
                v_all,
                pos_k,
            )
            state.offset[:] = torch.where(
                state.exec_mask,
                state.offset + chunk_length,
                state.offset,
            )
        return out

    def _update_streaming_cache_in_place(
        self,
        state: Any,
        cached_k: torch.Tensor,
        cached_v: torch.Tensor,
        cached_pos: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        pos_k: torch.Tensor,
    ) -> None:
        if self.context is None:
            self.source._update_streaming_cache(
                state,
                cached_k,
                cached_v,
                cached_pos,
                k_all,
                v_all,
                pos_k,
            )
            return

        context = int(self.context)
        exec_mask = state.exec_mask.to(device=cached_k.device, dtype=torch.bool)
        exec_mask_kv = exec_mask.view(-1, 1, 1, 1)
        exec_mask_pos = exec_mask.to(device=cached_pos.device).view(-1, 1)

        new_cached_k = k_all[:, :, -context:, :].contiguous()
        new_cached_v = v_all[:, :, -context:, :].contiguous()
        new_cached_pos = pos_k[:, -context:].contiguous()

        cached_k.copy_(torch.where(exec_mask_kv, new_cached_k, cached_k))
        cached_v.copy_(torch.where(exec_mask_kv, new_cached_v, cached_v))
        cached_pos.copy_(torch.where(exec_mask_pos, new_cached_pos, cached_pos))


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
        self.source = source
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
        with _attention_profile_interval(
            "projected_input_proj",
            metadata={
                "batch_size": int(x.shape[0]),
                "input_channels": int(x.shape[1]),
                "max_seqlen": int(x.shape[2]),
                "stage_type": self.source.__class__.__name__,
            },
        ):
            x = self.input_proj(x.transpose(1, 2))
        backend = self.transformer.resolve_attention_implementation(x)
        if not self.source.is_streaming and backend == "flash_attention_2":
            batch_size, max_seqlen, _ = x.shape
            if max_seqlen > 0 and bool(input_lengths.any().item()):
                max_valid_seqlen = int(input_lengths.max().item())
                with _attention_profile_interval(
                    "projected_pack_padded",
                    metadata={
                        "backend": backend,
                        "batch_size": batch_size,
                        "max_seqlen": max_seqlen,
                        "max_valid_seqlen": max_valid_seqlen,
                    },
                ):
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
                with _attention_profile_interval(
                    "projected_unpack_padded",
                    metadata={
                        "backend": backend,
                        "batch_size": batch_size,
                        "max_seqlen": max_seqlen,
                        "max_valid_seqlen": max_valid_seqlen,
                    },
                ):
                    x = _unpack_packed_sequence(
                        packed_x,
                        valid_mask,
                        batch_size,
                        max_seqlen,
                    )
            else:
                x = x.new_zeros(x.shape)
        else:
            with _attention_profile_interval(
                "projected_dense_path",
                metadata={
                    "backend": backend,
                    "is_streaming": bool(self.source.is_streaming),
                    "batch_size": int(x.shape[0]),
                    "max_seqlen": int(x.shape[1]),
                },
            ):
                x = self.transformer(x, input_lengths=input_lengths, **kwargs)
        with _attention_profile_interval(
            "projected_output_proj",
            metadata={
                "backend": backend,
                "batch_size": int(x.shape[0]),
                "max_seqlen": int(x.shape[1]),
            },
        ):
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
        "MOSS-TTS Local vocoder decoder=owned stages=%d",
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
    "MossTTSLocalProjectedTransformer",
    "MossTTSLocalTransformer",
    "MossTTSLocalTransformerLayer",
    "MossTTSLocalVocoderDecoder",
    "build_moss_tts_local_vocoder_decoder",
    "install_moss_tts_local_vocoder_decoder",
    "use_moss_tts_local_vocoder_decoder",
]
