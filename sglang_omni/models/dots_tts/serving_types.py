# SPDX-License-Identifier: Apache-2.0
"""Shared serving dataclasses for native dots TTS integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class DotsTTSPreparedInputs:
    """Prepared dots request inputs consumed by the SGLang request builder."""

    raw_inputs: dict[str, Any]
    input_ids: torch.Tensor | None = None
    generation_schedule: torch.Tensor | None = None
    audio_span_positions: torch.Tensor | None = None
    prefill_end: int | None = None
    audio_placeholder_ids: set[int] = field(default_factory=set)
    prompt_patch_embeddings: torch.Tensor | None = None
    prompt_conditioning: Any = None
    fm_state: Any = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DotsTTSAudioStepResult:
    """One dots audio decode step produced from a SGLang hidden state."""

    latent_patch: torch.Tensor
    feedback_embedding: torch.Tensor
    eos_score: torch.Tensor | None


@dataclass(frozen=True)
class DotsTTSFlowBatchKey:
    """Compatibility key for batching dots DiT/flow latent steps."""

    device: torch.device
    dtype: torch.dtype
    mode: str
    ode_method: str
    num_steps: int
    guidance_scale: float
    history_bucket_capacity: int
    latent_patch_size: int
    hidden_patch_size: int


@dataclass
class DotsTTSFlowBatchItem:
    """One scheduler row to include in a batched dots DiT/flow step."""

    request_index: int
    fm_state: Any
    hidden_state: torch.Tensor
    generation_kwargs: dict[str, Any]


@dataclass
class DotsTTSBatchedAudioStepResult:
    """Batched dots audio step outputs aligned to scheduler request indices."""

    request_indices: list[int]
    latent_patches: list[torch.Tensor]
    feedback_embeddings: list[torch.Tensor]
    eos_scores: list[torch.Tensor]


def as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def torch_dtype(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    return torch.float32


__all__ = [
    "DotsTTSAudioStepResult",
    "DotsTTSBatchedAudioStepResult",
    "DotsTTSFlowBatchItem",
    "DotsTTSFlowBatchKey",
    "DotsTTSPreparedInputs",
    "as_tensor",
    "torch_dtype",
]
