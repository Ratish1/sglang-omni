# SPDX-License-Identifier: Apache-2.0
"""Owned snapshots of actual checkpoint-reference execution boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeThinkerTextAttention,
)


class TensorLayout(TypedDict):
    dtype: str
    shape: list[int]
    strides_in_elements: list[int]
    storage_offset_in_elements: int
    address_bytes: int


def tensor_layout(tensor: torch.Tensor) -> TensorLayout:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "strides_in_elements": list(tensor.stride()),
        "storage_offset_in_elements": tensor.storage_offset(),
        "address_bytes": tensor.data_ptr(),
    }


@dataclass(kw_only=True)
class LayerObservation:
    family: Literal["audio", "vision", "thinker", "talker", "predictor", "vocoder"]
    layer_index: int
    calls: int = 0
    last_output: torch.Tensor | None = None

    def __call__(
        self,
        module: torch.nn.Module,
        arguments: tuple[torch.Tensor | None, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        self.calls += 1
        hidden_states = output[0] if isinstance(output, tuple) else output
        self.last_output = hidden_states.detach()


@dataclass(kw_only=True)
class CodecInputObservation:
    codec_chunks: list[torch.Tensor] = field(default_factory=list)

    def __call__(
        self,
        module: Qwen3OmniMoeCode2Wav,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        codec_ids = inputs[0]
        assert codec_ids.shape[1] == module.config.num_quantizers
        self.codec_chunks.append(codec_ids.detach().cpu())


@dataclass(kw_only=True)
class TensorCallObservation:
    operation_name: str
    first_call_only: bool = False
    tensor_calls: list[dict[str, torch.Tensor]] = field(default_factory=list)
    layout_calls: list[dict[str, TensorLayout]] = field(default_factory=list)

    def __call__(
        self,
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        if self.first_call_only and self.tensor_calls:
            return
        else:
            pass
        assert not module.training
        if isinstance(output, torch.Tensor):
            output_tensors = (output,)
        else:
            output_tensors = output
        self.tensor_calls.append(
            {
                **{
                    f"input_{index}": tensor.detach().cpu()
                    for index, tensor in enumerate(inputs)
                },
                **{
                    f"output_{index}": tensor.detach().cpu()
                    for index, tensor in enumerate(output_tensors)
                },
                "cuda_rng_state": torch.cuda.get_rng_state(inputs[0].device),
            }
        )
        if self.first_call_only:
            self.layout_calls.append(
                {
                    **{
                        f"input_{index}": tensor_layout(tensor)
                        for index, tensor in enumerate(inputs)
                    },
                    **{
                        f"output_{index}": tensor_layout(tensor)
                        for index, tensor in enumerate(output_tensors)
                    },
                }
            )
        else:
            pass


class AttentionForwardKeywords(TypedDict):
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    past_key_values: Cache | None
    use_cache: bool | None


class AttentionInputSnapshot(TypedDict):
    attention_mask: torch.Tensor | None
    position_ids: torch.Tensor | None
    cosine: torch.Tensor
    sine: torch.Tensor
    past_token_count: int
    cache_class: str
    use_cache: bool | None


@dataclass(kw_only=True)
class PrefillAttentionObservation:
    attention: Qwen3OmniMoeThinkerTextAttention
    snapshot: AttentionInputSnapshot | None = None

    def __call__(
        self,
        module: Qwen3OmniMoeThinkerTextAttention,
        inputs: tuple[torch.Tensor, ...],
        keywords: AttentionForwardKeywords,
    ) -> None:
        if self.snapshot is not None:
            return
        else:
            pass
        assert not inputs and not module.training
        attention_mask = keywords["attention_mask"]
        position_ids = keywords["position_ids"]
        cosine, sine = keywords["position_embeddings"]
        cache = keywords["past_key_values"]
        self.snapshot = {
            "attention_mask": (
                attention_mask.detach().cpu() if attention_mask is not None else None
            ),
            "position_ids": (
                position_ids.detach().cpu() if position_ids is not None else None
            ),
            "cosine": cosine.detach().cpu(),
            "sine": sine.detach().cpu(),
            "past_token_count": (
                cache.get_seq_length(module.layer_idx) if cache is not None else 0
            ),
            "cache_class": type(cache).__name__,
            "use_cache": keywords["use_cache"],
        }
