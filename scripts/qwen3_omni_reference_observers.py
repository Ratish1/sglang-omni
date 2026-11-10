# SPDX-License-Identifier: Apache-2.0
"""Owned snapshots of actual checkpoint-reference execution boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypedDict, Unpack

import torch
from transformers.cache_utils import Cache
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
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


class SDPAForwardKeywords(FlashAttentionKwargs, total=False):
    position_ids: torch.Tensor | None
    use_cache: bool | None
    output_router_logits: bool | None
    output_hidden_states: bool | None
    output_attentions: bool
    sliding_window: int | None
    is_causal: bool | None


class SDPACallConfiguration(TypedDict):
    dropout: float
    scaling: float | None
    is_causal: bool | None
    module_is_causal: bool
    key_value_groups: int
    keyword_names: list[str]


@dataclass(kw_only=True)
class SDPAPrefillObservation:
    attention_layers: dict[torch.nn.Module, int]
    tensor_calls: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    layouts: dict[int, dict[str, TensorLayout]] = field(default_factory=dict)
    configurations: dict[int, SDPACallConfiguration] = field(default_factory=dict)

    def __call__(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **keywords: Unpack[SDPAForwardKeywords],
    ) -> tuple[torch.Tensor, None]:
        layer_index = self.attention_layers.get(module)
        should_capture = (
            layer_index is not None and layer_index not in self.tensor_calls
        )
        if should_capture:
            tensors = {"query": query, "key": key, "value": value}
            if attention_mask is not None:
                tensors["attention_mask"] = attention_mask
            else:
                pass
            self.tensor_calls[layer_index] = {
                name: tensor.detach().cpu() for name, tensor in tensors.items()
            }
            self.layouts[layer_index] = {
                name: tensor_layout(tensor) for name, tensor in tensors.items()
            }
            assert isinstance(module, Qwen3OmniMoeThinkerTextAttention)
            self.configurations[layer_index] = {
                "dropout": dropout,
                "scaling": scaling,
                "is_causal": keywords.get("is_causal"),
                "module_is_causal": module.is_causal,
                "key_value_groups": module.num_key_value_groups,
                "keyword_names": sorted(keywords),
            }
            torch.cuda.nvtx.range_push(f"public_sdpa_prefill:{layer_index}")
        else:
            pass
        output = sdpa_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout=dropout,
            scaling=scaling,
            **keywords,
        )
        if should_capture:
            torch.cuda.nvtx.range_pop()
            self.tensor_calls[layer_index]["output"] = output[0].detach().cpu()
            self.layouts[layer_index]["output"] = tensor_layout(output[0])
        else:
            pass
        return output
