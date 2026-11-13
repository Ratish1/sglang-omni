"""Preserve captured native rotary geometry in operand and recurrence controls."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
from torch.utils.hooks import RemovableHandle
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerTextConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeTalkerRotaryEmbedding,
    Qwen3OmniMoeThinkerTextRMSNorm,
    apply_rotary_pos_emb,
)

from scripts.debug_qwen3_omni_native_weights import (
    NativeBufferRuntime,
    NativeRotaryModule,
)
from sglang_omni.models.qwen3_omni.hf_config import normalize_rope_scaling


class NativeRotaryOperation:
    def __init__(
        self,
        configuration: Qwen3OmniMoeTalkerTextConfig,
        capture_directory: Path,
        device: torch.device,
    ) -> None:
        scaling = normalize_rope_scaling(configuration.rope_parameters)
        assert scaling is not None
        sections = scaling["mrope_section"]
        assert isinstance(sections, list) and all(
            isinstance(size, int) for size in sections
        )
        interleaved = scaling.get("mrope_interleaved", False)
        interleaved_glm = scaling.get("mrope_interleaved_glm", False)
        assert isinstance(interleaved, bool) and isinstance(interleaved_glm, bool)
        assert not interleaved_glm
        buffers = torch.load(
            capture_directory / "native-talker-buffers.pt",
            map_location=device,
            weights_only=True,
        )
        prefix = "model.layers.0.self_attn.rotary_emb"
        runtime = NativeBufferRuntime.model_validate_json(
            (capture_directory / "native-talker-buffer-runtime.json").read_bytes()
        ).rotary_modules[prefix]
        self.cache: torch.Tensor = buffers[f"{prefix}.cos_sin_cache"]
        self.axis_map: torch.Tensor = buffers[f"{prefix}.axis_map"]
        self.head_dimensions: int = configuration.head_dim
        self.sections: list[int] = sections
        self.interleaved: bool = interleaved
        self.runtime: NativeRotaryModule = runtime
        assert runtime.class_name == "MRotaryEmbedding"
        assert runtime.is_neox_style and not runtime.use_fallback_kernel
        assert runtime.head_size == runtime.rotary_dim == self.head_dimensions
        assert (
            runtime.position_start == 0
            and runtime.base == configuration.rope_parameters["rope_theta"]
        )
        assert (
            self.cache.dtype == torch.bfloat16
            and self.cache.shape[1] == self.head_dimensions
        )
        expected_axes = torch.zeros_like(self.axis_map)
        if interleaved:
            for axis in (1, 2):
                expected_axes[axis : sections[axis] * 3 : 3] = axis
        else:
            expected_axes = torch.repeat_interleave(
                torch.arange(3, device=device), torch.tensor(sections, device=device)
            )
        assert torch.equal(self.axis_map, expected_axes)

    def apply(
        self, query: torch.Tensor, key: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert query.ndim == key.ndim == 2 and positions.shape == (3, len(query))
        assert query.dtype == key.dtype == self.cache.dtype
        assert query.device == key.device == positions.device == self.cache.device
        assert int(positions.min()) >= 0 and int(positions.max()) < len(self.cache)
        query_dimensions, key_dimensions = query.shape[1], key.shape[1]
        packed = torch.zeros(
            (len(query), query_dimensions + 2 * key_dimensions),
            device=query.device,
            dtype=query.dtype,
        )
        native_query, native_key, _ = packed.split(
            (query_dimensions, key_dimensions, key_dimensions), dim=-1
        )
        native_query.copy_(query)
        native_key.copy_(key)
        triton_mrope_fused(
            native_query,
            native_key,
            self.cache,
            positions,
            self.sections,
            self.head_dimensions,
            self.head_dimensions,
            self.interleaved,
            False,
            True,
            None,
        )
        return native_query, native_key


class NativeBackboneRotary:
    def __init__(
        self,
        reference: Qwen3OmniMoeTalkerForConditionalGeneration,
        capture_directory: Path,
    ) -> None:
        self.reference: Qwen3OmniMoeTalkerForConditionalGeneration = reference
        self.operation: NativeRotaryOperation = NativeRotaryOperation(
            reference.config.text_config, capture_directory, reference.device
        )
        self.enabled: bool = False
        self.calls: int = 0
        self.position_calls: int = 0
        self.active_layer: int | None = None
        self.positions: torch.Tensor | None = None
        self.handles: list[RemovableHandle] = []

    def install(self) -> None:
        assert modeling_qwen3_omni_moe.apply_rotary_pos_emb is apply_rotary_pos_emb
        self.handles.append(
            self.reference.model.rotary_emb.register_forward_hook(
                self.capture_positions
            )
        )
        for layer_index, layer in enumerate(self.reference.model.layers):
            self.handles.append(
                layer.self_attn.q_norm.register_forward_hook(
                    partial(self.activate, layer_index=layer_index)
                )
            )
        modeling_qwen3_omni_moe.apply_rotary_pos_emb = self.forward

    def capture_positions(
        self,
        module: Qwen3OmniMoeTalkerRotaryEmbedding,
        inputs: tuple[torch.Tensor, torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if self.enabled:
            assert module is self.reference.model.rotary_emb
            assert self.positions is None and self.active_layer is None
            assert inputs[1].shape == (3, 1, inputs[0].shape[1])
            assert output[0].dtype == output[1].dtype == torch.bfloat16
            self.positions = inputs[1][:, 0].contiguous()
            self.position_calls += 1
        else:
            pass

    def activate(
        self,
        module: Qwen3OmniMoeThinkerTextRMSNorm,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
        *,
        layer_index: int,
    ) -> None:
        if self.enabled:
            assert module is self.reference.model.layers[layer_index].self_attn.q_norm
            assert inputs[0].shape == output.shape and self.active_layer is None
            assert layer_index == self.calls % len(self.reference.model.layers)
            self.active_layer = layer_index
        else:
            pass

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled or self.active_layer is None:
            return apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim)
        else:
            pass
        assert unsqueeze_dim == 1 and q.shape[0] == k.shape[0] == 1
        assert self.positions is not None and cos.shape == sin.shape
        query = q.transpose(1, 2).reshape(q.shape[2], -1)
        key = k.transpose(1, 2).reshape(k.shape[2], -1)
        native_query, native_key = self.operation.apply(query, key, self.positions)
        self.calls += 1
        self.active_layer = None
        if self.calls % len(self.reference.model.layers) == 0:
            self.positions = None
        else:
            pass
        return native_query.reshape(1, q.shape[2], q.shape[1], q.shape[3]).transpose(
            1, 2
        ), native_key.reshape(1, k.shape[2], k.shape[1], k.shape[3]).transpose(1, 2)
