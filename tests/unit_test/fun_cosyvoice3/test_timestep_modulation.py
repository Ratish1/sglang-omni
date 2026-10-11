# SPDX-License-Identifier: Apache-2.0
"""Tests for the fixed-grid CosyVoice3 AdaLN modulation table."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.timestep_modulation import (
    TimestepModulationCache,
    apply_block_modulation,
    apply_final_modulation,
    euler_step_times,
)


class FakeTimestepEmbedding(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, channels)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        return self.linear(time[:, None])


class FakeAdaptiveLayerNorm(torch.nn.Module):
    def __init__(self, channels: int, modulation_parts: int) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(channels, channels * modulation_parts)
        self.norm = torch.nn.LayerNorm(channels, elementwise_affine=False)
        self.modulation_parts = modulation_parts

    def forward(
        self, hidden_states: torch.Tensor, timestep_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        modulation = self.linear(self.silu(timestep_embedding))
        if self.modulation_parts == 6:
            shift, scale, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(
                6, dim=1
            )
            return (
                self.norm(hidden_states) * (1 + scale[:, None]) + shift[:, None],
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            )
        else:
            scale, shift = modulation.chunk(2, dim=1)
            return (
                self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :],
            )


class FakeBlock(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attn_norm = FakeAdaptiveLayerNorm(channels, 6)


class FakeDiT(torch.nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.time_embed = FakeTimestepEmbedding(channels)
        self.transformer_blocks = torch.nn.ModuleList(
            [FakeBlock(channels) for _ in range(depth)]
        )
        self.norm_out = FakeAdaptiveLayerNorm(channels, 2)


def test_euler_step_times_preserves_the_solver_recurrence() -> None:
    time_span = torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float64)

    step_times = euler_step_times(time_span, flow_dtype=torch.float64, euler_steps=3)

    torch.testing.assert_close(
        torch.cat(step_times), torch.tensor([0.0, 0.1, 0.4], dtype=torch.float64)
    )


def test_modulation_cache_has_fixed_grid_shapes_and_values() -> None:
    torch.manual_seed(0)
    channels = 4
    depth = 2
    dit = FakeDiT(channels, depth).double().eval()
    time_span = torch.tensor([0.0, 0.1, 0.4, 1.0], dtype=torch.float64)

    cache = TimestepModulationCache.build(
        dit, time_span, flow_dtype=torch.float64, euler_steps=3
    )

    assert cache.block_modulations.shape == (3, depth, 1, 6 * channels)
    assert cache.final_modulations.shape == (3, 1, 2 * channels)
    assert cache.block_modulations.dtype == torch.float64
    assert cache.final_modulations.dtype == torch.float64
    for step, step_time in enumerate(
        euler_step_times(time_span, flow_dtype=torch.float64, euler_steps=3)
    ):
        timestep_embedding = dit.time_embed(step_time)
        expected_blocks = torch.stack(
            [
                block.attn_norm.linear(block.attn_norm.silu(timestep_embedding))
                for block in dit.transformer_blocks
            ]
        )
        expected_final = dit.norm_out.linear(dit.norm_out.silu(timestep_embedding))
        torch.testing.assert_close(
            cache.block_modulations[step], expected_blocks, rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.final_modulations[step], expected_final, rtol=0, atol=0
        )


def test_cached_modulation_helpers_match_adaptive_layer_norm() -> None:
    torch.manual_seed(1)
    channels = 4
    dit = FakeDiT(channels, depth=1).double().eval()
    hidden_states = torch.randn(1, 5, channels, dtype=torch.float64)
    timestep_embedding = dit.time_embed(torch.tensor([0.25], dtype=torch.float64))
    block = dit.transformer_blocks[0]
    block_modulation = block.attn_norm.linear(block.attn_norm.silu(timestep_embedding))
    final_modulation = dit.norm_out.linear(dit.norm_out.silu(timestep_embedding))

    expected_block = block.attn_norm(hidden_states, timestep_embedding)
    actual_block = apply_block_modulation(
        block.attn_norm, hidden_states, block_modulation
    )
    expected_final = dit.norm_out(hidden_states, timestep_embedding)[0]
    actual_final = apply_final_modulation(dit.norm_out, hidden_states, final_modulation)

    for actual, expected in zip(actual_block, expected_block, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_final, expected_final, rtol=0, atol=0)


def test_modulation_cache_rejects_a_different_schedule_or_model() -> None:
    dit = FakeDiT(channels=4, depth=1).double().eval()
    time_span = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    cache = TimestepModulationCache.build(
        dit, time_span, flow_dtype=torch.float64, euler_steps=2
    )

    cache.validate(dit, time_span.clone())
    with pytest.raises(RuntimeError, match="canonical schedule"):
        cache.validate(dit, torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="stale"):
        cache.validate(FakeDiT(channels=4, depth=1).double().eval(), time_span)
