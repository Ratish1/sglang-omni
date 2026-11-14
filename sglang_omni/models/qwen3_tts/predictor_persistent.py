# SPDX-License-Identifier: Apache-2.0
"""The code predictor's decoder layers as one persistent kernel: weight packing, CTA schedule and launch."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import torch
from sglang.kernels.jit.utils import is_arch_support_pdl, load_jit, make_cpp_args
from tvm_ffi.module import Module

PREDICTOR_LAYER_SOURCE = (
    Path(__file__).resolve().parent / "csrc" / "predictor_layer.cuh"
)
HIDDEN_SIZE = 1024
QKV_WIDTH = 4096
ATTENTION_WIDTH = 2048
INTERMEDIATE_SIZE = 3072
TILE_FEATURES = 16
CHUNK_DEPTH = 1024
KSTEP_DEPTH = 16
MAX_ROWS = 16
KEY_VALUE_HEADS = 8
SCHEDULE_TILE_BITS = 12
PHASE_QKV = 0
# Phases in launch order with their work item counts and weight chunks a tile.
QKV_TILES = QKV_WIDTH // TILE_FEATURES
OUTPUT_PROJECTION_TILES = HIDDEN_SIZE // TILE_FEATURES
GATE_UP_TILES = INTERMEDIATE_SIZE // (TILE_FEATURES // 2)
DOWN_TILES = HIDDEN_SIZE // TILE_FEATURES


@dataclass(frozen=True, kw_only=True)
class PredictorLayerWeights:
    """Weights as bulk-copy chunk images and norm weights: of one layer, or stacked with the layer first."""

    qkv_chunks: torch.Tensor
    output_projection_chunks: torch.Tensor
    gate_up_chunks: torch.Tensor
    down_chunks: torch.Tensor
    input_norm_weight: torch.Tensor
    post_norm_weight: torch.Tensor
    query_norm_weight: torch.Tensor
    key_norm_weight: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class PredictorSchedule:
    """Each CTA's work items in launch order, encoded phase << 12 | tile."""

    items: torch.Tensor
    lengths: torch.Tensor


def pack_weight_chunks(weight: torch.Tensor) -> torch.Tensor:
    """Chunks of 16 output features by 1024 input features, each laid out [k-step 64][half 2][feature 16][8], the order ldmatrix reads."""
    output_features, input_features = weight.shape
    tiles = output_features // TILE_FEATURES
    chunks = input_features // CHUNK_DEPTH
    return (
        weight.reshape(tiles, TILE_FEATURES, chunks, CHUNK_DEPTH // KSTEP_DEPTH, 2, 8)
        .permute(0, 2, 3, 4, 1, 5)
        .reshape(tiles * chunks, TILE_FEATURES * CHUNK_DEPTH)
        .contiguous()
    )


def pack_predictor_layer(
    *,
    qkv_weight: torch.Tensor,
    output_projection_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    input_norm_weight: torch.Tensor,
    post_norm_weight: torch.Tensor,
    query_norm_weight: torch.Tensor,
    key_norm_weight: torch.Tensor,
) -> PredictorLayerWeights:
    gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)
    # note (ratish): a tile holds 8 gate features and the same 8 up features, so one thread's
    # MMA fragment has both halves of each activation.
    interleaved_gate_up = torch.cat(
        (
            gate_weight.reshape(GATE_UP_TILES, 8, HIDDEN_SIZE),
            up_weight.reshape(GATE_UP_TILES, 8, HIDDEN_SIZE),
        ),
        dim=1,
    ).reshape(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    return PredictorLayerWeights(
        qkv_chunks=pack_weight_chunks(qkv_weight),
        output_projection_chunks=pack_weight_chunks(output_projection_weight),
        gate_up_chunks=pack_weight_chunks(interleaved_gate_up),
        down_chunks=pack_weight_chunks(down_weight),
        input_norm_weight=input_norm_weight.contiguous(),
        post_norm_weight=post_norm_weight.contiguous(),
        query_norm_weight=query_norm_weight.contiguous(),
        key_norm_weight=key_norm_weight.contiguous(),
    )


def stack_predictor_layers(
    layers: list[PredictorLayerWeights],
) -> PredictorLayerWeights:
    """The layers' weights stacked in launch order, the form the kernel takes."""
    return PredictorLayerWeights(
        qkv_chunks=torch.stack([layer.qkv_chunks for layer in layers]),
        output_projection_chunks=torch.stack(
            [layer.output_projection_chunks for layer in layers]
        ),
        gate_up_chunks=torch.stack([layer.gate_up_chunks for layer in layers]),
        down_chunks=torch.stack([layer.down_chunks for layer in layers]),
        input_norm_weight=torch.stack([layer.input_norm_weight for layer in layers]),
        post_norm_weight=torch.stack([layer.post_norm_weight for layer in layers]),
        query_norm_weight=torch.stack([layer.query_norm_weight for layer in layers]),
        key_norm_weight=torch.stack([layer.key_norm_weight for layer in layers]),
    )


def build_predictor_schedule(
    cta_count: int, rows: int, device: torch.device
) -> PredictorSchedule:
    """Balance each phase on its own, since every phase waits for the last CTA of the one before:
    a tile goes to the CTA with the least work in that phase, then the fewest weight chunks so far.
    """
    phases = (
        (QKV_TILES, 1),
        (rows * KEY_VALUE_HEADS, 0),
        (OUTPUT_PROJECTION_TILES, 2),
        (GATE_UP_TILES, 1),
        (DOWN_TILES, 3),
    )
    chunk_loads = [0] * cta_count
    items: list[list[int]] = [[] for _ in range(cta_count)]
    for phase, (tile_count, chunks_per_tile) in enumerate(phases):
        phase_loads = [0] * cta_count
        for tile in range(tile_count):
            cta = min(
                range(cta_count),
                key=lambda index: (phase_loads[index], chunk_loads[index]),
            )
            phase_loads[cta] += max(chunks_per_tile, 1)
            chunk_loads[cta] += chunks_per_tile
            items[cta].append((phase << SCHEDULE_TILE_BITS) | tile)
    # The kernel's epoch counter moves on after CTA 0's last item, which must depend on every
    # CTA's qkv output so that every CTA has read the counter by then.
    assert all(
        any(item >> SCHEDULE_TILE_BITS == PHASE_QKV for item in cta_items)
        for cta_items in items
    ), "every CTA needs a qkv tile"
    assert items[0][-1] >> SCHEDULE_TILE_BITS != PHASE_QKV, "CTA 0 must end past qkv"
    width = max(len(cta_items) for cta_items in items)
    encoded = torch.zeros(cta_count, width, dtype=torch.int16)
    for cta, cta_items in enumerate(items):
        encoded[cta, : len(cta_items)] = torch.tensor(cta_items, dtype=torch.int16)
    return PredictorSchedule(
        items=encoded.to(device),
        lengths=torch.tensor(
            [len(cta_items) for cta_items in items], dtype=torch.int32, device=device
        ),
    )


@functools.cache
def load_predictor_layer_module(use_pdl: bool) -> Module:
    template_arguments = make_cpp_args(use_pdl)
    return load_jit(
        "qwen3_tts_predictor_layer",
        *template_arguments,
        cuda_files=[str(PREDICTOR_LAYER_SOURCE)],
        cuda_wrappers=[
            (
                "predictor_layer",
                f"qwen3_tts_predictor::PredictorLayer<{template_arguments}>::run",
            )
        ],
    )


class PersistentPredictorLayers:
    """Launches a stack of predictor layers for up to 16 rows as one kernel; owns the handoff words and
    the epoch counter, which every call reuses."""

    def __init__(
        self, device: torch.device, rope_cos_sin: torch.Tensor, norm_epsilon: float
    ) -> None:
        self.device = device
        self.rope_cos_sin = rope_cos_sin.contiguous()
        self.norm_epsilon = norm_epsilon
        self.cta_count = torch.cuda.get_device_properties(device).multi_processor_count
        # Each word holds two bf16 values and the epoch of the layer that wrote them; zeroed words
        # carry epoch 0, which no layer writes.
        self.qkv_words = torch.zeros(
            MAX_ROWS, QKV_WIDTH // 2, device=device, dtype=torch.uint64
        )
        self.attention_words = torch.zeros(
            MAX_ROWS, ATTENTION_WIDTH // 2, device=device, dtype=torch.uint64
        )
        self.middle_words = torch.zeros(
            MAX_ROWS, HIDDEN_SIZE // 2, device=device, dtype=torch.uint64
        )
        self.activation_words = torch.zeros(
            MAX_ROWS, INTERMEDIATE_SIZE // 2, device=device, dtype=torch.uint64
        )
        # Layers hand their output to the next through alternating halves.
        self.hidden_words = torch.zeros(
            2, MAX_ROWS, HIDDEN_SIZE // 2, device=device, dtype=torch.uint64
        )
        # The kernel advances it by the layer count every launch, so epochs never repeat.
        self.epoch_counter = torch.zeros(1, device=device, dtype=torch.uint32)
        self.schedules: dict[int, PredictorSchedule] = {}
        self.module = load_predictor_layer_module(is_arch_support_pdl())

    def schedule_for(self, rows: int) -> PredictorSchedule:
        schedule = self.schedules.get(rows)
        if schedule is None:
            schedule = build_predictor_schedule(self.cta_count, rows, self.device)
            self.schedules[rows] = schedule
        else:
            pass
        return schedule

    def forward(
        self,
        weights: PredictorLayerWeights,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        position: int,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> None:
        """output = the stacked layers applied in order to hidden_states (rows, 1024) at cache slot position.

        The caches are (layers, rows, slots, kv heads, head dim) views.
        """
        schedule = self.schedule_for(hidden_states.shape[0])
        self.module.predictor_layer(
            hidden_states,
            output,
            weights.qkv_chunks,
            weights.output_projection_chunks,
            weights.gate_up_chunks,
            weights.down_chunks,
            weights.input_norm_weight,
            weights.post_norm_weight,
            weights.query_norm_weight,
            weights.key_norm_weight,
            self.rope_cos_sin,
            key_cache,
            value_cache,
            self.qkv_words,
            self.attention_words,
            self.middle_words,
            self.activation_words,
            self.hidden_words,
            self.epoch_counter,
            schedule.items,
            schedule.lengths,
            position,
            self.norm_epsilon,
        )
