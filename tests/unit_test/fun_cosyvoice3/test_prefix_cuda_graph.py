# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import (
    PrefixCudaGraphCache,
    PrefixCudaGraphEnvelope,
    resolve_prefix_cuda_graph_max_slack,
    route_prefix_cuda_graph_envelope,
)


def test_prefix_cuda_graph_capture_requires_input_factory() -> None:
    parameter = inspect.signature(PrefixCudaGraphCache.capture).parameters[
        "capture_input_factory"
    ]
    assert parameter.default is inspect.Parameter.empty


def test_prepare_prefix_cuda_graph_capture_inputs_uses_warmup_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameter = torch.nn.Parameter(torch.zeros(1))
    fake_flow = SimpleNamespace(
        token_mel_ratio=2,
        pre_lookahead_len=1,
        output_size=2,
        spk_embed_affine_layer=torch.nn.Linear(3, 2),
        parameters=lambda: iter((parameter,)),
    )
    fake_flow.flow = fake_flow

    warmup_token_counts: list[int] = []

    def make_warmup_flow_input(token_count: int) -> stages.FlowBatchInput:
        warmup_token_counts.append(token_count)
        return stages.FlowBatchInput(
            token=torch.full((1, token_count), 3, dtype=torch.int32),
            prompt_token=torch.full((1, 1), 2, dtype=torch.int32),
            prompt_feat=torch.full((1, 2, 2), 4.0),
            embedding=torch.full((1, 3), 5.0),
        )

    scheduler = SimpleNamespace(
        token_hop_len=1,
        make_warmup_flow_input=make_warmup_flow_input,
    )
    packed_batches: list[stages.PackedFlowBatch] = []
    original_pack_flow_inputs = stages.pack_flow_inputs

    def recording_pack_flow_inputs(
        flow_model: stages.FunCosyVoice3Flow,
        inputs: list[stages.FlowBatchInput],
    ) -> stages.PackedFlowBatch:
        packed = original_pack_flow_inputs(flow_model, inputs)
        packed_batches.append(packed)
        return packed

    monkeypatch.setattr(stages, "pack_flow_inputs", recording_pack_flow_inputs)

    token_condition = torch.arange(1, 25, dtype=torch.float32).reshape(2, 2, 6)
    noisy_mel = token_condition + 100
    prompt_mel = token_condition + 200
    speaker_embedding = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    time_span = torch.tensor([0.0, 0.5, 1.0])
    finalize_values: list[bool] = []

    def fake_prepare_flow_conditioning(
        flow_model: stages.FunCosyVoice3Flow,
        packed: stages.PackedFlowBatch,
        *,
        finalize: bool,
    ) -> stages.FlowConditioning:
        del flow_model, packed
        finalize_values.append(finalize)
        return stages.FlowConditioning(
            token_condition=token_condition,
            mel_lengths=(4, 6),
            speaker_embedding=speaker_embedding,
            prompt_mel=prompt_mel,
            noisy_mel=noisy_mel,
            time_span=time_span,
        )

    monkeypatch.setattr(
        stages, "prepare_flow_conditioning", fake_prepare_flow_conditioning
    )

    capture_inputs = stages.prepare_prefix_cuda_graph_capture_inputs(
        fake_flow,
        scheduler,
        (4, 6),
        device=torch.device("cpu"),
    )

    assert warmup_token_counts == [2, 3]
    assert len(packed_batches) == 1
    assert packed_batches[0].target_token_lengths == (2, 3)
    assert packed_batches[0].prompt_mel_lengths == (2, 2)
    assert finalize_values == [False]
    assert capture_inputs[0].shape == (1, 10, 2)
    assert capture_inputs[2].shape == (1, 10, 2)
    assert capture_inputs[4].shape == (1, 10, 2)
    assert torch.equal(capture_inputs[1], time_span)
    assert torch.equal(capture_inputs[3], speaker_embedding)
    assert torch.equal(
        capture_inputs[2],
        torch.cat(
            (
                token_condition[0, :, :4].transpose(0, 1),
                token_condition[1, :, :6].transpose(0, 1),
            ),
            dim=0,
        ).unsqueeze(0),
    )
    assert all(torch.count_nonzero(value) > 0 for value in capture_inputs)


@pytest.mark.parametrize(
    ("capture_shapes", "message"),
    [
        (((1, 100, 100, 512, (75,)),), "multiples"),
        (((1, 100, 100, 512, (50,)),), "sum to N"),
        (((2, 200, 100, 512, (100,)),), "equal B"),
    ],
)
def test_verify_prefix_cuda_graph_capture_shapes_rejects_invalid_overrides(
    capture_shapes: tuple[tuple[int, int, int, int, tuple[int, ...]], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        stages.verify_prefix_cuda_graph_capture_shapes(
            capture_shapes,
            chunk_frames=50,
        )


def test_prefix_cuda_graph_router_uses_supplied_chunk_frames() -> None:
    envelope = PrefixCudaGraphEnvelope("B1-N40-M40-E80", 1, 40, 40, 80, (40,))
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[40],
            total_frame_counts=[40],
            envelopes=(envelope,),
            chunk_frames=40,
            max_slack_frames=40,
        )
        == envelope
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=(envelope,),
            chunk_frames=40,
            max_slack_frames=40,
        )
        is None
    )


def test_prefix_cuda_graph_shape_validation_uses_supplied_chunk_frames() -> None:
    valid_shapes = ((2, 120, 80, 160, (80, 40)),)
    assert (
        stages.verify_prefix_cuda_graph_capture_shapes(
            valid_shapes,
            chunk_frames=40,
        )
        == valid_shapes
    )
    with pytest.raises(ValueError, match="chunk size 40"):
        stages.verify_prefix_cuda_graph_capture_shapes(
            ((2, 120, 100, 160, (100, 20)),),
            chunk_frames=40,
        )


def test_prefix_cuda_graph_router_uses_configured_max_slack() -> None:
    envelope = PrefixCudaGraphEnvelope("B1-N200-M200-E200", 1, 200, 200, 200, (200,))
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100],
            total_frame_counts=[100],
            envelopes=(envelope,),
            chunk_frames=50,
            max_slack_frames=50,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100],
            total_frame_counts=[100],
            envelopes=(envelope,),
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelope
    )


def test_prefix_cuda_graph_max_slack_defaults_to_two_chunks() -> None:
    assert resolve_prefix_cuda_graph_max_slack(50, None) == 100
    assert resolve_prefix_cuda_graph_max_slack(50, 50) == 50
    with pytest.raises(ValueError, match="greater than zero"):
        resolve_prefix_cuda_graph_max_slack(50, 0)
    with pytest.raises(ValueError, match="chunk size 50"):
        resolve_prefix_cuda_graph_max_slack(50, 75)


def test_prefix_cuda_graph_routes_physical_envelopes() -> None:
    envelopes = (
        PrefixCudaGraphEnvelope("B1-N100-M100-E512", 1, 100, 100, 512, (100,)),
        PrefixCudaGraphEnvelope("B2-N200-M100-E512", 2, 200, 100, 512, (100, 100)),
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[0]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50, 50],
            total_frame_counts=[50, 50],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50, 50],
            total_frame_counts=[50, 50],
            envelopes=(envelopes[0],),
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100, 50],
            total_frame_counts=[100, 50],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100, 100],
            total_frame_counts=[100, 100],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[1]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100],
            total_frame_counts=[100],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        == envelopes[0]
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[75],
            total_frame_counts=[75],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=(
                PrefixCudaGraphEnvelope("B1-N200-M200-E512", 1, 200, 200, 512, (200,)),
            ),
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[50],
            envelopes=(
                PrefixCudaGraphEnvelope("B1-N125-M100-E512", 1, 125, 100, 512, (125,)),
            ),
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[150],
            total_frame_counts=[150],
            envelopes=(
                PrefixCudaGraphEnvelope("B1-N200-M100-E512", 1, 200, 100, 512, (200,)),
            ),
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[50],
            total_frame_counts=[600],
            envelopes=envelopes,
            chunk_frames=50,
            max_slack_frames=100,
        )
        is None
    )
    assert (
        route_prefix_cuda_graph_envelope(
            new_frame_counts=[100, 100],
            total_frame_counts=[100, 100],
            envelopes=(
                PrefixCudaGraphEnvelope(
                    "B2-N200-M100-E512", 2, 200, 100, 512, (100, 100)
                ),
                PrefixCudaGraphEnvelope(
                    "B2-N300-M150-E1024", 2, 300, 150, 1024, (150, 150)
                ),
            ),
            chunk_frames=50,
            max_slack_frames=100,
        ).name
        == "B2-N200-M100-E512"
    )
