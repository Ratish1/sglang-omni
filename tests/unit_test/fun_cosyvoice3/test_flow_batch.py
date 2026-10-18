# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang_omni.models.fun_cosyvoice3.stages as stages
from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedRows
from sglang_omni.models.fun_cosyvoice3.stages import (
    FLOW_EULER_STEPS,
    FlowBatchInput,
    FunCosyVoice3Flow,
    pack_flow_inputs,
)


class RecordingEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, torch.Tensor | bool]] = []

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        *,
        streaming: bool,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "x": x.detach().clone(),
                "mask": mask.detach().clone(),
                "mu": mu.detach().clone(),
                "t": t.detach().clone(),
                "spks": spks.detach().clone(),
                "cond": cond.detach().clone(),
                "streaming": streaming,
            }
        )
        return (0.1 * x + mu + spks.unsqueeze(-1) + cond) * mask


class RecordingPackedEstimator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.is_ragged = False
        self.is_compiled = False
        self.build_modulations(
            torch.linspace(0, 1, FLOW_EULER_STEPS + 1), torch.float32
        )

    def build_modulations(self, time_span: torch.Tensor, dtype: torch.dtype) -> None:
        step_count = len(time_span) - 1
        self.block_modulations = ((),) * step_count
        self.final_modulations = (torch.zeros(1, dtype=dtype),) * step_count

    def row_attention(
        self, rows: PackedRows, *, streaming: bool, dtype: torch.dtype
    ) -> SimpleNamespace:
        return SimpleNamespace(rows=rows, streaming=streaming, dtype=dtype)

    def rope_angles(self, frame_count: int) -> torch.Tensor:
        return torch.arange(frame_count, dtype=torch.float32).view(1, -1, 1)

    def forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        block_modulations: tuple[torch.Tensor, ...],
        final_modulation: torch.Tensor,
        rows: PackedRows,
        attention: SimpleNamespace,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        self.calls.append({"lengths": rows.lengths, "streaming": attention.streaming})
        positions = rows.positions.to(x.dtype).view(1, -1, 1)
        return 0.1 * x + mu + spks + cond + 0.01 * positions


def make_packed(flow: FakeFlow) -> FunCosyVoice3Flow:
    return FunCosyVoice3Flow(flow, packed_estimator=flow.packed_estimator)


class RecordingTRTEstimator:
    def __init__(self, max_batch: int = 16) -> None:
        self.max_batch = max_batch
        self.calls: list[dict[str, torch.Tensor]] = []

    def execute(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(
            {
                "x": x.detach().clone(),
                "mask": mask.detach().clone(),
                "mu": mu.detach().clone(),
                "t": t.detach().clone(),
                "spks": spks.detach().clone(),
                "cond": cond.detach().clone(),
            }
        )
        return (0.1 * x + mu + spks.unsqueeze(-1) + cond) * mask


class FakeDecoder:
    def __init__(
        self,
        channels: int,
        *,
        max_frames: int = 64,
        estimator: object | None = None,
    ) -> None:
        self.rand_noise = (
            torch.arange(channels * max_frames, dtype=torch.float32).reshape(
                1, channels, max_frames
            )
            / 100
        )
        self.t_scheduler = "cosine"
        self.inference_cfg_rate = 0.7
        self.estimator = estimator or RecordingEstimator()

    def forward_estimator(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        *,
        streaming: bool,
    ) -> torch.Tensor:
        return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)


class FakeFlow(torch.nn.Module):
    def __init__(
        self,
        *,
        channels: int = 4,
        token_mel_ratio: int = 2,
        max_frames: int = 64,
        estimator: object | None = None,
    ) -> None:
        super().__init__()
        self.output_size = channels
        self.token_mel_ratio = token_mel_ratio
        self.input_embedding = torch.nn.Embedding(32, channels)
        self.spk_embed_affine_layer = torch.nn.Linear(3, channels, bias=False)
        self.pre_lookahead_layer = lambda x, context=None: x
        self.pre_lookahead_len = 3
        self.decoder = FakeDecoder(channels, max_frames=max_frames, estimator=estimator)
        self.packed_estimator = RecordingPackedEstimator()
        with torch.no_grad():
            self.input_embedding.weight.copy_(
                torch.arange(32 * channels, dtype=torch.float32).reshape(32, channels)
                / 50
            )
            self.spk_embed_affine_layer.weight.copy_(
                torch.arange(channels * 3, dtype=torch.float32).reshape(channels, 3)
                / 20
            )


def make_input(
    token: list[int],
    *,
    prompt_token: list[int] | None = None,
    prompt_value: float = 0.0,
    embedding: tuple[float, float, float] = (1.0, 2.0, 3.0),
    channels: int = 4,
) -> FlowBatchInput:
    prompt_token = prompt_token or []
    return FlowBatchInput(
        token=torch.tensor([token], dtype=torch.int32),
        prompt_token=torch.tensor([prompt_token], dtype=torch.int32),
        prompt_feat=torch.full((1, len(prompt_token) * 2, channels), prompt_value),
        embedding=torch.tensor([embedding]),
    )


def infer_flow(flow: FakeFlow, inputs: list[FlowBatchInput]) -> list[torch.Tensor]:
    return FunCosyVoice3Flow(flow).inference(inputs)


def test_pack_flow_inputs_keeps_prompt_and_target_contiguous() -> None:
    flow = FakeFlow()
    items = [
        make_input([0, 8, 0], prompt_token=[4]),
        make_input([0], prompt_token=[5, 6, 7]),
    ]

    packed = pack_flow_inputs(flow, items)

    assert packed.token.dtype == torch.int32
    assert packed.token.tolist() == [[4, 0, 8, 0], [5, 6, 7, 0]]
    assert packed.combined_token_lengths == (4, 4)
    assert packed.prompt_token_lengths == (1, 3)
    assert packed.target_token_lengths == (3, 1)
    assert packed.combined_token_lengths_tensor.tolist() == [4, 4]
    assert packed.token_mask.squeeze(-1).tolist() == [
        [True, True, True, True],
        [True, True, True, True],
    ]


def test_pack_flow_inputs_builds_variable_length_token_masks() -> None:
    packed = pack_flow_inputs(
        FakeFlow(),
        [
            make_input([0], prompt_token=[]),
            make_input([3, 0], prompt_token=[4, 0]),
        ],
    )

    assert packed.token.tolist() == [[0, 0, 0, 0], [4, 0, 3, 0]]
    assert packed.token_mask.squeeze(-1).tolist() == [
        [True, False, False, False],
        [True, True, True, True],
    ]
    assert packed.prompt_mel_lengths == (0, 4)
    assert packed.total_mel_lengths == (2, 8)


def test_flow_batch_builds_variable_length_mel_masks_and_conditions() -> None:
    flow = FakeFlow()
    items = [
        make_input([1], prompt_token=[], prompt_value=3.0),
        make_input([2, 3], prompt_token=[4, 5], prompt_value=7.0),
    ]

    infer_flow(flow, items)

    first = flow.decoder.estimator.calls[0]
    mask = first["mask"][:2]
    cond = first["cond"][:2]
    assert mask[:, 0].bool().tolist() == [
        [True, True, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True],
    ]
    assert torch.count_nonzero(cond[0]) == 0
    torch.testing.assert_close(cond[1, :, :4], torch.full((4, 4), 7.0))
    assert torch.count_nonzero(cond[1, :, 4:]) == 0


@pytest.mark.parametrize("enable_buffered_packed_flow", [False, True])
def test_buffered_flow_route_is_explicit_and_preserves_split_order(
    monkeypatch: pytest.MonkeyPatch,
    enable_buffered_packed_flow: bool,
) -> None:
    flow = FakeFlow(max_frames=128)
    packed_flow = FunCosyVoice3Flow(flow, packed_estimator=flow.packed_estimator)
    assert packed_flow.enable_buffered_packed_flow is False
    packed_flow.enable_buffered_packed_flow = enable_buffered_packed_flow
    items = [
        make_input([1], prompt_token=[2], prompt_value=0.5),
        make_input([3, 4], prompt_token=[5, 6], prompt_value=1.5),
    ]
    native_generate = Mock(wraps=stages.generate_flow)
    packed_generate = Mock(wraps=stages.generate_flow_packed)
    monkeypatch.setattr(stages, "generate_flow", native_generate)
    monkeypatch.setattr(stages, "generate_flow_packed", packed_generate)

    generated = packed_flow.inference(items)

    expected_shapes = [(1, 4, 2), (1, 4, 4)]
    assert [tuple(mel.shape) for mel in generated] == expected_shapes
    if enable_buffered_packed_flow:
        packed_generate.assert_called_once()
        assert packed_generate.call_args.kwargs == {
            "streaming": False,
            "finalize": True,
        }
        native_generate.assert_not_called()
    else:
        packed_generate.assert_not_called()
        native_generate.assert_called_once()


@pytest.mark.parametrize("has_graph_runner", [False, True])
def test_buffered_packed_flow_inference_uses_packed_eager_without_native_retry(
    monkeypatch: pytest.MonkeyPatch,
    has_graph_runner: bool,
) -> None:
    flow = FakeFlow(max_frames=128)
    packed_flow = FunCosyVoice3Flow(flow, packed_estimator=flow.packed_estimator)
    packed_flow.enable_buffered_packed_flow = True
    assert flow.packed_estimator.is_compiled is False

    graph_runner: Mock | None = None
    if has_graph_runner:
        graph_runner = Mock()
        graph_runner.run.return_value = None
        packed_flow.whole_history_cuda_graph_runner = graph_runner
    else:
        pass
    packed_solve = Mock(wraps=stages.solve_flow_euler_packed)
    native_solve = Mock(
        side_effect=AssertionError("Packed graph miss retried native Flow")
    )
    monkeypatch.setattr(stages, "solve_flow_euler_packed", packed_solve)
    monkeypatch.setattr(stages, "solve_flow_euler", native_solve)

    inputs = [
        make_input([1], prompt_token=[2]),
        make_input([3, 4], prompt_token=[5, 6]),
    ]
    generated = packed_flow.inference(inputs)

    packed_solve.assert_called_once()
    native_solve.assert_not_called()
    assert [tuple(mel.shape) for mel in generated] == [(1, 4, 2), (1, 4, 4)]
    if graph_runner is not None:
        graph_runner.run.assert_called_once()
        assert graph_runner.run.call_args.kwargs["lengths"] == (4, 8)
    else:
        pass

    packed_failure = Mock(side_effect=RuntimeError("packed Flow failed"))
    monkeypatch.setattr(stages, "solve_flow_euler_packed", packed_failure)
    with pytest.raises(RuntimeError, match="packed Flow failed"):
        packed_flow.inference(inputs)


def test_buffered_packed_flow_inference_replays_whole_history_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow = FakeFlow(max_frames=128)
    packed_flow = FunCosyVoice3Flow(flow, packed_estimator=flow.packed_estimator)
    packed_flow.enable_buffered_packed_flow = True

    class GraphHit:
        def __init__(self) -> None:
            self.row_lengths: list[tuple[int, ...]] = []
            self.outputs: list[torch.Tensor] = []

        def run(
            self,
            *,
            noise: torch.Tensor,
            time_span: torch.Tensor,
            mu: torch.Tensor,
            speaker_embeddings: torch.Tensor,
            mel_conditioning: torch.Tensor,
            lengths: Sequence[int],
        ) -> torch.Tensor:
            assert time_span.ndim == 1
            assert mu.shape == noise.shape
            assert mel_conditioning.shape == noise.shape
            assert speaker_embeddings.shape[0] == len(lengths)
            self.row_lengths.append(tuple(lengths))
            generated = torch.arange(
                noise.numel(), dtype=noise.dtype, device=noise.device
            ).reshape_as(noise)
            self.outputs.append(generated)
            return generated

    graph_runner = GraphHit()
    packed_flow.whole_history_cuda_graph_runner = graph_runner
    packed_solve = Mock(side_effect=AssertionError("Packed Graph hit ran eager Flow"))
    native_solve = Mock(side_effect=AssertionError("Packed Graph hit ran native Flow"))
    monkeypatch.setattr(stages, "solve_flow_euler_packed", packed_solve)
    monkeypatch.setattr(stages, "solve_flow_euler", native_solve)

    inputs = [
        make_input([1], prompt_token=[2]),
        make_input([3, 4], prompt_token=[5, 6]),
    ]
    generated = packed_flow.inference(inputs)

    assert graph_runner.row_lengths == [(4, 8)]
    assert len(graph_runner.outputs) == 1
    packed_output = graph_runner.outputs[0]
    expected_row_lengths = (4, 8)
    expected_prompt_frame_counts = (2, 4)
    expected_packed_frame_count = sum(expected_row_lengths)
    expected_packed_output = torch.arange(
        expected_packed_frame_count * flow.output_size,
        dtype=packed_output.dtype,
        device=packed_output.device,
    ).reshape_as(packed_output)
    torch.testing.assert_close(packed_output, expected_packed_output)
    expected: list[torch.Tensor] = []
    start_frame = 0
    for total_frames, prompt_frames in zip(
        expected_row_lengths, expected_prompt_frame_counts, strict=True
    ):
        expected.append(
            expected_packed_output[
                :, start_frame + prompt_frames : start_frame + total_frames, :
            ].transpose(1, 2)
        )
        start_frame += total_frames

    assert [tuple(mel.shape) for mel in generated] == [(1, 4, 2), (1, 4, 4)]
    for actual, expected_mel in zip(generated, expected, strict=True):
        torch.testing.assert_close(actual, expected_mel)
    packed_solve.assert_not_called()
    native_solve.assert_not_called()
    assert flow.packed_estimator.calls == []


def test_flow_batch_cfg_uses_two_times_request_batch() -> None:
    flow = FakeFlow()

    infer_flow(
        flow,
        [make_input([1]), make_input([2]), make_input([3])],
    )

    for call in flow.decoder.estimator.calls:
        assert call["x"].shape[0] == 6
        assert call["mask"].shape[0] == 6
        assert call["mu"].shape[0] == 6
        # The ODE time is a scalar shared by the batch and by both CFG halves,
        # so only one row is materialised; the DiT broadcasts it.
        assert call["t"].shape[0] == 1
        assert call["spks"].shape[0] == 6
        assert call["cond"].shape[0] == 6
        assert call["streaming"] is False


def test_flow_batch_cfg_builds_conditional_and_unconditional_halves() -> None:
    flow = FakeFlow()
    items = [
        make_input([1], prompt_token=[2], prompt_value=2.0),
        make_input([3], prompt_token=[4], prompt_value=4.0),
    ]

    infer_flow(flow, items)

    first = flow.decoder.estimator.calls[0]
    torch.testing.assert_close(first["x"][:2], first["x"][2:])
    torch.testing.assert_close(first["mask"][:2], first["mask"][2:])
    assert torch.count_nonzero(first["mu"][:2]) > 0
    assert torch.count_nonzero(first["spks"][:2]) > 0
    assert torch.count_nonzero(first["cond"][:2]) > 0
    assert torch.count_nonzero(first["mu"][2:]) == 0
    assert torch.count_nonzero(first["spks"][2:]) == 0
    assert torch.count_nonzero(first["cond"][2:]) == 0


def test_flow_batch_matches_serial_reference_for_mixed_lengths() -> None:
    items = [
        make_input([1, 0], prompt_token=[2], prompt_value=0.5),
        make_input([3], prompt_token=[4, 5], prompt_value=1.5),
        make_input([6, 7, 8], prompt_token=[], prompt_value=2.5),
    ]
    serial = [infer_flow(FakeFlow(), [item])[0] for item in items]
    batched = infer_flow(FakeFlow(), items)

    assert [mel.shape for mel in batched] == [(1, 4, 4), (1, 4, 2), (1, 4, 6)]
    for actual, expected in zip(batched, serial, strict=True):
        torch.testing.assert_close(actual, expected)


def test_flow_batch_rejects_noise_buffer_overflow() -> None:
    flow = FakeFlow(max_frames=4)

    with pytest.raises(ValueError, match="rand_noise.*4.*6"):
        infer_flow(flow, [make_input([1, 2, 3])])


def test_flow_batch_rejects_prompt_alignment_mismatch() -> None:
    item = make_input([1], prompt_token=[2])
    item = FlowBatchInput(
        token=item.token,
        prompt_token=item.prompt_token,
        prompt_feat=torch.zeros(1, 1, 4),
        embedding=item.embedding,
    )

    with pytest.raises(ValueError, match="prompt feature length"):
        infer_flow(FakeFlow(), [item])


def test_flow_batch_tensorrt_matches_pytorch_serial() -> None:
    items = [
        make_input([1, 0], prompt_token=[2], prompt_value=0.5),
        make_input([3], prompt_token=[4, 5], prompt_value=1.5),
    ]
    serial = [infer_flow(FakeFlow(), [item])[0] for item in items]
    batched = infer_flow(FakeFlow(estimator=RecordingTRTEstimator(max_batch=2)), items)
    for actual, expected in zip(batched, serial, strict=True):
        torch.testing.assert_close(actual, expected)


def test_flow_causal_batch_uses_streaming_mask_and_strips_lookahead() -> None:
    flow = FakeFlow(max_frames=128)
    items = [
        make_input([1] * 28),
        make_input([2] * 28),
    ]
    mels = make_packed(flow).inference_causal(items)
    assert all(call["streaming"] is True for call in flow.packed_estimator.calls)
    assert flow.packed_estimator.calls[0]["lengths"] == (50, 50, 50, 50)
    assert mels[0].shape == (1, 4, 50)
    assert mels[1].shape == (1, 4, 50)


def test_flow_causal_batch_mixed_prompt_matches_serial() -> None:
    items = [
        make_input([1] * 78, prompt_token=[3] * 25),
        make_input([2] * 78, prompt_token=[4] * 50),
    ]
    serial = [
        make_packed(FakeFlow(max_frames=512)).inference_causal([item])[0]
        for item in items
    ]
    batched = make_packed(FakeFlow(max_frames=512)).inference_causal(items)
    assert batched[0].shape == serial[0].shape
    assert batched[1].shape == serial[1].shape
    for actual, expected in zip(batched, serial, strict=True):
        torch.testing.assert_close(actual, expected)
