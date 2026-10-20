# SPDX-License-Identifier: Apache-2.0
"""Contracts of the Qwen3-TTS decode step around the code predictor.

The layer 0 token id is the only value the host reads from a step, so its
pinned copy is enqueued before the predictor and the finalize wait returns
while the predictor still runs. A step's launch enqueues the predictor, writes
each slot's feedback row and snapshots the frames; its resolve reads that
snapshot, so it holds when the next step launches before this one resolves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
from sglang_omni.scheduling.types import RequestOutput

EOS = 42
PREDICTOR_CYCLES = 1_000_000_000


@pytest.fixture(autouse=True)
def require_cuda_for_accelerator_tests(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("accelerator") and not torch.cuda.is_available():
        pytest.skip("token id staging on a stream needs CUDA")


class RecordedEvent:
    def __init__(self) -> None:
        self.is_recorded = False

    def record(self) -> None:
        self.is_recorded = True


def make_runner(code_predictor_forward, device: torch.device) -> Qwen3TTSModelRunner:
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.resolving_code_step = None
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    runner.model = SimpleNamespace(
        config=SimpleNamespace(codec_eos_token_id=EOS),
        code_predictor_forward=code_predictor_forward,
        output_codes=torch.zeros((4, 3), dtype=torch.long, device=device),
        output_embeds=torch.zeros((4, 2), device=device),
    )
    runner.feedback_pool = torch.zeros((8, 2), device=device)
    return runner


def step(ids: torch.Tensor):
    batch_size = ids.shape[0]
    result = SimpleNamespace(
        next_token_ids=ids,
        logits_output=SimpleNamespace(
            hidden_states=torch.ones((batch_size, 2), device=ids.device)
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        positions=torch.full((batch_size,), 3, dtype=torch.long, device=ids.device),
        req_pool_indices=torch.tensor([5, 2, 7][:batch_size], device=ids.device),
    )
    requests = [
        SimpleNamespace(
            request_id=f"r{row}",
            data=Qwen3TTSSGLangRequestData(
                req=SimpleNamespace(
                    inflight_middle_chunks=0,
                    is_retracted=False,
                    finished=lambda: False,
                )
            ),
        )
        for row in range(batch_size)
    ]
    return result, forward_batch, requests


def outputs_of(ids: torch.Tensor, requests) -> dict[str, RequestOutput]:
    return {
        req.request_id: RequestOutput(req.request_id, data=int(ids[row]))
        for row, req in enumerate(requests)
    }


def make_writing_runner() -> Qwen3TTSModelRunner:
    runner = make_runner(None, torch.device("cpu"))

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        rows = layer0_codes.shape[0]
        runner.model.output_codes[:rows] = layer0_codes + torch.arange(3)
        runner.model.output_embeds[:rows] = layer0_codes.to(torch.float32)

    runner.model.code_predictor_forward = code_predictor_forward
    return runner


def test_a_step_stages_its_ids_before_the_predictor_and_feeds_each_slot():
    ids = torch.tensor([7, EOS, 9], dtype=torch.long)
    result, forward_batch, requests = step(ids)
    staged_when_called = []
    runner = make_writing_runner()
    write_codes = runner.model.code_predictor_forward

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        staged_when_called.append(
            result._host_token_ids
        )  # noqa: leading-underscore  # production name
        write_codes(layer0_codes, hidden, semantic_positions)

    runner.model.code_predictor_forward = code_predictor_forward

    runner.post_decode(result, forward_batch, object(), requests)

    assert len(staged_when_called) == 1
    assert staged_when_called[0] is ids
    assert runner.feedback_pool[[5, 2, 7]].tolist() == [
        [7.0, 7.0],
        [42.0, 42.0],
        [9.0, 9.0],
    ]

    runner.post_process_outputs(
        result, SimpleNamespace(requests=requests), outputs_of(ids, requests)
    )

    assert [c.tolist() for c in requests[0].data.output_codes] == [[7, 8, 9]]
    assert requests[1].data.output_codes == []
    assert [c.tolist() for c in requests[2].data.output_codes] == [[9, 10, 11]]


def test_a_resolve_reads_the_frames_its_launch_snapshotted():
    ids = torch.tensor([7, 9], dtype=torch.long)
    result, forward_batch, requests = step(ids)
    runner = make_writing_runner()
    runner.codes_event_module = SimpleNamespace(Event=RecordedEvent)

    launched = runner.post_decode_launch(result, forward_batch, requests)
    runner.model.output_codes.fill_(-1)
    runner.post_decode_resolve(launched, result, forward_batch, object(), requests)
    runner.post_process_outputs(
        result, SimpleNamespace(requests=requests), outputs_of(ids, requests)
    )

    assert [c.tolist() for c in requests[0].data.output_codes] == [[7, 8, 9]]
    assert [c.tolist() for c in requests[1].data.output_codes] == [[9, 10, 11]]
    assert launched.codes_ready.is_recorded
    assert requests[0].data.codes_ready_event is launched.codes_ready


def test_a_request_finished_or_retracted_after_its_step_launched_gets_no_frame():
    ids = torch.tensor([7, 8, 9], dtype=torch.long)
    result, forward_batch, requests = step(ids)
    runner = make_writing_runner()

    launched = runner.post_decode_launch(result, forward_batch, requests)
    requests[0].data.req.finished = lambda: True
    requests[1].data.req.is_retracted = True
    runner.post_decode_resolve(launched, result, forward_batch, object(), requests)
    runner.post_process_outputs(
        result, SimpleNamespace(requests=requests), outputs_of(ids, requests)
    )

    assert requests[0].data.output_codes == []
    assert requests[0].data.latest_stream_code_chunk is None
    assert requests[1].data.output_codes == []
    assert [c.tolist() for c in requests[2].data.output_codes] == [[9, 10, 11]]


def test_a_middle_prefill_chunk_adds_no_frame_or_stream_chunk():
    ids = torch.tensor([7, 9], dtype=torch.long)
    result, forward_batch, requests = step(ids)
    requests[0].data.req.inflight_middle_chunks = 1
    runner = make_writing_runner()

    runner.post_decode(result, forward_batch, object(), requests)
    runner.post_process_outputs(
        result, SimpleNamespace(requests=requests), outputs_of(ids, requests)
    )

    assert requests[0].data.output_codes == []
    assert requests[0].data.latest_stream_code_chunk is None
    assert [c.tolist() for c in requests[1].data.output_codes] == [[9, 10, 11]]
    assert requests[1].data.latest_stream_code_chunk is not None


def test_a_middle_prefill_chunk_does_not_advance_its_generation_step():
    _, _, requests = step(torch.tensor([7, 9], dtype=torch.long))
    requests[1].data.req.inflight_middle_chunks = 2
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)

    assert runner.finalize_skip_rids(SimpleNamespace(requests=requests)) == {"r1"}


@pytest.mark.accelerator
def test_staged_ids_resolve_while_the_predictor_is_still_running():
    device = torch.device("cuda")
    ids = torch.tensor([7, 8], dtype=torch.long, device=device)
    result, forward_batch, requests = step(ids)

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        torch.cuda._sleep(PREDICTOR_CYCLES)  # noqa: leading-underscore  # upstream name

    runner = make_runner(code_predictor_forward, device)
    stream = torch.cuda.current_stream(device)

    runner.post_decode(result, forward_batch, object(), requests)
    host_ids = runner.resolve_host_token_ids(result)
    predictor_done = stream.query()
    torch.cuda.synchronize()

    assert host_ids.tolist() == [7, 8]
    assert predictor_done is False


@pytest.mark.accelerator
def test_staged_ids_keep_the_sampled_values_when_later_stream_work_overwrites_them():
    device = torch.device("cuda")
    ids = torch.tensor([7, 8], dtype=torch.long, device=device)
    result, forward_batch, requests = step(ids)

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        layer0_codes.fill_(-1)

    runner = make_runner(code_predictor_forward, device)

    runner.post_decode(result, forward_batch, object(), requests)

    assert runner.resolve_host_token_ids(result).tolist() == [7, 8]
    assert result.next_token_ids.tolist() == [-1, -1]
