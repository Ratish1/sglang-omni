# SPDX-License-Identifier: Apache-2.0
"""Contracts of the Qwen3-TTS decode step around the code predictor.

The layer 0 token id is the only value the host reads from a step, so its
pinned copy is enqueued before the predictor and the finalize wait returns
while the predictor still runs. The code rows keep coming from a snapshot taken
behind the predictor, and each request slot's feedback row is written right after
it. The runner is sync only: its collect has no launch and resolve halves, so it
never takes the lookahead path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
from sglang_omni.scheduling.types import RequestOutput
from tests.unit_test.fixtures.qwen3_tts_code_table import attach_code_table

EOS = 42
PREDICTOR_CYCLES = 1_000_000_000


@pytest.fixture(autouse=True)
def require_cuda_for_accelerator_tests(request: pytest.FixtureRequest):
    if request.node.get_closest_marker("accelerator") and not torch.cuda.is_available():
        pytest.skip("token id staging on a stream needs CUDA")


def make_runner(
    code_predictor_forward, device: torch.device, requests
) -> Qwen3TTSModelRunner:
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.has_pending_code_step = False
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    runner.token_id_events = []
    runner.token_id_event_slot = 0
    runner.model = SimpleNamespace(
        config=SimpleNamespace(codec_eos_token_id=EOS),
        code_predictor_forward=code_predictor_forward,
        output_codes=torch.zeros((4, 3), dtype=torch.long, device=device),
        output_embeds=torch.zeros((4, 2), device=device),
        prepare_decode_buffers=lambda requests: True,
    )
    runner.slot_feedback_rows = torch.zeros((8, 2), device=device)
    attach_code_table(runner, requests, row_frames=4, codebook_count=3, device=device)
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
                    rid=f"r{row}", inflight_middle_chunks=0, origin_input_ids=[0] * 4
                )
            ),
        )
        for row in range(batch_size)
    ]
    return result, forward_batch, requests


def test_collect_codes_stages_the_ids_before_the_predictor_runs():
    ids = torch.tensor([7, EOS, 9], dtype=torch.long)
    result, forward_batch, requests = step(ids)
    staged_when_called = []

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        staged_when_called.append(
            result._host_token_ids
        )  # noqa: leading-underscore  # production name
        runner.model.output_codes[:3] = layer0_codes + torch.arange(3)
        runner.model.output_embeds[:3] = layer0_codes.to(torch.float32)

    runner = make_runner(code_predictor_forward, torch.device("cpu"), requests)

    runner.collect_codes(result, forward_batch, object(), requests)

    assert len(staged_when_called) == 1
    assert staged_when_called[0] is ids
    assert runner.slot_feedback_rows[[5, 2, 7]].tolist() == [
        [7.0, 7.0],
        [42.0, 42.0],
        [9.0, 9.0],
    ]

    outputs = {
        req.request_id: RequestOutput(req.request_id, data=int(ids[row]))
        for row, req in enumerate(requests)
    }
    runner.post_process_outputs(result, SimpleNamespace(requests=requests), outputs)
    runner.model.output_codes.zero_()
    runner.model.output_embeds.zero_()

    assert requests[0].data.committed_codes(0, 1).tolist() == [[7, 8, 9]]
    assert requests[1].data.frame_count == 0
    assert requests[2].data.committed_codes(0, 1).tolist() == [[9, 10, 11]]
    assert [request.data.speech_output_tokens for request in requests] == [1, 1, 1]


def test_a_middle_prefill_chunk_adds_no_frame_or_stream_chunk():
    ids = torch.tensor([7, 9], dtype=torch.long)
    result, forward_batch, requests = step(ids)
    requests[0].data.req.inflight_middle_chunks = 1

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        runner.model.output_codes[:2] = layer0_codes + torch.arange(3)
        runner.model.output_embeds[:2] = layer0_codes.to(torch.float32)

    runner = make_runner(code_predictor_forward, torch.device("cpu"), requests)
    runner.collect_codes(result, forward_batch, object(), requests)
    outputs = {
        req.request_id: RequestOutput(req.request_id, data=int(ids[row]))
        for row, req in enumerate(requests)
    }
    runner.post_process_outputs(result, SimpleNamespace(requests=requests), outputs)

    assert requests[0].data.frame_count == 0
    assert requests[0].data.latest_stream_code_chunk is None
    assert requests[1].data.committed_codes(0, 1).tolist() == [[9, 10, 11]]
    assert requests[1].data.latest_stream_code_chunk is not None
    assert [request.data.speech_output_tokens for request in requests] == [0, 1]


def test_a_stream_step_carries_each_request_frame_count_and_holds_an_opened_row_once():
    ids = torch.tensor([7, 9], dtype=torch.long)
    result, forward_batch, requests = step(ids)

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        runner.model.output_codes[:2] = layer0_codes + torch.arange(3)

    runner = make_runner(code_predictor_forward, torch.device("cpu"), requests)
    runner.enable_stream_steps("vocoder")
    for request in requests:
        request.data.stream_codec_output = True
        request.data.stage_payload = SimpleNamespace(
            request=SimpleNamespace(params={"stream": True})
        )
    outputs = {
        req.request_id: RequestOutput(req.request_id, data=int(ids[row]))
        for row, req in enumerate(requests)
    }
    messages = []
    for _ in range(2):
        runner.collect_codes(result, forward_batch, object(), requests)
        runner.post_process_outputs(result, SimpleNamespace(requests=requests), outputs)
        messages.append(runner.take_stream_step({"r1"}, frozenset()))

    rows = [request.data.code_row for request in requests]
    assert [message.data.request_ids for message in messages] == [["r0"], ["r0"]]
    assert [message.data.data.frame_counts for message in messages] == [[1], [2]]
    assert [list(message.data.opened) for message in messages] == [["r0"], []]
    assert [message.data.data.rows for message in messages] == [rows[:1]] * 2
    assert [runner.code_table.holders[row] for row in rows] == [2, 1]


def test_a_code_row_frees_once_its_last_holder_drops_it():
    _, _, requests = step(torch.tensor([7], dtype=torch.long))
    runner = make_runner(None, torch.device("cpu"), requests)
    row = requests[0].data.code_row
    runner.code_table.hold(row)

    requests[0].data.on_detached()
    free_after_engine = runner.code_table.free_row_count()
    runner.code_table.drop(row)

    assert free_after_engine == 1
    assert runner.code_table.free_row_count() == 2


def test_a_reference_as_long_as_its_prompt_cannot_open_a_row():
    _, _, requests = step(torch.tensor([7], dtype=torch.long))
    requests[0].data.ref_code = torch.zeros((4, 3), dtype=torch.long)
    requests[0].data.ref_code_len = 4

    with pytest.raises(AssertionError, match="4 reference frames"):
        make_runner(None, torch.device("cpu"), requests)


def test_a_middle_prefill_chunk_does_not_advance_its_generation_step():
    _, _, requests = step(torch.tensor([7, 9], dtype=torch.long))
    requests[1].data.req.inflight_middle_chunks = 2
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)

    assert runner.finalize_skip_rids(SimpleNamespace(requests=requests)) == {"r1"}


def test_lookahead_is_never_eligible():
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    history_free = SimpleNamespace(
        sampling_params=SimpleNamespace(
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            min_new_tokens=0,
        ),
        custom_logit_processor=None,
    )

    assert runner.lookahead_eligible(SimpleNamespace(reqs=[history_free])) is False
    assert runner.lookahead_eligible(SimpleNamespace(reqs=[])) is False


@pytest.mark.accelerator
def test_staged_ids_resolve_while_the_predictor_is_still_running():
    device = torch.device("cuda")
    ids = torch.tensor([7, 8], dtype=torch.long, device=device)
    result, forward_batch, requests = step(ids)
    predictor_cycles = 0

    def code_predictor_forward(layer0_codes, hidden, semantic_positions=None):
        torch.cuda._sleep(predictor_cycles)  # noqa: leading-underscore  # upstream name

    runner = make_runner(code_predictor_forward, device, requests)
    stream = torch.cuda.current_stream(device)
    runner.collect_codes(result, forward_batch, object(), requests)
    torch.cuda.synchronize()
    predictor_cycles = PREDICTOR_CYCLES

    runner.collect_codes(result, forward_batch, object(), requests)
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

    runner = make_runner(code_predictor_forward, device, requests)

    runner.collect_codes(result, forward_batch, object(), requests)

    assert runner.resolve_host_token_ids(result).tolist() == [7, 8]
    assert result.next_token_ids.tolist() == [-1, -1]
