# SPDX-License-Identifier: Apache-2.0
"""Contracts of the Qwen3-TTS code table and the one message a step that reaches the vocoder."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.qwen3_tts.code_table import CodecFrameStep, Qwen3TTSCodeTable
from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.request_builders import (
    Qwen3TTSSGLangRequestData,
    apply_sglang_qwen3_tts_result,
)
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    IncrementalDecodeBatch,
    IncrementalDecodePlan,
    Qwen3TTSStreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.message import StreamStep
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, detach_request_data
from sglang_omni.utils.audio_payload import audio_waveform_payload

CODEBOOKS = 3
MAX_FRAMES = 16


def make_table(num_rows: int = 4) -> Qwen3TTSCodeTable:
    return Qwen3TTSCodeTable(
        num_rows=num_rows,
        max_frames=MAX_FRAMES,
        num_codebooks=CODEBOOKS,
        device="cpu",
    )


def make_runner(table: Qwen3TTSCodeTable, output_codes: torch.Tensor):
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.code_table = table
    runner.model = SimpleNamespace(
        config=SimpleNamespace(codec_eos_token_id=99, num_code_groups=CODEBOOKS),
        output_codes=output_codes,
    )
    runner.stream_step_target = "vocoder"
    runner.step_frame_request_ids = []
    runner.step_opened_streams = {}
    runner.step_codes_ready = None
    runner.code_step_index = 0
    return runner


def make_request(
    table: Qwen3TTSCodeTable,
    request_id: str,
    *,
    prompt_rows: int = 6,
    ref_frames: int = 0,
    streams: bool = True,
    middle_chunks: int = 0,
) -> SimpleNamespace:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="text", params={"stream": streams}),
        data={},
    )
    return SimpleNamespace(
        request_id=request_id,
        data=Qwen3TTSSGLangRequestData(
            req=SimpleNamespace(
                output_ids=[],
                origin_input_ids=[0] * prompt_rows,
                inflight_middle_chunks=middle_chunks,
            ),
            code_table=table,
            code_row=table.acquire(),
            ref_code=(
                torch.arange(ref_frames * CODEBOOKS).reshape(ref_frames, CODEBOOKS)
                if ref_frames
                else None
            ),
            ref_code_len=ref_frames,
            streams_frames=streams,
            stage_payload=payload,
        ),
    )


class FakeDecoder(torch.nn.Module):
    total_upsample = 4


def make_scheduler(**kwargs: object) -> Qwen3TTSStreamingVocoderScheduler:
    tokenizer = SimpleNamespace(
        model=SimpleNamespace(decoder=FakeDecoder()),
        get_output_sample_rate=lambda: 24000,
    )
    scheduler = Qwen3TTSStreamingVocoderScheduler(tokenizer, device="cpu", **kwargs)
    scheduled: list[tuple[str, str]] = []
    scheduler.schedule_initial = lambda request_id, state: scheduled.append(
        ("initial", request_id)
    )
    scheduler.schedule_followup = lambda request_id, state: scheduled.append(
        ("followup", request_id)
    )
    scheduler.scheduled_decodes = scheduled
    return scheduler


def frame_step(
    table: Qwen3TTSCodeTable,
    request_ids: list[str],
    opened: dict[str, dict[str, object]] | None = None,
) -> StreamStep:
    return StreamStep(
        request_ids=request_ids,
        opened=opened or {},
        data=CodecFrameStep(table=table, ready=None, step=0),
    )


def opening(row: int, ref_frames: int = 0) -> dict[str, object]:
    return {
        "modality": "audio_codes",
        "stream": True,
        "num_quantizers": CODEBOOKS,
        "ref_code_len": ref_frames,
        "code_row": row,
    }


def test_a_row_returns_to_the_free_list_when_every_holder_dropped_it() -> None:
    table = make_table(num_rows=1)

    row = table.acquire()
    table.hold(row)
    table.drop(row)
    assert table.free_rows() == 0

    table.drop(row)
    assert table.free_rows() == 1
    assert table.acquire() == row


def test_admission_takes_the_smaller_of_free_request_rows_and_table_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        omni_scheduler._Upstream,  # noqa: leading-underscore  # module alias
        "get_num_allocatable_reqs",
        lambda *args, **kwargs: 5,
        raising=False,
    )
    capacities = iter([2, 9, None])
    scheduler = SimpleNamespace(
        model_runner=SimpleNamespace(admission_capacity=lambda: next(capacities)),
        session_bridge=None,
    )

    assert OmniScheduler.get_num_allocatable_reqs(scheduler, 0) == 2
    assert OmniScheduler.get_num_allocatable_reqs(scheduler, 0) == 5
    assert OmniScheduler.get_num_allocatable_reqs(scheduler, 0) == 5


def test_detaching_a_request_drops_the_engines_hold_once() -> None:
    table = make_table(num_rows=1)
    request = make_request(table, "req")
    req = SimpleNamespace(omni_data=request.data)

    detach_request_data(req)
    detach_request_data(req)

    assert req.omni_data is None
    assert table.free_rows() == 1


def test_step_frames_land_after_the_reference_at_the_sampled_position() -> None:
    table = make_table()
    runner = make_runner(table, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    first = make_request(table, "first", prompt_rows=6, ref_frames=2)
    second = make_request(table, "second", prompt_rows=9)
    runner.decode_frame_index_bases = runner.frame_index_bases([first, second])
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: False)
    )

    runner.write_step_frames(
        forward_batch, [first, second], torch.tensor([6, 8], dtype=torch.long)
    )

    assert table.codes[first.data.code_row, 2 + 1].tolist() == [1, 2, 3]
    assert table.codes[second.data.code_row, 0].tolist() == [4, 5, 6]


def test_a_middle_prefill_chunk_writes_no_frame_even_at_a_generated_position() -> None:
    table = make_table()
    runner = make_runner(table, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    middle = make_request(table, "middle", prompt_rows=6, middle_chunks=1)
    table.codes[middle.data.code_row, 1] = torch.tensor([7, 7, 7])
    last = make_request(table, "last", prompt_rows=6)
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True)
    )

    runner.write_step_frames(
        forward_batch, [middle, last], torch.tensor([6, 5], dtype=torch.long)
    )

    assert table.codes[middle.data.code_row, 1].tolist() == [7, 7, 7]
    assert table.codes[last.data.code_row, 0].tolist() == [4, 5, 6]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_prefills_frames_write_without_waiting_for_the_predictor() -> None:
    device = torch.device("cuda")
    table = Qwen3TTSCodeTable(
        num_rows=2, max_frames=MAX_FRAMES, num_codebooks=CODEBOOKS, device=device
    )
    runner = make_runner(
        table, torch.ones(2, CODEBOOKS, dtype=torch.long, device=device)
    )
    requests = [make_request(table, "first"), make_request(table, "second")]
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_extend=lambda: True)
    )
    positions = torch.tensor([5, 5], dtype=torch.long, device=device)
    torch.cuda.synchronize()
    torch.cuda._sleep(1_000_000_000)  # noqa: leading-underscore  # upstream name
    stream = torch.cuda.current_stream(device)

    runner.write_step_frames(forward_batch, requests, positions)
    still_running = stream.query()
    torch.cuda.synchronize()

    assert still_running is False
    assert table.codes[requests[0].data.code_row, 0].tolist() == [1, 1, 1]


def test_a_step_counts_only_real_frames_and_opens_each_stream_once() -> None:
    table = make_table()
    runner = make_runner(table, torch.zeros(3, CODEBOOKS, dtype=torch.long))
    streamed = make_request(table, "streamed")
    finished = make_request(table, "finished")
    whole = make_request(table, "whole", streams=False)
    output = SimpleNamespace(requests=[streamed, finished, whole])

    runner.collect_step_frames(output, [5, 99, 6])
    message = runner.take_stream_step(set())
    runner.collect_step_frames(output, [5, 99, 6])
    next_message = runner.take_stream_step(set())

    assert streamed.data.frame_count == 2
    assert finished.data.frame_count == 0
    assert whole.data.frame_count == 2
    assert message.type == "stream_step"
    assert message.target == "vocoder"
    assert message.data.request_ids == ["streamed"]
    assert message.data.opened["streamed"]["code_row"] == streamed.data.code_row
    assert next_message.data.opened == {}


def test_only_a_delivered_stream_gives_the_vocoder_a_hold() -> None:
    table = make_table()
    runner = make_runner(table, torch.zeros(2, CODEBOOKS, dtype=torch.long))
    kept = make_request(table, "kept")
    aborted = make_request(table, "aborted")

    runner.collect_step_frames(SimpleNamespace(requests=[kept, aborted]), [5, 6])
    message = runner.take_stream_step({"aborted"})
    detach_request_data(SimpleNamespace(omni_data=kept.data))
    detach_request_data(SimpleNamespace(omni_data=aborted.data))

    assert message.data.request_ids == ["kept"]
    assert list(message.data.opened) == ["kept"]
    assert table.free_rows() == 3


def test_an_empty_step_sends_nothing() -> None:
    runner = make_runner(make_table(), torch.zeros(1, CODEBOOKS, dtype=torch.long))

    assert runner.take_stream_step(set()) is None


def test_the_result_reads_reference_and_frames_from_the_table_and_outlives_the_row() -> (
    None
):
    table = make_table(num_rows=1)
    request = make_request(table, "req", ref_frames=1)
    row = request.data.code_row
    table.codes[row, :3] = torch.tensor([[9, 9, 9], [1, 1, 1], [2, 2, 2]])
    request.data.frame_count = 2

    result = apply_sglang_qwen3_tts_result(request.data.stage_payload, request.data)
    table.codes[row].zero_()

    assert result.data["audio_codes"].tolist() == [[9, 9, 9], [1, 1, 1], [2, 2, 2]]
    assert result.data["completion_tokens"] == 2


def test_the_vocoder_counts_a_steps_frames_and_schedules_at_each_threshold() -> None:
    table = make_table()
    scheduler = make_scheduler(async_decode=False, initial_chunk_frames=2)
    row = table.acquire()

    scheduler.handle_stream_step(
        frame_step(table, ["req"], {"req": opening(row, ref_frames=3)})
    )
    state = scheduler.stream_states["req"]
    assert (state.ref_frames, state.total_frames) == (3, 4)
    assert state.frame_row == row
    assert state.code_chunks == []
    assert scheduler.scheduled_decodes == []

    scheduler.handle_stream_step(frame_step(table, ["req"]))
    assert scheduler.scheduled_decodes == [("initial", "req")]

    state.decoded_chunks = 1
    state.next_decode_generated_frames = 3
    scheduler.handle_stream_step(frame_step(table, ["req"]))
    assert scheduler.scheduled_decodes[-1] == ("followup", "req")


def test_a_late_frame_of_an_aborted_stream_is_dropped_and_its_row_returned() -> None:
    table = make_table(num_rows=1)
    scheduler = make_scheduler(async_decode=False)
    row = table.acquire()
    table.hold(row)
    scheduler.record_aborted_request_id("gone")

    scheduler.handle_stream_step(frame_step(table, ["gone"], {"gone": opening(row)}))
    table.drop(row)

    assert "gone" not in scheduler.stream_states
    assert table.free_rows() == 1


def test_a_bad_opening_fails_its_own_stream_and_the_step_continues() -> None:
    table = make_table()
    scheduler = make_scheduler(async_decode=False)
    good_row, bad_row = table.acquire(), table.acquire()
    errors: list[str] = []
    scheduler.emit_error = lambda request_id, exc: errors.append(request_id)
    bad_metadata = opening(bad_row) | {"num_quantizers": 0}

    scheduler.handle_stream_step(
        frame_step(
            table,
            ["bad", "good"],
            {"bad": bad_metadata, "good": opening(good_row)},
        )
    )

    assert errors == ["bad"]
    assert "bad" not in scheduler.stream_states
    assert scheduler.stream_states["good"].total_frames == 1


def test_a_finished_stream_drops_the_vocoders_hold() -> None:
    table = make_table(num_rows=1)
    scheduler = make_scheduler(async_decode=False)
    row = table.acquire()

    scheduler.handle_stream_step(frame_step(table, ["req"], {"req": opening(row)}))
    scheduler.clear_request_state("req")

    assert table.free_rows() == 1


def table_plan(
    row: int, start: int, fresh: int, trim: int = 0, emit: int | None = None
) -> IncrementalDecodePlan:
    return IncrementalDecodePlan(
        decoder_input=None,
        slot=row,
        fresh_frames=fresh,
        reference_trim_frames=trim,
        generated_frames=fresh - trim if emit is None else emit,
        emitted_generated_frames=0,
        frame_row=row,
        start_frame=start,
        frames=CodecFrameStep(table=None, ready=None, step=0),
    )


def scheduler_on_table(table: Qwen3TTSCodeTable) -> Qwen3TTSStreamingVocoderScheduler:
    scheduler = make_scheduler(async_decode=False)
    scheduler.code_table = table
    scheduler.codec_arena = SimpleNamespace(
        staged=lambda name, values: torch.tensor(values, dtype=torch.long)
    )
    return scheduler


def test_a_cohorts_gather_equals_each_rows_slice_as_the_decoder_reads_it() -> None:
    table = make_table()
    table.codes.copy_(torch.arange(table.codes.numel()).reshape(table.codes.shape))
    scheduler = scheduler_on_table(table)

    codes = scheduler.plans_decoder_input(
        [table_plan(0, start=2, fresh=3), table_plan(3, start=5, fresh=3)]
    )

    assert torch.equal(
        codes,
        torch.stack(
            (table.codes[0, 2:5].transpose(0, 1), table.codes[3, 5:8].transpose(0, 1))
        ),
    )


def test_a_cohort_waits_for_its_newest_rows_step_and_no_newer_one() -> None:
    table = make_table()
    scheduler = scheduler_on_table(table)
    waited: list[str] = []
    scheduler.wait_frames_ready = lambda frames: waited.append(frames.ready)
    older = table_plan(0, start=0, fresh=1)
    newer = replace(
        table_plan(1, start=0, fresh=1),
        frames=CodecFrameStep(table=table, ready="newer", step=7),
    )

    scheduler.gather_table_codes([older, newer], 1)

    assert waited == ["newer"]


def test_a_stream_remembers_the_step_of_its_own_last_frame() -> None:
    table = make_table()
    scheduler = make_scheduler(async_decode=False)
    row = table.acquire()
    first = frame_step(table, ["req"], {"req": opening(row)})
    second = frame_step(table, ["other"])

    scheduler.handle_stream_step(first)
    scheduler.handle_stream_step(second)

    assert scheduler.stream_states["req"].last_frames is first.data


def test_a_cohort_of_different_widths_pads_the_short_rows_with_zeros() -> None:
    table = make_table()
    table.codes.fill_(7)
    scheduler = scheduler_on_table(table)

    codes = scheduler.plans_decoder_input(
        [table_plan(0, start=0, fresh=3), table_plan(1, start=0, fresh=1)], width=3
    )

    assert codes[0].eq(7).all()
    assert codes[1, :, 0].eq(7).all()
    assert codes[1, :, 1:].eq(0).all()


def test_a_batch_with_a_row_off_the_table_keeps_each_rows_codes() -> None:
    table = make_table()
    table.codes.fill_(7)
    scheduler = scheduler_on_table(table)
    chunk_plan = IncrementalDecodePlan(
        decoder_input=torch.full((1, CODEBOOKS, 2), 3),
        slot=0,
        fresh_frames=2,
        reference_trim_frames=0,
        generated_frames=2,
        emitted_generated_frames=0,
    )

    codes = scheduler.plans_decoder_input([chunk_plan, table_plan(1, 0, fresh=2)])

    assert codes[0].eq(3).all()
    assert codes[1].eq(7).all()


def test_a_streams_own_frames_are_a_copy_the_screen_can_clamp() -> None:
    table = make_table()
    table.codes.fill_(7)
    scheduler = scheduler_on_table(table)
    state = SimpleNamespace(
        frame_row=1, last_frames=CodecFrameStep(table=table, ready=None, step=0)
    )

    codes = scheduler.table_codes(state, 0, 2)
    codes.fill_(0)

    assert table.codes[1].eq(7).all()


def test_a_cohorts_rect_holds_what_each_rows_delta_would() -> None:
    scheduler = scheduler_on_table(make_table())
    samples = scheduler.samples_per_frame
    waveform = torch.arange(2.0 * 3 * samples).reshape(2, 1, 3 * samples)
    plans = [
        table_plan(0, start=0, fresh=3, trim=1, emit=2),
        table_plan(1, start=0, fresh=3, trim=1, emit=2),
    ]
    batch = IncrementalDecodeBatch(decoder=None, arena=None, slots=[0, 1])

    rect = scheduler.incremental_rect(plans, waveform, batch)
    deltas = scheduler.incremental_row_deltas(plans, waveform, batch)

    assert torch.equal(rect, torch.stack(deltas))


def test_a_cohort_with_different_spans_has_no_rect() -> None:
    scheduler = scheduler_on_table(make_table())
    samples = scheduler.samples_per_frame
    plans = [
        table_plan(0, start=0, fresh=3, trim=1, emit=2),
        table_plan(1, start=0, fresh=3, trim=0, emit=3),
    ]
    batch = IncrementalDecodeBatch(decoder=None, arena=None, slots=[0, 1])

    assert (
        scheduler.incremental_rect(plans, torch.zeros(2, 1, 3 * samples), batch) is None
    )


def test_a_followup_payload_matches_the_shared_audio_payload() -> None:
    scheduler = scheduler_on_table(make_table())
    delta = torch.arange(8, dtype=torch.float32)
    state = SimpleNamespace(whole_audio=None)

    payload = scheduler.followup_payload(("req", state, None), delta)

    expected = audio_waveform_payload(
        delta,
        sample_rate=scheduler.sample_rate,
        modality="audio",
        source_hint=scheduler.stream_source_hint,
    )
    assert payload == expected
    assert np.frombuffer(payload["audio_waveform"], dtype=np.float32).tolist() == [
        float(value) for value in range(8)
    ]


def test_a_whole_request_or_session_row_builds_its_chunk_the_other_way() -> None:
    scheduler = scheduler_on_table(make_table())
    delta = torch.zeros(4)
    scheduler.session_appends["unit"] = object()

    assert (
        scheduler.followup_payload(
            ("req", SimpleNamespace(whole_audio=[]), None), delta
        )
        is None
    )
    assert (
        scheduler.followup_payload(
            ("unit", SimpleNamespace(whole_audio=None), None), delta
        )
        is None
    )


class BindingScheduler(OmniScheduler):
    def __init__(self) -> None:
        self.bound: list[str] = []

    def enable_stream_steps(self, target: str) -> None:
        self.bound.append(target)


def binding_stage(
    *,
    role: str = "single",
    stream_targets: tuple[str, ...] = ("vocoder",),
    same_process: bool = True,
    replicated: bool = False,
    accepts: bool = True,
    before_payload: bool = True,
) -> SimpleNamespace:
    receiver = SimpleNamespace(
        can_accept_stream_before_payload=before_payload,
        stream_queue=object(),
        scheduler=make_scheduler(async_decode=accepts),
    )
    return SimpleNamespace(
        name="tts_engine",
        role=role,
        stream_targets=list(stream_targets),
        same_process_targets={"vocoder"} if same_process else set(),
        replica_topology=SimpleNamespace(is_replicated=lambda target: replicated),
        local_dispatcher=SimpleNamespace(get_stage=lambda source, target: receiver),
        scheduler=BindingScheduler(),
    )


@pytest.mark.parametrize(
    ("overrides", "bound"),
    [
        ({}, ["vocoder"]),
        ({"role": "leader"}, []),
        ({"stream_targets": ("vocoder", "other")}, []),
        ({"same_process": False}, []),
        ({"replicated": True}, []),
        ({"accepts": False}, []),
        ({"before_payload": False}, []),
    ],
)
def test_only_a_same_process_edge_taking_steps_binds(
    overrides: dict[str, object], bound: list[str]
) -> None:
    stage = binding_stage(**overrides)

    Stage.bind_stream_steps(stage)

    assert stage.scheduler.bound == bound
