"""Exercise the real Python ownership paths with deterministic device faults."""

import asyncio
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts import codec_state_arena
from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from tests.unit_test.qwen3_tts.test_incremental_codec import Decoder, make_arena
from tests.unit_test.qwen3_tts.test_pipeline import (
    admit_reference_stream,
    run_whole_requests,
    whole_request_message,
    whole_request_scheduler,
)


class ReleaseEvent:
    producer: str | None = None

    def record(self, stream: "DecodeStream") -> None:
        self.producer = stream.name


class DecodeStream:
    def __init__(self, name: str, *, fail_wait: bool = False) -> None:
        self.name = name
        self.fail_wait = fail_wait
        self.dependencies: list[str | None] = []

    def wait_event(self, event: ReleaseEvent) -> None:
        if self.fail_wait:
            raise RuntimeError("injected event wait failure")
        self.dependencies.append(event.producer)


def release_dependency_after_wait_failure() -> dict[str, str | list[str | None]]:
    _, arena = make_arena(Decoder(), slots=1)
    producer = DecodeStream("producer")
    rejected_consumer = DecodeStream("rejected consumer", fail_wait=True)
    retry_consumer = DecodeStream("retry consumer")
    current_stream = producer
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(codec_state_arena, "has_device_streams", lambda device: True)
        patch.setattr(
            arena,
            "device_module",
            SimpleNamespace(Event=ReleaseEvent, current_stream=lambda device: current_stream),
        )
        assert arena.acquire() == 0
        arena.release(0)
        current_stream = rejected_consumer
        with pytest.raises(RuntimeError, match="injected event wait failure"):
            arena.acquire()
        current_stream = retry_consumer
        assert arena.acquire() == 0
    assert retry_consumer.dependencies == ["rejected consumer"]
    return {
        "original_release_producer": producer.name,
        "retry_dependencies": retry_consumer.dependencies,
    }


def waiting_whole_displaced_by_new_streams() -> dict[str, int]:
    with pytest.MonkeyPatch.context() as patch, asyncio.Runner() as event_loop:
        scheduler, _, runner = whole_request_scheduler(patch)
        scheduler.followup_worker = scheduler.initial_worker
        arena = scheduler.codec_arena
        held_slots = [arena.acquire() for _ in range(arena.num_slots - 1)]
        codes = torch.arange(1, 65).view(-1, 1).repeat(1, 2)
        active = admit_reference_stream(scheduler, "first", codes, ref_code_len=0)
        scheduler.initial_queue.put(("first", active))
        scheduler.run_initial_pass()
        assert active.codec_slot is not None
        scheduler.run_non_streaming_batch(
            [whole_request_message("whole", 4)], event_loop.get_loop()
        )
        scheduler.run_initial_pass()
        assert scheduler.slot_waiting_whole_request_count == 1
        replay_count = len(runner.calls)
        handoffs = 12
        for index in range(handoffs):
            scheduler.release_codec_slot(active)
            request_id = f"replacement-{index}"
            active = admit_reference_stream(scheduler, request_id, codes, ref_code_len=0)
            scheduler.initial_queue.put((request_id, active))
            scheduler.run_initial_pass()
            assert active.codec_slot is not None
            assert scheduler.whole_requests[0].slot is None
            assert scheduler.slot_waiting_whole_request_count == 1
        scheduler.release_codec_slot(active)
        for slot in held_slots:
            arena.release(slot)
        replies = run_whole_requests(scheduler)
        assert replies["whole"].type == "result"
        return {"row_handoffs_to_new_streams": handoffs, "whole_requests_completed_when_arrivals_pause": 1}


class RejectedCUDAWait:
    def __init__(self) -> None:
        self.should_fail = True

    def wait_event(self, event: torch.cuda.Event) -> None:
        self.should_fail = False
        raise RuntimeError("injected event wait failure")


def cuda_release_dependency_after_wait_failure() -> dict[str, int | bool]:
    device = torch.device("cuda:0")
    arena = Qwen3TTSCodecStateArena(
        Qwen3TTSIncrementalDecoder(Decoder()), num_slots=1,
        device=device, dtype=torch.float32,
    )
    producer = torch.cuda.Stream(device=device)
    rejected_consumer = torch.cuda.Stream(device=device)
    retry_consumer = torch.cuda.Stream(device=device)
    producer_complete = torch.cuda.Event()
    torch.cuda.synchronize()
    with torch.cuda.stream(retry_consumer):
        assert arena.acquire() == 0
        arena.release(0)
    with torch.cuda.stream(producer):
        assert arena.acquire() == 0
        torch.cuda._sleep(500_000_000)
        arena.start_slot(0, 17)
        arena.release(0)
        producer_complete.record()
    real_current_stream = torch.cuda.current_stream
    failing_wait = RejectedCUDAWait()
    with torch.cuda.stream(rejected_consumer), pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            arena,
            "device_module",
            SimpleNamespace(
                Event=torch.cuda.Event,
                current_stream=lambda device: (
                    failing_wait
                    if failing_wait.should_fail
                    else real_current_stream(device)
                ),
            ),
        )
        with pytest.raises(RuntimeError, match="injected event wait failure"):
            arena.acquire()
    with torch.cuda.stream(retry_consumer):
        assert arena.acquire() == 0
        retry_consumer.synchronize()
        observed_before_completion = int(arena.storage.frame_positions[0].item())
    was_producer_pending = not producer_complete.query()
    producer_complete.synchronize()
    final_position = int(arena.storage.frame_positions[0].item())
    assert was_producer_pending and observed_before_completion == 0 and final_position == 17
    arena.release(0)
    return {
        "producer_pending_after_retry_zeroed_row": was_producer_pending,
        "position_after_retry_zero": observed_before_completion,
        "position_after_previous_producer_completed": final_position,
    }


if __name__ == "__main__" and sys.argv[1:] == ["cuda"]:
    print(json.dumps(cuda_release_dependency_after_wait_failure(), indent=2))
elif __name__ == "__main__":
    print(json.dumps({
        "event_wait_failure": release_dependency_after_wait_failure(),
        "mixed_schedule": waiting_whole_displaced_by_new_streams(),
    }, indent=2))
