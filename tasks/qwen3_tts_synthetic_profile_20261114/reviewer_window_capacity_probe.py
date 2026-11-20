"""Compare dedicated row capacity with the same real codec and graph batch sizes."""

import asyncio
import json
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import nullcontext
from unittest.mock import patch

import torch

from sglang_omni.models.qwen3_tts import streaming_vocoder
from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
from sglang_omni.models.qwen3_tts.stages import create_vocoder_executor
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.qwen3_tts.test_pipeline import (
    admit_reference_stream,
    make_payload,
)


def run_probe(whole_capacity: int) -> dict:
    arena_type = (
        nullcontext()
        if whole_capacity == 8
        else patch.object(
            streaming_vocoder,
            "Qwen3TTSCodecStateArena",
            lambda *args, reserved_slots, **kwargs: Qwen3TTSCodecStateArena(
                *args, reserved_slots=whole_capacity, **kwargs
            ),
        )
    )
    with arena_type:
        scheduler = create_vocoder_executor(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            device="cuda", gpu_id=0, dtype="bfloat16", codec_state_slots=128,
            followup_worker_count=1, incremental_codec_compile=False,
            max_in_flight=128, shares_engine_speech_tokenizer=False,
            suppress_bootstrap_silence=False,
        )
    assert scheduler.decodes_whole_requests
    scheduler.initial_worker = threading.current_thread()
    scheduler.followup_worker = scheduler.initial_worker
    scheduler.activate_decode_worker(scheduler.decode_stream)
    arena = scheduler.codec_arena
    runner = scheduler.initial_window_decode_graphs
    assert runner.enabled and runner.graphs
    assert runner.batch_sizes[-1] == 8
    assert len(arena.reserved_slots) == whole_capacity
    assert scheduler.codec_arena.num_slots == 128
    assert tuple(runner.fresh_frames[-3:]) == (16, 32, 64)
    widest = runner.fresh_frames[-1]
    generator = torch.Generator().manual_seed(42)
    quantizers = scheduler.decoder.config.num_quantizers
    vocabulary = scheduler.decoder.config.codebook_size

    def random_codes(frames: int) -> torch.Tensor:
        return torch.randint(vocabulary, (frames, quantizers), generator=generator)

    def message(request_id: str, frames: int) -> IncomingMessage:
        payload = make_payload(inputs=request_id)
        payload.request_id = request_id
        payload.data = Qwen3TTSState(
            audio_codes=random_codes(frames), ref_code_len=0,
            prompt_tokens=0, completion_tokens=frames,
        ).to_dict()
        return IncomingMessage(request_id, "new_request", payload)

    def observe_passes(
        loop: asyncio.AbstractEventLoop,
        requests: list[IncomingMessage],
        *,
        check_isolation: bool = False,
    ) -> dict:
        launch_shapes: list[tuple[int, int]] = []
        completion_passes: dict[str, int] = {}
        completion_times: dict[str, float] = {}
        original_decode = runner.decode_slots

        def decode(
            codes: torch.Tensor, slots: Sequence[int], valid_frames: Sequence[int] | None = None
        ) -> torch.Tensor | None:
            decoded = original_decode(codes, slots, valid_frames)
            assert decoded is not None, (codes.shape, slots, runner.stats())
            launch_shapes.append((int(codes.shape[-1]), len(slots)))
            return decoded

        scheduler.run_non_streaming_batch(requests, loop)
        started = time.perf_counter()
        with patch.object(runner, "decode_slots", decode):
            for pass_number in range(1, 40):
                scheduler.run_initial_pass()
                if check_isolation and pass_number == 1:
                    allocated_rows = [
                        request.slot for request in scheduler.whole_requests
                        if request.slot is not None
                    ]
                    assert len(allocated_rows) == 8
                    assert all(slot in arena.reserved_slots for slot in allocated_rows)
                    stream = admit_reference_stream(scheduler, "live-stream", random_codes(64), ref_code_len=0)
                    scheduler.initial_queue.put(("live-stream", stream))
                if check_isolation and pass_number == 2:
                    assert stream.codec_slot in range(arena.num_slots)
                    assert not stream.incremental_codec_fallback
                    scheduler.abort("live-stream")
                while not scheduler.outbox.empty():
                    entry = scheduler.outbox.get_nowait()
                    if entry.type in {"result", "error"}:
                        assert entry.type == "result", (entry.request_id, entry.data)
                        completion_passes[entry.request_id] = pass_number
                        completion_times[entry.request_id] = (time.perf_counter() - started) * 1000
                if not scheduler.whole_requests and scheduler.initial_queue.empty():
                    break
        assert len(completion_passes) == len(requests)
        assert len(scheduler.idle_whole_request_slots) == whole_capacity
        assert arena.active_slots() == 0
        short_ids = [key for key in completion_passes if "-short-" in key]
        return {
            "replay_shapes": launch_shapes,
            "short_completion_passes": [completion_passes[key] for key in short_ids],
            "short_completion_ms": [round(completion_times[key], 2) for key in short_ids],
        }

    def queue(name: str, long_lengths: list[int], short_length: int) -> list[IncomingMessage]:
        return [
            *(message(f"{name}-long-{index}", length) for index, length in enumerate(long_lengths)),
            *(message(f"{name}-short-{index}", short_length) for index in range(7)),
        ]

    with asyncio.Runner() as event_loop, torch.inference_mode():
        loop = event_loop.get_loop()
        observe_passes(loop, queue("warmup", [widest * 2] * 8, 16), check_isolation=True)
        long_run = observe_passes(loop, queue("long", [widest * 6] * 8, 16))
        mixed_tail = observe_passes(loop, queue("tails", [widest + 16] + [widest + 32] * 7, 16))
    return {
        "whole_capacity": whole_capacity,
        "arena_total_bytes": arena.total_bytes,
        "long_run": long_run,
        "mixed_tail": mixed_tail,
        "stream_acquired_during_whole_replay_and_aborted_cleanly": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(int(sys.argv[1])), indent=2))
