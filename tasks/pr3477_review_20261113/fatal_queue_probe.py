"""A retired last whole-request slot must not prevent pending requests from failing."""

import asyncio
import json

import pytest
import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from tests.unit_test.qwen3_tts.test_pipeline import (
    whole_request_message,
    whole_request_scheduler,
)


def main() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler, decoder, runner = whole_request_scheduler(patch)
        arena = Qwen3TTSCodecStateArena(
            decoder,
            num_slots=1,
            reserved_slots=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        scheduler.codec_arena = arena
        runner.arena = arena
        scheduler.idle_whole_request_slots = list(arena.reserved_slots)

        def fail_completion(plans, *, stream, incremental=None):
            scheduler.cuda_decode_failed = True
            for slot in incremental.slots:
                incremental.arena.retire(slot)
            raise RuntimeError("decode completion not proven")

        patch.setattr(scheduler, "launch_decode_plans", fail_completion)
        loop = asyncio.new_event_loop()
        try:
            scheduler.run_non_streaming_batch(
                [
                    whole_request_message("faulted", 8),
                    whole_request_message("waiting", 8),
                ],
                loop,
            )
            for _ in range(6):
                scheduler.run_initial_pass()
            messages = {}
            while not scheduler.outbox.empty():
                message = scheduler.outbox.get_nowait()
                messages[message.request_id] = message.type
            print(
                json.dumps(
                    {
                        "messages": messages,
                        "waiting": [
                            request.request_id for request in scheduler.whole_requests
                        ],
                        "idle_slots": scheduler.idle_whole_request_slots,
                        "retired_slots": sorted(arena.retired),
                    }
                )
            )
            assert messages.get("faulted") == "error"
            assert (
                messages.get("waiting") == "error"
            ), "waiting request cannot reach the disabled-CUDA guard"
            assert not scheduler.whole_requests
        finally:
            loop.close()


if __name__ == "__main__":
    main()
