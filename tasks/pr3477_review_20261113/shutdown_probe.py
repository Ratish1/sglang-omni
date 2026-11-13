"""Stopping between whole-request replays must release that request's state."""

import asyncio
import json
import threading

import pytest

from tests.unit_test.qwen3_tts.test_pipeline import (
    whole_request_message,
    whole_request_scheduler,
)


def main() -> None:
    with pytest.MonkeyPatch.context() as patch:
        scheduler, _, _ = whole_request_scheduler(patch)
        first_replay_completed = threading.Event()
        release_worker = threading.Event()
        replay = scheduler.run_whole_request_replay

        def paused_replay() -> None:
            replay()
            first_replay_completed.set()
            assert release_worker.wait(10)

        patch.setattr(scheduler, "run_whole_request_replay", paused_replay)
        loop = asyncio.new_event_loop()
        scheduler.on_serving_start()
        try:
            scheduler.run_non_streaming_batch(
                [whole_request_message("stopping", 8)], loop
            )
            assert first_replay_completed.wait(10)
            scheduler.signal_async_stop()
            release_worker.set()
            scheduler.stop()
            print(
                json.dumps(
                    {
                        "pending_requests": [
                            request.request_id for request in scheduler.whole_requests
                        ],
                        "owned_slots": [
                            request.slot for request in scheduler.whole_requests
                        ],
                        "idle_slots": len(scheduler.idle_whole_request_slots),
                        "worker_joined": scheduler.initial_worker is None,
                    }
                )
            )
            assert scheduler.initial_worker is None
            assert (
                not scheduler.whole_requests
            ), "stopped scheduler retains whole-request state"
            assert len(scheduler.idle_whole_request_slots) == 8
        finally:
            release_worker.set()
            scheduler.stop()
            loop.close()


if __name__ == "__main__":
    main()
