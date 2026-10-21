"""Force pending-readback lifecycle boundaries through the actual scheduler."""

import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang_omni.model_runner.base import ModelRunner
from tests.unit_test.fun_cosyvoice3.test_streaming import (
    drain,
    item,
    make_scheduler,
    stream_ids,
    stream_payload,
)


class PendingCopy:
    def __init__(self, fail=False):
        self.fail = fail

    def synchronize(self):
        if self.fail:
            raise RuntimeError("forced readback failure")


def pending_step(request_id):
    _, scheduler = make_scheduler()
    original = scheduler.vocoder.hift_step

    def hift_step(rows):
        return dataclasses.replace(original(rows), copy_done=PendingCopy())

    scheduler.vocoder.hift_step = hift_step
    scheduler.handle_streaming_new_request(request_id, stream_payload(request_id))
    scheduler.ingest_stream_item(request_id, item(list(range(28))))
    scheduler.pump_one_step()
    assert drain(scheduler) == []
    return scheduler


def main():
    outcomes = {}
    scheduler = pending_step("old")
    scheduler.handle_streaming_new_request("new", stream_payload("new"))
    scheduler.ingest_stream_item("new", item(list(range(28))))
    with patch.object(
        scheduler.vocoder,
        "hop_batch",
        side_effect=RuntimeError("forced launch failure"),
    ):
        failed = scheduler.pump_one_step()
    messages = drain(scheduler)
    assert failed == ["new"]
    assert stream_ids(messages) == ["old"]
    assert [(m.request_id, m.type) for m in messages if m.type == "error"] == [
        ("new", "error")
    ]
    assert scheduler.launched_step is None
    outcomes["new_launch_failure_preserves_previous_audio"] = True

    scheduler = pending_step("old")
    scheduler.launched_step.samples.copy_done.fail = True
    scheduler.handle_streaming_new_request("new", stream_payload("new"))
    scheduler.ingest_stream_item("new", item(list(range(28))))
    assert scheduler.pump_one_step() == ["old"]
    assert stream_ids(drain(scheduler)) == []
    assert scheduler.pump_one_step() == []
    assert stream_ids(drain(scheduler)) == ["new"]
    assert scheduler.is_aborted("old") and not scheduler.is_aborted("new")
    outcomes["previous_read_failure_isolates_new_request"] = True

    scheduler = pending_step("same")
    old_state = scheduler.stream_states["same"]
    scheduler.clear_request_state("same")
    scheduler.handle_streaming_new_request("same", stream_payload("same"))
    assert scheduler.stream_states["same"] is not old_state
    assert scheduler.pump_one_step() == []
    assert drain(scheduler) == []
    outcomes["replacement_state_receives_no_old_audio"] = True

    scheduler = pending_step("stop")
    scheduler.stop()
    assert scheduler.launched_step is None and scheduler.stream_states == {}
    assert drain(scheduler) == [] and not scheduler.has_ready_work()
    outcomes["stop_drops_pending_audio_and_state"] = True

    runner = ModelRunner.__new__(ModelRunner)
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    runner.token_id_events = []
    runner.token_id_event_slot = 0
    previous = None
    with torch.inference_mode():
        for step, size in enumerate((1, 8, 2, 16, 4, 32)):
            ids = torch.arange(size, device="cuda") + step * 100
            result = SimpleNamespace()
            runner.stage_token_ids(result, ids)
            if previous is not None:
                previous_result, expected = previous
                assert torch.equal(
                    runner.resolve_host_token_ids(previous_result), expected
                )
            previous = result, torch.arange(size) + step * 100
        assert torch.equal(runner.resolve_host_token_ids(previous[0]), previous[1])
    outcomes["event_reuse_with_buffer_growth_preserves_lagged_ids"] = True
    print(json.dumps(outcomes), flush=True)


if __name__ == "__main__":
    main()
