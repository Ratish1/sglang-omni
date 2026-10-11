import asyncio
import queue
import threading
from collections import deque
from concurrent.futures import Future

import pytest

from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from tests.unit_test.fixtures.pipeline_fakes import (
    RecordingCoordinatorControlPlane,
    RecordingStageControlPlane,
)
from tests.unit_test.pipeline.helpers import make_stage
from tests.unit_test.pipeline.test_scheduler import (
    init_sync_request_build_state,
    init_terminal_output_state,
)


@pytest.fixture
def scheduler() -> OmniScheduler:
    scheduler = object.__new__(OmniScheduler)
    init_sync_request_build_state(scheduler)
    init_terminal_output_state(scheduler)
    scheduler.session_bridge = None
    scheduler.scheduler_thread_id = threading.get_ident()
    scheduler.tp_size = 1
    scheduler.admin_lock = threading.Lock()
    scheduler._engine_paused = False
    scheduler.last_pause_mode = None
    scheduler.inbox = queue.Queue()
    scheduler.outbox = queue.Queue()
    scheduler.waiting_queue = []
    scheduler.running_batch = scheduler.cur_batch = scheduler.last_batch = None
    scheduler.abort_callback = None
    scheduler.aborted_request_ids = set()
    scheduler.aborted_request_id_order = deque()
    scheduler.dirty_deferred_request_ids = set()
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    return scheduler


@pytest.mark.parametrize("streams", [False, True])
@pytest.mark.parametrize("action", ["pause", "weight-failure"])
def test_admin_drop_reaches_client_and_releases_build(
    scheduler: OmniScheduler, streams: bool, action: str
) -> None:
    async def run() -> None:
        coordinator = Coordinator(
            "inproc://completion",
            "inproc://abort",
            entry_stage="decode",
            terminal_stages=["decode"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("decode", "inproc://decode")
        request = OmniRequest(inputs="hello")
        stream = coordinator.stream("admin-drop", request)
        pending_client = asyncio.create_task(
            anext(stream) if streams else coordinator.submit("admin-drop", request)
        )
        await asyncio.sleep(0)
        assert "admin-drop" in coordinator.requests
        build: Future[None] = Future()
        payload = StagePayload(request_id="admin-drop", request=request, data={})
        scheduler.pending_request_builds["admin-drop"] = (payload, 0.0, build)

        def failed_update(payload: dict[str, object]) -> tuple[bool, str]:
            raise RuntimeError("weight update failed")

        if action == "pause":
            scheduler.admin_pause_generation({"mode": "abort"})
        else:
            with pytest.raises(RuntimeError, match="weight update failed"):
                scheduler.run_weight_update_with_lifecycle(
                    {"abort_all_requests": True}, failed_update, {}
                )
        assert build.cancelled()
        assert not scheduler.outbox.empty()
        stage_control_plane = RecordingStageControlPlane()
        stage = make_stage(
            name="decode", scheduler=scheduler, control_plane=stage_control_plane
        )
        stage.active_requests.add("admin-drop")
        await stage.drain_outbox_external()
        completion = deserialize_message(
            serialize_message(stage_control_plane.completions[0])
        )
        await coordinator.handle_completion(completion)
        await coordinator.handle_completion(completion)
        if streams:
            with pytest.raises(RuntimeError, match="aborted"):
                await pending_client
        else:
            with pytest.raises(asyncio.CancelledError, match="aborted"):
                await pending_client
        assert [message.request_id for message in control_plane.aborts] == [
            "admin-drop"
        ]
        assert not coordinator.request_id_is_reserved("admin-drop")
        assert "admin-drop" not in stage.active_requests

    asyncio.run(run())


def test_follower_drops_without_external_error(scheduler: OmniScheduler) -> None:
    scheduler.is_entry_rank = False
    scheduler.tp_size = 2
    build: Future[None] = Future()
    scheduler.pending_request_builds["follower"] = (
        StagePayload(
            request_id="follower", request=OmniRequest(inputs="hello"), data={}
        ),
        0.0,
        build,
    )
    scheduler.abort_all_requests()
    assert build.cancelled()
    assert scheduler.outbox.empty()
