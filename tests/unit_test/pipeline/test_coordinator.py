# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc

import pytest

from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


class SignallingCoordinatorControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.submitted_event = asyncio.Event()
        self.fail_abort = False

    async def submit_to_stage(self, stage: str, endpoint: str, msg: object) -> None:
        await super().submit_to_stage(stage, endpoint, msg)
        self.submitted_event.set()

    async def broadcast_abort(self, msg: object) -> None:
        await super().broadcast_abort(msg)
        if self.fail_abort:
            raise RuntimeError("abort transport failed")


class FailingSubmissionControlPlane(RecordingCoordinatorControlPlane):
    async def submit_to_stage(self, stage: str, endpoint: str, msg: object) -> None:
        del stage, endpoint, msg
        raise RuntimeError("submit transport failed")


class BlockingAbortControlPlane(SignallingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.abort_started = asyncio.Event()
        self.release_abort = asyncio.Event()

    async def broadcast_abort(self, msg: object) -> None:
        self.aborts.append(msg)
        self.abort_started.set()
        await self.release_abort.wait()


def _coordinator_with_control_plane(
    control_plane: RecordingCoordinatorControlPlane,
) -> Coordinator:
    coordinator = Coordinator(
        "inproc://complete",
        "inproc://abort",
        entry_stage="preprocess",
        terminal_stages=["decode"],
    )
    coordinator.control_plane = control_plane
    coordinator.register_stage("preprocess", "inproc://preprocess")
    return coordinator


def test_coordinator_multi_terminal_completion_and_abort_contracts() -> None:
    """Preserves multi-terminal completion and abort cancellation semantics."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", {"text": "hello"})
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert not coordinator._completion_futures["req-1"].done()
        await coordinator._handle_completion(
            CompleteMessage("req-1", "code2wav", True, result={"audio": "ok"})
        )
        assert coordinator._completion_futures["req-1"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

        await coordinator._submit_request("req-2", "hello")
        future = coordinator._completion_futures["req-2"]
        assert await coordinator.abort("req-2") is True
        assert control_plane.aborts[0].request_id == "req-2"
        with pytest.raises(asyncio.CancelledError):
            await future

    asyncio.run(_run())


def test_coordinator_resolves_active_terminal_subset_per_request() -> None:
    async def _run() -> None:
        def terminal_stages(request: OmniRequest) -> list[str]:
            assert isinstance(request, OmniRequest)
            if request.metadata.get("audio"):
                return ["decode", "code2wav"]
            return ["decode"]

        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=terminal_stages,
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request(
            "text-req",
            OmniRequest(inputs="hello", metadata={"audio": False}),
        )
        await coordinator._handle_completion(
            CompleteMessage("text-req", "decode", True, result={"text": "hi"})
        )
        assert coordinator._completion_futures["text-req"].result() == {"text": "hi"}

        await coordinator._submit_request("raw-text-req", "hello")
        await coordinator._handle_completion(
            CompleteMessage("raw-text-req", "decode", True, result={"text": "raw"})
        )
        assert coordinator._completion_futures["raw-text-req"].result() == {
            "text": "raw"
        }

        await coordinator._submit_request(
            "audio-req",
            OmniRequest(inputs="hello", metadata={"audio": True}),
        )
        await coordinator._handle_completion(
            CompleteMessage("audio-req", "decode", True, result={"text": "hi"})
        )
        assert not coordinator._completion_futures["audio-req"].done()
        await coordinator._handle_completion(
            CompleteMessage(
                "audio-req",
                "code2wav",
                True,
                result={"audio": "ok"},
            )
        )
        assert coordinator._completion_futures["audio-req"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

    asyncio.run(_run())


def test_coordinator_rejects_invalid_resolved_terminal_subset() -> None:
    async def _run() -> None:
        for resolved, error in (
            ([], "no terminal stages"),
            (["decode", "missing"], "outside the static terminal stages"),
            ("decode", "must return a sequence"),
        ):
            coordinator = Coordinator(
                "inproc://complete",
                "inproc://abort",
                entry_stage="preprocess",
                terminal_stages=["decode", "code2wav"],
                terminal_stages_resolver=lambda request, resolved=resolved: resolved,
            )
            coordinator.control_plane = RecordingCoordinatorControlPlane()
            coordinator.register_stage("preprocess", "inproc://preprocess")

            with pytest.raises(ValueError, match=error):
                await coordinator._submit_request("req-1", OmniRequest(inputs="hello"))
            assert coordinator._requests == {}
            assert coordinator.control_plane.submitted == []

    asyncio.run(_run())


def test_coordinator_stream_cleans_queue_when_terminal_resolver_rejects() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=lambda request: [],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        stream = coordinator.stream("req-1", OmniRequest(inputs="hello"))
        with pytest.raises(ValueError, match="no terminal stages"):
            await stream.__anext__()
        await stream.aclose()

        assert coordinator._stream_queues == {}
        assert coordinator._completion_futures == {}
        assert coordinator.control_plane.submitted == []

    asyncio.run(_run())


def test_coordinator_stream_uses_request_terminal_subset_after_cleanup() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
            terminal_stages_resolver=lambda request: ["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        events = []

        async def _consume() -> None:
            async for event in coordinator.stream("req-1", OmniRequest(inputs="hello")):
                events.append(event)

        task = asyncio.create_task(_consume())
        for _ in range(10):
            if "req-1" in coordinator._requests:
                break
            await asyncio.sleep(0)
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        await asyncio.wait_for(task, timeout=1)

        assert [event.from_stage for event in events] == ["decode"]

    asyncio.run(_run())


def test_coordinator_stream_received_event_pairs_terminal_chunk(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.coordinator._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        queue: asyncio.Queue = asyncio.Queue()
        coordinator._stream_queues["req-1"] = queue

        await coordinator._handle_stream(
            StreamMessage(
                request_id="req-1",
                from_stage="decode",
                chunk={"text": "hi"},
                modality="text",
                chunk_id=1,
            )
        )

        routed = queue.get_nowait()
        assert routed.chunk_id == 1

    asyncio.run(_run())

    receive_events = [
        event
        for event in events
        if event["event_name"] == "stage_stream_chunk_received"
    ]
    assert len(receive_events) == 1
    assert receive_events[0]["stage"] == "coordinator"
    assert receive_events[0]["metadata"] == {
        "from_stage": "decode",
        "chunk_id": 1,
        "modality": "text",
    }


def test_stream_message_round_trips_terminal_chunk_id() -> None:
    msg = StreamMessage(
        request_id="req-1",
        from_stage="decode",
        chunk={"text": "hi"},
        modality="text",
        chunk_id=3,
    )

    round_trip = StreamMessage.from_dict(msg.to_dict())

    assert round_trip.chunk_id == 3
    assert round_trip.modality == "text"


def test_coordinator_failure_completion_fails_fast_and_cleans_state() -> None:
    """Preserves fail-fast behavior and cleanup after any terminal failure."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        future = coordinator._completion_futures["req-1"]
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert coordinator._partial_results["req-1"] == {"decode": {"text": "hi"}}

        await coordinator._handle_completion(
            CompleteMessage("req-1", "code2wav", False, error="boom")
        )

        with pytest.raises(RuntimeError, match="boom"):
            await future
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._partial_results
        assert control_plane.aborts[-1].request_id == "req-1"

    asyncio.run(_run())


def test_coordinator_fail_pending_requests_resolves_waiters() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", "hello")
        future = coordinator._completion_futures["req-1"]

        await coordinator.fail_pending_requests(RuntimeError("stage died"))

        with pytest.raises(RuntimeError, match="stage died"):
            await future
        assert coordinator._requests == {}
        assert coordinator._partial_results == {}

    asyncio.run(_run())


async def _drive_stream_until_registered(coordinator: Coordinator, request_id: str):
    """Start consuming a stream and return (task, error_sink, future) once the
    request's completion future has been created."""
    error_sink: list[str] = []

    async def _consume() -> None:
        try:
            async for _msg in coordinator.stream(request_id, "hello"):
                pass
        except RuntimeError as exc:
            error_sink.append(str(exc))

    task = asyncio.create_task(_consume())
    for _ in range(100):
        if request_id in coordinator._completion_futures:
            break
        await asyncio.sleep(0)
    future = coordinator._completion_futures[request_id]
    return task, error_sink, future


def test_coordinator_stream_abort_cancels_future_without_unretrieved_exception() -> (
    None
):
    """Aborting a streaming request cancels its completion future instead of
    setting an exception no one retrieves, so the event loop never reports a
    'Future exception was never retrieved' error."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        loop = asyncio.get_running_loop()
        handler_contexts: list = []
        loop.set_exception_handler(
            lambda _loop, context: handler_contexts.append(context)
        )

        task, error_sink, future = await _drive_stream_until_registered(
            coordinator, "req-1"
        )

        assert await coordinator.abort("req-1") is True
        await asyncio.wait_for(task, timeout=1)

        # Stream terminated via its queue; the future is cancelled rather than
        # carrying an un-retrieved exception.
        assert error_sink == ["aborted"]
        assert future.cancelled() is True
        assert "req-1" not in coordinator._completion_futures

        # Dropping the future must not trip the loop's exception handler.
        del future
        gc.collect()
        assert not any(
            "never retrieved" in str(ctx.get("message", "")) for ctx in handler_contexts
        )

    asyncio.run(_run())


def test_coordinator_stream_fail_pending_requests_cancels_future() -> None:
    """A coordinator failure reaches the stream without leaving an exception
    on its unused completion future."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        loop = asyncio.get_running_loop()
        handler_contexts: list = []
        loop.set_exception_handler(
            lambda _loop, context: handler_contexts.append(context)
        )

        task, error_sink, future = await _drive_stream_until_registered(
            coordinator, "req-1"
        )

        await coordinator.fail_pending_requests(RuntimeError("stage died"))
        await asyncio.wait_for(task, timeout=1)

        assert error_sink == ["stage died"]
        assert future.cancelled() is True
        assert "req-1" not in coordinator._completion_futures

        del future
        gc.collect()
        assert not any(
            "never retrieved" in str(ctx.get("message", "")) for ctx in handler_contexts
        )

    asyncio.run(_run())


def test_coordinator_stream_stage_failure_cancels_future() -> None:
    """A stage failure on a streaming request cancels the completion future
    (which the stream consumer never awaits) rather than setting an exception
    that would be reported as never retrieved."""

    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode"],
        )
        coordinator.control_plane = RecordingCoordinatorControlPlane()
        coordinator.register_stage("preprocess", "inproc://preprocess")

        task, error_sink, future = await _drive_stream_until_registered(
            coordinator, "req-1"
        )

        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", False, error="boom")
        )
        await asyncio.wait_for(task, timeout=1)

        assert error_sink == ["boom"]
        assert future.cancelled() is True
        assert "req-1" not in coordinator._completion_futures

    asyncio.run(_run())


def test_duplicate_submit_does_not_clean_up_existing_request() -> None:
    async def _run() -> None:
        control_plane = RecordingCoordinatorControlPlane()
        coordinator = _coordinator_with_control_plane(control_plane)
        await coordinator._submit_request("shared", "winner")
        winner_future = coordinator._completion_futures["shared"]

        with pytest.raises(ValueError, match="already exists"):
            await coordinator.submit("shared", "loser")

        assert coordinator._completion_futures["shared"] is winner_future
        assert "shared" in coordinator._requests
        assert control_plane.aborts == []

        assert await coordinator.abort("shared") is True
        with pytest.raises(asyncio.CancelledError):
            await winner_future

    asyncio.run(_run())


def test_submit_cancellation_aborts_once_and_cleans_owned_state() -> None:
    async def _run() -> None:
        control_plane = SignallingCoordinatorControlPlane()
        coordinator = _coordinator_with_control_plane(control_plane)
        task = asyncio.create_task(coordinator.submit("req-1", "hello"))

        await control_plane.submitted_event.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert [msg.request_id for msg in control_plane.aborts] == ["req-1"]
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._completion_futures

    asyncio.run(_run())


def test_submit_transport_failure_cleans_owned_state() -> None:
    async def _run() -> None:
        control_plane = FailingSubmissionControlPlane()
        coordinator = _coordinator_with_control_plane(control_plane)

        with pytest.raises(RuntimeError, match="submit transport failed"):
            await coordinator.submit("req-1", "hello")

        assert [msg.request_id for msg in control_plane.aborts] == ["req-1"]
        assert "req-1" not in coordinator._requests
        assert "req-1" not in coordinator._completion_futures

    asyncio.run(_run())


def test_completion_future_reserves_request_id_until_waiter_cleanup() -> None:
    async def _run() -> None:
        control_plane = SignallingCoordinatorControlPlane()
        coordinator = _coordinator_with_control_plane(control_plane)
        task = asyncio.create_task(coordinator.submit("req-1", "first"))

        await control_plane.submitted_event.wait()
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result="done")
        )
        assert "req-1" not in coordinator._requests
        assert "req-1" in coordinator._completion_futures

        with pytest.raises(ValueError, match="already exists"):
            await coordinator._submit_request("req-1", "too early")

        assert await task == "done"
        await coordinator._submit_request("req-1", "reused")
        assert await coordinator.abort("req-1") is True

    asyncio.run(_run())


def test_completion_future_reserves_id_until_abort_publication_finishes() -> None:
    async def _run() -> None:
        control_plane = BlockingAbortControlPlane()
        coordinator = _coordinator_with_control_plane(control_plane)
        task = asyncio.create_task(coordinator.submit("req-1", "first"))

        await control_plane.submitted_event.wait()
        task.cancel()
        await control_plane.abort_started.wait()

        future = coordinator._completion_futures["req-1"]
        assert future.cancelled()
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result="late")
        )
        assert "req-1" not in coordinator._requests
        with pytest.raises(ValueError, match="already exists"):
            await coordinator._submit_request("req-1", "too early")

        control_plane.release_abort.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "req-1" not in coordinator._completion_futures

        await coordinator._submit_request("req-1", "safe reuse")
        assert await coordinator.abort("req-1") is True

    asyncio.run(_run())


def test_stream_close_aborts_once_but_natural_completion_does_not() -> None:
    async def _run() -> None:
        close_control_plane = SignallingCoordinatorControlPlane()
        close_coordinator = _coordinator_with_control_plane(close_control_plane)
        stream = close_coordinator.stream("early", "hello")
        next_event = asyncio.create_task(anext(stream))

        await close_control_plane.submitted_event.wait()
        await close_coordinator._handle_stream(
            StreamMessage(
                request_id="early",
                from_stage="decode",
                chunk={"text": "partial"},
                modality="text",
                chunk_id=0,
            )
        )
        assert (await next_event).chunk == {"text": "partial"}
        await stream.aclose()

        assert [msg.request_id for msg in close_control_plane.aborts] == ["early"]
        assert "early" not in close_coordinator._requests
        assert "early" not in close_coordinator._stream_queues
        assert "early" not in close_coordinator._completion_futures

        complete_control_plane = SignallingCoordinatorControlPlane()
        complete_coordinator = _coordinator_with_control_plane(complete_control_plane)
        events: list[CompleteMessage | StreamMessage] = []

        async def _consume() -> None:
            async for event in complete_coordinator.stream("natural", "hello"):
                events.append(event)

        consume_task = asyncio.create_task(_consume())
        await complete_control_plane.submitted_event.wait()
        await complete_coordinator._handle_completion(
            CompleteMessage("natural", "decode", True, result="done")
        )
        await consume_task

        assert len(events) == 1
        assert isinstance(events[0], CompleteMessage)
        assert events[0].result == "done"
        assert complete_control_plane.aborts == []
        assert "natural" not in complete_coordinator._requests
        assert "natural" not in complete_coordinator._stream_queues
        assert "natural" not in complete_coordinator._completion_futures

    asyncio.run(_run())


def test_abort_transport_failure_preserves_cancellation_and_id_ownership(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _run() -> None:
        control_plane = SignallingCoordinatorControlPlane()
        control_plane.fail_abort = True
        coordinator = _coordinator_with_control_plane(control_plane)
        task = asyncio.create_task(coordinator.submit("req-1", "hello"))

        await control_plane.submitted_event.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert [msg.request_id for msg in control_plane.aborts] == ["req-1"]
        assert "req-1" in coordinator._requests
        assert "req-1" not in coordinator._completion_futures
        with pytest.raises(ValueError, match="already exists"):
            await coordinator._submit_request("req-1", "unsafe reuse")

        control_plane.fail_abort = False
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result="late")
        )
        await coordinator._submit_request("req-1", "safe reuse")
        assert await coordinator.abort("req-1") is True

    with caplog.at_level("WARNING"):
        asyncio.run(_run())

    assert "Failed to abort request req-1 after waiter interruption" in caplog.text
