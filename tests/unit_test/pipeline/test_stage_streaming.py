# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataKind, DataRef, TransportKind
from sglang_omni.comm.engine import CommEngine
from sglang_omni.config.schema import StageConfig
from sglang_omni.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage.stream_queue import StreamQueue
from sglang_omni.proto import (
    DataAckMessage,
    DataReadyMessage,
    OmniRequest,
    StagePayload,
)
from sglang_omni.relay.shm import ShmRelay
from sglang_omni.scheduling.messages import OutgoingMessage


class _FakeControlPlane:
    recv_endpoint = "inproc://stage"

    def __init__(self) -> None:
        self.streams = []
        self.stage_messages = []
        self.completions = []

    async def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def send_stream(self, msg) -> None:
        self.streams.append(msg)

    async def send_to_stage(self, target, endpoint, msg) -> None:
        self.stage_messages.append((target, endpoint, msg))

    async def send_complete(self, msg) -> None:
        self.completions.append(msg)


class _FakeRelay:
    def __init__(self) -> None:
        self.device = "cpu"
        self.puts = []

    async def put_async(self, tensor, request_id, receiver_id=None):
        del receiver_id
        self.puts.append((request_id, tensor))
        return _DoneOp(tensor.numel())

    def close(self) -> None:
        pass

    def cleanup(self, request_id: str) -> None:
        pass


class _DoneOp:
    def __init__(self, size: int = 1) -> None:
        self.metadata = {"transfer_info": {"size": size}}

    async def wait_for_completion(self) -> None:
        pass

    def mark_receiver_done(self) -> None:
        pass

    def mark_receiver_failed(self, exc: BaseException) -> None:
        raise exc


class _AbortOnReadRelay(_FakeRelay):
    def __init__(self, on_wait) -> None:
        super().__init__()
        self._on_wait = on_wait
        self.gets = 0

    async def get_async(self, metadata, dest_tensor, request_id):
        del metadata, dest_tensor, request_id
        self.gets += 1
        return _CallbackOp(self._on_wait)


class _CallbackOp:
    def __init__(self, on_wait) -> None:
        self._on_wait = on_wait

    async def wait_for_completion(self) -> None:
        self._on_wait()

    def mark_receiver_done(self) -> None:
        pass

    def mark_receiver_failed(self, exc: BaseException) -> None:
        raise exc


# Stream chunks now always move through the transport router's relay (CUDA-IPC on
# GPU, shm on CPU). These helpers build real relay-backed messages over an
# in-process ShmRelay so the receive path is exercised end to end, matching what
# ``write_stream_chunk`` / ``write_payload`` produce on the wire.


async def _make_relay_chunk(
    relay,
    *,
    request_id: str,
    from_stage: str,
    to_stage: str,
    chunk_id: int,
    data,
    metadata: dict | None = None,
) -> DataReadyMessage:
    object_id = f"{request_id}:stream:{from_stage}:{to_stage}:{chunk_id}"
    data_ref, op = await stage_io.write_tensor(
        relay,
        object_id,
        data,
        transport=TransportKind.SHM,
        kind=DataKind.STREAM_CHUNK,
        request_id=request_id,
        from_stage=from_stage,
        to_stage=to_stage,
    )
    pending_ops = [op]
    data_ref = await stage_io._with_stream_metadata(
        relay,
        data_ref,
        metadata,
        TransportKind.SHM,
        pending_ops,
    )
    return DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage=to_stage,
        data_ref=data_ref.to_dict(),
        chunk_id=chunk_id,
    )


async def _make_relay_payload(
    relay,
    payload: StagePayload,
    *,
    from_stage: str = "tts_engine",
    to_stage: str = "vocoder",
) -> DataReadyMessage:
    data_ref, op = await stage_io.write_payload(
        relay,
        payload.request_id,
        payload,
        transport=TransportKind.SHM,
    )
    return DataReadyMessage(
        request_id=payload.request_id,
        from_stage=from_stage,
        to_stage=to_stage,
        data_ref=data_ref.to_dict(),
    )


def _make_receive_stage() -> Stage:
    scheduler = SimpleNamespace(
        inbox=queue.Queue(),
        outbox=queue.Queue(),
        abort=lambda request_id: None,
        stop=lambda: None,
    )
    return Stage(
        name="receiver",
        role="single",
        get_next=lambda request_id, output: None,
        gpu_id=None,
        endpoints={"upstream": "inproc://upstream"},
        control_plane=_FakeControlPlane(),
        relay=_FakeRelay(),
        scheduler=scheduler,
    )


def _receive_message(
    *,
    request_id: str = "req",
    from_stage: str = "upstream",
    chunk_id: int | None = None,
    is_done: bool = False,
    error: str | None = None,
) -> DataReadyMessage:
    is_signal = is_done or error is not None
    return DataReadyMessage(
        request_id=request_id,
        from_stage=from_stage,
        to_stage="receiver",
        data_ref=None if is_signal else {"test": True},
        chunk_id=chunk_id,
        is_done=is_done,
        error=error,
    )


def test_stage_preserves_receive_order_per_source(monkeypatch) -> None:
    async def _run() -> None:
        stage = _make_receive_stage()
        payload_started = asyncio.Event()
        release_payload = asyncio.Event()
        chunk_zero_started = asyncio.Event()
        release_chunk_zero = asyncio.Event()
        committed: list[str] = []

        async def receive_payload(msg) -> None:
            del msg
            payload_started.set()
            await release_payload.wait()
            committed.append("payload")

        async def receive_chunk(msg) -> None:
            if msg.chunk_id == 0:
                chunk_zero_started.set()
                await release_chunk_zero.wait()
            committed.append(f"chunk-{msg.chunk_id}")

        async def receive_signal(msg) -> None:
            assert msg.is_done
            committed.append("done")

        monkeypatch.setattr(stage, "_on_data_ready", receive_payload)
        monkeypatch.setattr(stage, "_on_stream_chunk", receive_chunk)
        monkeypatch.setattr(stage, "_on_stream_signal", receive_signal)

        await stage._handle_message(_receive_message())
        await stage._handle_message(_receive_message(chunk_id=0))
        await stage._handle_message(_receive_message(chunk_id=1))
        await stage._handle_message(_receive_message(is_done=True))
        tail = stage._receive_tails[("req", "upstream")]

        await asyncio.wait_for(payload_started.wait(), timeout=1.0)
        assert not chunk_zero_started.is_set()
        assert committed == []

        release_payload.set()
        await asyncio.wait_for(chunk_zero_started.wait(), timeout=1.0)
        assert committed == ["payload"]

        release_chunk_zero.set()
        await asyncio.wait_for(asyncio.shield(tail), timeout=1.0)
        await asyncio.sleep(0)

        assert committed == ["payload", "chunk-0", "chunk-1", "done"]
        assert stage._receive_tasks == set()
        assert stage._receive_tails == {}

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("blocked_key", "independent_key"),
    [
        (("req", "source-a"), ("req", "source-b")),
        (("req-a", "source"), ("req-b", "source")),
    ],
)
def test_stage_receive_order_isolated_per_request_and_source(
    monkeypatch,
    blocked_key,
    independent_key,
) -> None:
    async def _run() -> None:
        stage = _make_receive_stage()
        blocked_started = asyncio.Event()
        release_blocked = asyncio.Event()
        independent_done = asyncio.Event()

        async def receive_payload(msg) -> None:
            key = (msg.request_id, msg.from_stage)
            if key == blocked_key:
                blocked_started.set()
                await release_blocked.wait()
            if key == independent_key:
                independent_done.set()

        monkeypatch.setattr(stage, "_on_data_ready", receive_payload)

        await stage._handle_message(
            _receive_message(
                request_id=blocked_key[0],
                from_stage=blocked_key[1],
            )
        )
        blocked_task = stage._receive_tails[blocked_key]
        await asyncio.wait_for(blocked_started.wait(), timeout=1.0)

        await stage._handle_message(
            _receive_message(
                request_id=independent_key[0],
                from_stage=independent_key[1],
            )
        )
        independent_task = stage._receive_tails[independent_key]
        await asyncio.wait_for(independent_done.wait(), timeout=1.0)
        assert not blocked_task.done()

        release_blocked.set()
        await asyncio.wait_for(
            asyncio.gather(blocked_task, independent_task), timeout=1.0
        )

    asyncio.run(_run())


def test_stage_data_ack_bypasses_blocked_receive_order(monkeypatch) -> None:
    async def _run() -> None:
        stage = _make_receive_stage()
        receive_started = asyncio.Event()
        release_receive = asyncio.Event()
        acknowledgements = []

        async def receive_payload(msg) -> None:
            del msg
            receive_started.set()
            await release_receive.wait()

        monkeypatch.setattr(stage, "_on_data_ready", receive_payload)
        monkeypatch.setattr(
            stage._comm,
            "ack_transfer",
            lambda ack: acknowledgements.append(ack),
        )

        await stage._handle_message(_receive_message())
        receive_task = stage._receive_tails[("req", "upstream")]
        await asyncio.wait_for(receive_started.wait(), timeout=1.0)

        ack = DataAckMessage(
            request_id="req",
            from_stage="downstream",
            to_stage="receiver",
            object_id="object-0",
        )
        await stage._handle_message(ack)
        assert acknowledgements == [ack]
        assert not receive_task.done()

        release_receive.set()
        await asyncio.wait_for(receive_task, timeout=1.0)

    asyncio.run(_run())


def test_stage_receive_tail_cleanup_keeps_newest_task(monkeypatch) -> None:
    async def _run() -> None:
        stage = _make_receive_stage()
        second_started = asyncio.Event()
        release_second = asyncio.Event()
        call_count = 0

        async def receive_payload(msg) -> None:
            nonlocal call_count
            del msg
            call_count += 1
            if call_count == 2:
                second_started.set()
                await release_second.wait()

        monkeypatch.setattr(stage, "_on_data_ready", receive_payload)

        await stage._handle_message(_receive_message())
        first_task = stage._receive_tails[("req", "upstream")]
        await stage._handle_message(_receive_message())
        second_task = stage._receive_tails[("req", "upstream")]

        await asyncio.wait_for(second_started.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert first_task.done()
        assert stage._receive_tails[("req", "upstream")] is second_task

        release_second.set()
        await asyncio.wait_for(second_task, timeout=1.0)
        await asyncio.sleep(0)
        assert stage._receive_tails == {}

    asyncio.run(_run())


def test_stage_stop_cancels_and_clears_receive_order(monkeypatch) -> None:
    async def _run() -> None:
        stage = _make_receive_stage()
        first_started = asyncio.Event()
        never_release = asyncio.Event()

        async def receive_payload(msg) -> None:
            del msg
            first_started.set()
            await never_release.wait()

        monkeypatch.setattr(stage, "_on_data_ready", receive_payload)

        await stage._handle_message(_receive_message())
        first_task = stage._receive_tails[("req", "upstream")]
        await stage._handle_message(_receive_message())
        second_task = stage._receive_tails[("req", "upstream")]
        await asyncio.wait_for(first_started.wait(), timeout=1.0)

        await asyncio.wait_for(stage.stop(), timeout=1.0)

        assert first_task.cancelled()
        assert second_task.cancelled()
        assert stage._receive_tasks == set()
        assert stage._receive_tails == {}

    asyncio.run(_run())


def test_terminal_scheduler_stream_routes_to_coordinator() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=_FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage._active_requests.add("req")
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data={"audio_data": [0.1], "modality": "audio"},
            )
        )
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data={"audio_data": [0.2], "modality": "audio"},
            )
        )

        await stage._drain_outbox_external()

        assert len(control_plane.streams) == 2
        msg = control_plane.streams[0]
        assert msg.request_id == "req"
        assert msg.from_stage == "vocoder"
        assert msg.chunk == {"audio_data": [0.1], "modality": "audio"}
        assert msg.modality == "audio"
        assert [msg.chunk_id for msg in control_plane.streams] == [0, 1]

    asyncio.run(_run())


def test_explicit_scheduler_stream_target_keeps_stage_to_stage_routing() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        scheduler = SimpleNamespace(outbox=queue.Queue())
        codes = torch.empty(11, 1, dtype=torch.long)
        stage = Stage(
            name="tts_engine",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"vocoder": "inproc://vocoder"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._active_requests.add("req")
        scheduler.outbox.put(
            OutgoingMessage(
                request_id="req",
                type="stream",
                data=codes,
                target="vocoder",
                metadata={"modality": "audio_codes"},
            )
        )

        await stage._drain_outbox_external()

        assert control_plane.streams == []
        assert len(relay.puts) == 1
        assert len(control_plane.stage_messages) == 1
        target, endpoint, msg = control_plane.stage_messages[0]
        assert target == "vocoder"
        assert endpoint == "inproc://vocoder"
        assert msg.request_id == "req"
        assert msg.from_stage == "tts_engine"
        assert msg.to_stage == "vocoder"
        assert msg.chunk_id == 0
        assert DataRef.from_dict(msg.data_ref).metadata == {"modality": "audio_codes"}

    asyncio.run(_run())


def test_stage_config_rejects_unknown_model_transport_field() -> None:
    field_name = "stream_" + "transport"
    with pytest.raises(ValidationError):
        StageConfig(
            name="tts_engine",
            factory="pkg.create",
            next="vocoder",
            stream_to=["vocoder"],
            **{field_name: {"vocoder": "relay"}},
        )


def test_s2pro_config_declares_topology_without_transport_policy() -> None:
    config = S2ProPipelineConfig(model_path="dummy")
    tts_stage = next(stage for stage in config.stages if stage.name == "tts_engine")
    vocoder_stage = next(stage for stage in config.stages if stage.name == "vocoder")
    assert tts_stage.stream_to == ["vocoder"]
    assert vocoder_stage.can_accept_stream_before_payload
    assert "stream_transport" not in StageConfig.model_fields


def test_stage_fails_pre_payload_stream_chunk_by_default() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        codes = torch.arange(11, dtype=torch.float32)

        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
            )
        )

        assert scheduler.inbox.empty()
        assert len(control_plane.completions) == 1
        assert control_plane.completions[0].success is False
        assert "pre-payload stream data" in control_plane.completions[0].error

    asyncio.run(_run())


def test_stage_routes_stream_chunk_after_payload_by_default() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={"ready": True},
        )
        await stage._on_data_ready(await _make_relay_payload(relay, payload))
        codes = torch.arange(11, dtype=torch.float32)
        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
            )
        )
        payload_msg = scheduler.inbox.get_nowait()
        chunk_msg = scheduler.inbox.get_nowait()
        assert payload_msg.type == "new_request"
        assert chunk_msg.type == "stream_chunk"
        assert torch.equal(chunk_msg.data.data, codes)

    asyncio.run(_run())


def test_stage_routes_pre_payload_stream_events_for_capable_receiver() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
            can_accept_stream_before_payload=True,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        codes = torch.arange(11, dtype=torch.float32)

        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
                metadata={"modality": "audio_codes"},
            )
        )

        chunk_msg = scheduler.inbox.get_nowait()
        assert chunk_msg.request_id == "req"
        assert chunk_msg.type == "stream_chunk"
        assert torch.equal(chunk_msg.data.data, codes)
        assert chunk_msg.data.metadata == {"modality": "audio_codes"}

        await stage._on_stream_signal(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                data_ref=None,
                is_done=True,
            )
        )
        early_done_msg = scheduler.inbox.get_nowait()
        assert early_done_msg.request_id == "req"
        assert early_done_msg.type == "stream_done"

        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={"ready": True},
        )
        await stage._on_data_ready(await _make_relay_payload(relay, payload))
        payload_msg = scheduler.inbox.get_nowait()
        assert payload_msg.request_id == "req"
        assert payload_msg.type == "new_request"
        assert payload_msg.data.data == {"ready": True}

    asyncio.run(_run())


def test_stage_stream_chunk_received_after_relay_materialization(monkeypatch) -> None:
    """Cross-GPU chunks emit receive only after relay data and metadata are restored."""
    order: list[str] = []

    async def fake_read_stream_chunk(self, *, relay, data_ref):
        del self, relay, data_ref
        order.append("stream_chunk_read")
        return "chunk", {"modality": "audio_codes"}

    async def fake_route(self, request_id, item):
        del self, request_id, item
        order.append("routed")

    monkeypatch.setattr(CommEngine, "read_stream_chunk", fake_read_stream_chunk)
    monkeypatch.setattr(Stage, "_route_stream_item_or_fail", fake_route)
    monkeypatch.setattr(
        "sglang_omni.pipeline.stage.runtime._emit_event",
        lambda **kwargs: order.append(kwargs["event_name"]),
    )

    async def _run() -> None:
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=_FakeControlPlane(),
            relay=_FakeRelay(),
            scheduler=SimpleNamespace(outbox=queue.Queue()),
            can_accept_stream_before_payload=True,
        )
        await stage._on_stream_chunk(
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                data_ref=(
                    await _make_relay_chunk(
                        _FakeRelay(),
                        request_id="req",
                        from_stage="tts_engine",
                        to_stage="vocoder",
                        chunk_id=0,
                        data=torch.empty(1),
                    )
                ).data_ref,
                chunk_id=0,
            )
        )

    asyncio.run(_run())

    assert order == ["stream_chunk_read", "stage_stream_chunk_received", "routed"]


def test_stage_stream_error_fails_request_even_with_stream_queue() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            aborted=[],
            abort=lambda request_id: scheduler.aborted.append(request_id),
        )
        stage = Stage(
            name="decode",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=_FakeRelay(),
            scheduler=scheduler,
            is_terminal=True,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        stage._stream_queue.open("req")

        await stage._queue_stream_error(
            "req",
            from_stage="thinker",
            error=RuntimeError("stream failed"),
        )

        assert scheduler.aborted == ["req"]
        assert len(control_plane.completions) == 1
        assert control_plane.completions[0].success is False
        assert control_plane.completions[0].error == "stream failed"
        assert not stage._stream_queue.has("req")
        assert "req" in stage._aborted

    asyncio.run(_run())


def test_write_stream_chunk_uses_relay() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        relay = _FakeRelay()
        codes = torch.empty(11, 1, dtype=torch.long)

        data_ref, ops = await stage_io.write_stream_chunk(
            relay,
            request_id="req",
            data=codes,
            target_stage="vocoder",
            from_stage="tts_engine",
            chunk_id=0,
            transport=TransportKind.SHM,
        )
        await control_plane.send_to_stage(
            "vocoder",
            "inproc://vocoder",
            DataReadyMessage(
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                data_ref=data_ref.to_dict(),
                chunk_id=0,
            ),
        )
        for op in ops:
            op.mark_receiver_done()
            await op.wait_for_completion()

        assert len(relay.puts) == 1
        assert relay.puts[0][0] == "req:stream:tts_engine:vocoder:0"
        assert len(control_plane.stage_messages) == 1
        _, _, msg = control_plane.stage_messages[0]
        expected_size = codes.contiguous().view(torch.uint8).numel()
        data_ref = DataRef.from_dict(msg.data_ref)
        assert data_ref.buffer.info == {"transfer_info": {"size": expected_size}}

    asyncio.run(_run())


def test_stage_drops_stream_chunk_after_abort_during_relay_read() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        codes = torch.empty(11, 1, dtype=torch.long)
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = _AbortOnReadRelay(lambda: stage._on_abort("req"))
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = None

        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
            )
        )

        assert scheduler.inbox.empty()
        assert relay.gets == 1

    asyncio.run(_run())


def test_stage_drains_relay_stream_chunk_for_already_aborted_request() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        codes = torch.empty(11, 1, dtype=torch.long)
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = _AbortOnReadRelay(lambda: None)
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._aborted.add("req")
        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
                metadata={"latency": torch.zeros(1)},
            )
        )

        assert scheduler.inbox.empty()
        assert relay.gets == 2

    asyncio.run(_run())


def test_stage_drains_relay_payload_for_already_aborted_request() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = _AbortOnReadRelay(lambda: None)
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._aborted.add("req")
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={},
        )

        await stage._on_data_ready(await _make_relay_payload(relay, payload))

        assert scheduler.inbox.empty()
        assert relay.gets == 1

    asyncio.run(_run())


def test_stage_routes_relay_stream_chunk_to_scheduler() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        codes = torch.arange(2048, dtype=torch.float32)
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = ShmRelay(engine_id="t", device="cpu")
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        stage._stream_queue = StreamQueue(max_pending=4096)
        stage._stream_queue.open("req")

        await stage._on_stream_chunk(
            await _make_relay_chunk(
                relay,
                request_id="req",
                from_stage="tts_engine",
                to_stage="vocoder",
                chunk_id=0,
                data=codes,
                metadata={"modality": "audio_codes"},
            )
        )

        queued = scheduler.inbox.get_nowait()
        assert queued.request_id == "req"
        assert queued.type == "stream_chunk"
        assert torch.equal(queued.data.data, codes)
        assert queued.data.metadata == {"modality": "audio_codes"}

    asyncio.run(_run())


def test_stage_drops_payload_after_abort_during_relay_read() -> None:
    async def _run() -> None:
        control_plane = _FakeControlPlane()
        scheduler = SimpleNamespace(
            outbox=queue.Queue(),
            inbox=queue.Queue(),
            abort=lambda request_id: None,
        )
        relay = _AbortOnReadRelay(lambda: stage._on_abort("req"))
        stage = Stage(
            name="vocoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=None,
            endpoints={"tts_engine": "inproc://tts_engine"},
            control_plane=control_plane,
            relay=relay,
            scheduler=scheduler,
        )
        payload = StagePayload(
            request_id="req",
            request=OmniRequest(inputs="hello"),
            data={},
        )

        await stage._on_data_ready(await _make_relay_payload(relay, payload))

        assert scheduler.inbox.empty()

    asyncio.run(_run())
