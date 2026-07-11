# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc
import weakref
from typing import Any

import pytest
import torch

from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter
from sglang_omni.proto import DataAckMessage, DataReadyMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    RecordingStageControlPlane,
    make_stage_payload,
)


class _AckedOp:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self._metadata = metadata
        self.acked = False
        self.waited = False
        self.failed: BaseException | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def mark_receiver_done(self) -> None:
        self.acked = True

    def mark_receiver_failed(self, exc: BaseException) -> None:
        self.failed = exc

    async def wait_for_completion(self, timeout: float = 30.0) -> None:
        del timeout
        self.waited = True
        if self.failed is not None:
            raise self.failed
        if not self.acked:
            raise RuntimeError("waited before receiver ack")


class _AckedRelay:
    device = "cpu"

    def __init__(self) -> None:
        self.storage: dict[str, torch.Tensor] = {}
        self.ops: list[_AckedOp] = []
        self.receiver_ids: list[str | None] = []

    async def put_async(
        self,
        tensor: torch.Tensor,
        request_id: str | None = None,
        dst_rank: int | None = None,
        receiver_id: str | None = None,
    ) -> _AckedOp:
        del dst_rank
        self.receiver_ids.append(receiver_id)
        key = str(request_id)
        self.storage[key] = tensor.detach().clone()
        op = _AckedOp({"transfer_info": {"size": int(tensor.numel())}, "key": key})
        self.ops.append(op)
        return op

    async def get_async(
        self,
        metadata: dict[str, Any],
        dest_tensor: torch.Tensor,
        request_id: str | None = None,
    ) -> _AckedOp:
        key = str(metadata.get("key", request_id))
        stored = self.storage[key]
        dest_tensor.reshape(-1)[: stored.numel()].copy_(stored.reshape(-1))
        return _AckedOp(metadata)

    def cleanup(self, request_id: str) -> None:
        del request_id

    def close(self) -> None:
        pass


class _BlockedControlPlane(RecordingStageControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.failure: BaseException | None = None

    async def send_to_stage(self, target: str, endpoint: str, msg: Any) -> None:
        self.entered.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        await super().send_to_stage(target, endpoint, msg)


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("condition was not met")
        await asyncio.sleep(0)


def test_comm_engine_releases_sender_op_after_data_ack() -> None:
    async def _run() -> None:
        relay = _AckedRelay()
        control_plane = RecordingStageControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )
        payload = make_stage_payload(request_id="req-1", data={"x": torch.ones(2)})

        data_ref = await engine.send_payload(
            relay=relay,
            control_plane=control_plane,
            request_id="req-1",
            payload=payload,
            transport=TransportKind.SHM,
            from_stage="sender",
            to_stage="receiver",
            target_endpoint="inproc://receiver",
        )

        op = relay.ops[0]
        assert not op.waited
        target, _, msg = control_plane.sent_to_stage[0]
        assert target == "receiver"
        assert relay.receiver_ids == ["inproc://receiver"]
        assert DataRef.from_dict(msg.data_ref).object_id == data_ref.object_id

        engine.ack_transfer(
            DataAckMessage(
                request_id="req-1",
                from_stage="receiver",
                to_stage="sender",
                object_id=data_ref.object_id,
            )
        )
        await _wait_until(lambda: op.waited)
        assert op.acked

    asyncio.run(_run())


def test_comm_engine_starts_ack_timeout_after_publication() -> None:
    async def _run() -> None:
        relay = _AckedRelay()
        control_plane = _BlockedControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )
        payload = make_stage_payload(request_id="req-1", data={"x": torch.ones(2)})
        send = asyncio.create_task(
            engine.send_payload(
                relay=relay,
                control_plane=control_plane,
                request_id="req-1",
                payload=payload,
                transport=TransportKind.SHM,
                from_stage="sender",
                to_stage="receiver",
                target_endpoint="inproc://receiver",
            )
        )

        await control_plane.entered.wait()
        pending = next(iter(engine._pending.values()))
        assert pending.task is None

        control_plane.release.set()
        data_ref = await send
        assert engine._pending[data_ref.object_id].task is not None
        engine.ack_transfer(
            DataAckMessage(
                request_id="req-1",
                from_stage="receiver",
                to_stage="sender",
                object_id=data_ref.object_id,
            )
        )
        await _wait_until(lambda: data_ref.object_id not in engine._pending)

        failed_control_plane = _BlockedControlPlane()
        failed_control_plane.failure = RuntimeError("publication failed")
        failed_control_plane.release.set()
        with pytest.raises(RuntimeError, match="publication failed"):
            await engine.send_payload(
                relay=relay,
                control_plane=failed_control_plane,
                request_id="req-2",
                payload=make_stage_payload(
                    request_id="req-2",
                    data={"x": torch.ones(2)},
                ),
                transport=TransportKind.SHM,
                from_stage="sender",
                to_stage="receiver",
                target_endpoint="inproc://receiver",
            )
        await _wait_until(lambda: not engine._pending)
        assert relay.ops[-1].waited
        assert isinstance(relay.ops[-1].failed, RuntimeError)

    asyncio.run(_run())


def test_comm_engine_caller_cancellation_does_not_cancel_owned_send() -> None:
    async def _run() -> None:
        relay = _AckedRelay()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )

        payload_control = _BlockedControlPlane()
        payload_send = asyncio.create_task(
            engine.send_payload(
                relay=relay,
                control_plane=payload_control,
                request_id="payload",
                payload=make_stage_payload(
                    request_id="payload",
                    data={"x": torch.ones(2)},
                ),
                transport=TransportKind.SHM,
                from_stage="sender",
                to_stage="receiver",
                target_endpoint="inproc://receiver",
            )
        )
        await payload_control.entered.wait()
        payload_send.cancel()
        with pytest.raises(asyncio.CancelledError):
            await payload_send
        payload_control.release.set()
        await _wait_until(lambda: bool(payload_control.sent_to_stage))
        payload_ref = DataRef.from_dict(payload_control.sent_to_stage[0][2].data_ref)
        engine.ack_transfer(
            DataAckMessage(
                request_id="payload",
                from_stage="receiver",
                to_stage="sender",
                object_id=payload_ref.object_id,
            )
        )
        await _wait_until(lambda: payload_ref.object_id not in engine._pending)

        stream_control = _BlockedControlPlane()
        stream_send = asyncio.create_task(
            engine.send_stream_chunk(
                relay=relay,
                control_plane=stream_control,
                request_id="stream",
                data=torch.arange(4),
                target_stage="receiver",
                target_endpoint="inproc://receiver",
                from_stage="sender",
                chunk_id=0,
                metadata={"token_id": 1},
                transport=TransportKind.SHM,
            )
        )
        await stream_control.entered.wait()
        stream_send.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stream_send
        stream_control.release.set()
        await _wait_until(lambda: bool(stream_control.sent_to_stage))
        stream_ref = DataRef.from_dict(stream_control.sent_to_stage[0][2].data_ref)
        engine.ack_transfer(
            DataAckMessage(
                request_id="stream",
                from_stage="receiver",
                to_stage="sender",
                object_id=stream_ref.object_id,
            )
        )
        await _wait_until(lambda: stream_ref.object_id not in engine._pending)

        assert all(op.waited for op in relay.ops)
        assert not engine._pending
        assert not engine._send_workers["receiver"].done()

    asyncio.run(_run())


def test_comm_engine_idle_worker_releases_completed_send_job() -> None:
    async def _run() -> None:
        relay = _AckedRelay()
        control_plane = RecordingStageControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )
        tensor = torch.ones(2)
        tensor_ref = weakref.ref(tensor)
        payload = make_stage_payload(request_id="req", data={"x": tensor})
        data_ref = await engine.send_payload(
            relay=relay,
            control_plane=control_plane,
            request_id="req",
            payload=payload,
            transport=TransportKind.SHM,
            from_stage="sender",
            to_stage="receiver",
            target_endpoint="inproc://receiver",
        )
        engine.ack_transfer(
            DataAckMessage(
                request_id="req",
                from_stage="receiver",
                to_stage="sender",
                object_id=data_ref.object_id,
            )
        )
        await _wait_until(lambda: data_ref.object_id not in engine._pending)

        del tensor
        del payload
        gc.collect()

        assert tensor_ref() is None
        assert not engine._send_workers["receiver"].done()

    asyncio.run(_run())


def test_comm_engine_ignores_unknown_data_ack() -> None:
    engine = CommEngine(
        CommRouter(
            stage_name="sender",
            gpu_id=None,
            same_process_targets=set(),
            gpu_stage_names=set(),
        )
    )

    engine.ack_transfer(
        DataAckMessage(
            request_id="req-1",
            from_stage="receiver",
            to_stage="sender",
            object_id="missing",
        )
    )


def test_comm_engine_ignores_duplicate_data_ack() -> None:
    async def _run() -> None:
        relay = _AckedRelay()
        control_plane = RecordingStageControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 1.0},
                injected_relay=relay,
            )
        )
        payload = make_stage_payload(request_id="req-1", data={"x": torch.ones(2)})
        data_ref = await engine.send_payload(
            relay=relay,
            control_plane=control_plane,
            request_id="req-1",
            payload=payload,
            transport=TransportKind.SHM,
            from_stage="sender",
            to_stage="receiver",
            target_endpoint="inproc://receiver",
        )
        ack = DataAckMessage(
            request_id="req-1",
            from_stage="receiver",
            to_stage="sender",
            object_id=data_ref.object_id,
        )

        engine.ack_transfer(ack)
        await _wait_until(lambda: relay.ops[0].waited)

        engine.ack_transfer(ack)

    asyncio.run(_run())


def test_data_messages_reject_missing_data_ref() -> None:
    with pytest.raises(TypeError, match="data_ref"):
        DataReadyMessage(
            request_id="req-1",
            from_stage="a",
            to_stage="b",
            data_ref=None,
        ).to_dict()

    with pytest.raises(TypeError, match="success"):
        DataAckMessage.from_dict(
            {
                "type": "data_ack",
                "request_id": "req-1",
                "from_stage": "b",
                "to_stage": "a",
                "object_id": "obj",
            }
        )

    with pytest.raises(TypeError, match="is_done"):
        DataReadyMessage.from_dict(
            {
                "type": "data_ready",
                "request_id": "req-1",
                "from_stage": "a",
                "to_stage": "b",
                "is_done": "false",
            }
        )

    with pytest.raises(TypeError, match="chunk_id"):
        DataReadyMessage.from_dict(
            {
                "type": "data_ready",
                "request_id": "req-1",
                "from_stage": "a",
                "to_stage": "b",
                "data_ref": {"version": 1},
                "chunk_id": True,
            }
        )

    with pytest.raises(ValueError, match="both done and error"):
        DataReadyMessage(
            request_id="req-1",
            from_stage="a",
            to_stage="b",
            data_ref=None,
            is_done=True,
            error="boom",
        ).to_dict()


def test_data_ref_rejects_bool_int_fields() -> None:
    data_ref = {
        "_type": "DataRef",
        "version": 1,
        "kind": "stream_chunk",
        "object_id": "obj",
        "transport": "shm",
        "layout": "raw_tensor",
        "buffer": {"transport": "shm", "info": {}, "length": 1},
        "tensors": [],
        "shape": [1],
        "dtype": "torch.uint8",
        "offset": 0,
    }

    data_ref["version"] = True
    with pytest.raises(TypeError, match="version must be int"):
        DataRef.from_dict(data_ref)

    data_ref["version"] = 1
    data_ref["shape"] = [True]
    with pytest.raises(TypeError, match="shape must be list\\[int\\]"):
        DataRef.from_dict(data_ref)
