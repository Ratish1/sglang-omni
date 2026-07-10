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


def test_comm_engine_releases_completed_send_job_while_idle() -> None:
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
        payload_ref = weakref.ref(payload)

        try:
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
            completion_task = engine._pending[data_ref.object_id].task
            assert completion_task is not None

            engine.ack_transfer(
                DataAckMessage(
                    request_id="req-1",
                    from_stage="receiver",
                    to_stage="sender",
                    object_id=data_ref.object_id,
                )
            )
            await completion_task

            del payload
            await asyncio.sleep(0)
            gc.collect()

            assert payload_ref() is None
        finally:
            engine.close()

    asyncio.run(_run())


@pytest.mark.parametrize("kind", ["payload", "stream"])
def test_comm_engine_starts_ack_timeout_after_control_publication(kind: str) -> None:
    async def _run() -> None:
        class _BlockedControlPlane:
            def __init__(self) -> None:
                self.send_started = asyncio.Event()
                self.release_send = asyncio.Event()
                self.message: DataReadyMessage | None = None

            async def send_to_stage(self, target, endpoint, msg) -> None:
                del target, endpoint
                self.message = msg
                self.send_started.set()
                await self.release_send.wait()

        relay = _AckedRelay()
        control_plane = _BlockedControlPlane()
        engine = CommEngine(
            CommRouter(
                stage_name="sender",
                gpu_id=None,
                same_process_targets=set(),
                gpu_stage_names=set(),
                comm_config={"ack_timeout_s": 0.01},
                injected_relay=relay,
            )
        )

        try:
            if kind == "payload":
                send_coro = engine.send_payload(
                    relay=relay,
                    control_plane=control_plane,
                    request_id="req-1",
                    payload=make_stage_payload(
                        request_id="req-1", data={"x": torch.ones(2)}
                    ),
                    transport=TransportKind.SHM,
                    from_stage="sender",
                    to_stage="receiver",
                    target_endpoint="inproc://receiver",
                )
            else:
                send_coro = engine.send_stream_chunk(
                    relay=relay,
                    control_plane=control_plane,
                    request_id="req-1",
                    data=torch.ones(2),
                    target_stage="receiver",
                    target_endpoint="inproc://receiver",
                    from_stage="sender",
                    chunk_id=0,
                    metadata=None,
                    transport=TransportKind.SHM,
                )
            send_task = asyncio.create_task(send_coro)
            await control_plane.send_started.wait()
            await asyncio.sleep(0.05)

            op = relay.ops[0]
            assert op.failed is None
            assert not op.waited

            control_plane.release_send.set()
            result = await send_task
            assert control_plane.message is not None
            data_ref = (
                result
                if kind == "payload"
                else DataRef.from_dict(control_plane.message.data_ref)
            )
            assert isinstance(data_ref, DataRef)
            completion_task = engine._pending[data_ref.object_id].task
            assert completion_task is not None

            engine.ack_transfer(
                DataAckMessage(
                    request_id="req-1",
                    from_stage="receiver",
                    to_stage="sender",
                    object_id=data_ref.object_id,
                )
            )
            await completion_task
            assert op.acked
            assert op.waited
        finally:
            engine.close()

    asyncio.run(_run())


def test_comm_engine_accepts_ack_during_control_publication() -> None:
    async def _run() -> None:
        class _ImmediateAckControlPlane:
            def __init__(self) -> None:
                self.engine: CommEngine | None = None

            async def send_to_stage(self, target, endpoint, msg) -> None:
                del target, endpoint
                assert self.engine is not None
                data_ref = DataRef.from_dict(msg.data_ref)
                self.engine.ack_transfer(
                    DataAckMessage(
                        request_id=msg.request_id,
                        from_stage=msg.to_stage,
                        to_stage=msg.from_stage,
                        object_id=data_ref.object_id,
                    )
                )

        relay = _AckedRelay()
        control_plane = _ImmediateAckControlPlane()
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
        control_plane.engine = engine

        try:
            await engine.send_payload(
                relay=relay,
                control_plane=control_plane,
                request_id="req-1",
                payload=make_stage_payload(
                    request_id="req-1", data={"x": torch.ones(2)}
                ),
                transport=TransportKind.SHM,
                from_stage="sender",
                to_stage="receiver",
                target_endpoint="inproc://receiver",
            )
            await _wait_until(lambda: relay.ops[0].waited)

            assert relay.ops[0].acked
            assert relay.ops[0].failed is None
        finally:
            engine.close()

    asyncio.run(_run())


@pytest.mark.parametrize("kind", ["payload", "stream"])
def test_comm_engine_settles_operations_when_control_publication_fails(
    kind: str,
) -> None:
    async def _run() -> None:
        class _FailingControlPlane:
            async def send_to_stage(self, target, endpoint, msg) -> None:
                del target, endpoint, msg
                raise RuntimeError("control publication failed")

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

        try:
            if kind == "payload":
                send_coro = engine.send_payload(
                    relay=relay,
                    control_plane=_FailingControlPlane(),
                    request_id="req-1",
                    payload=make_stage_payload(
                        request_id="req-1", data={"x": torch.ones(2)}
                    ),
                    transport=TransportKind.SHM,
                    from_stage="sender",
                    to_stage="receiver",
                    target_endpoint="inproc://receiver",
                )
            else:
                send_coro = engine.send_stream_chunk(
                    relay=relay,
                    control_plane=_FailingControlPlane(),
                    request_id="req-1",
                    data=torch.ones(2),
                    target_stage="receiver",
                    target_endpoint="inproc://receiver",
                    from_stage="sender",
                    chunk_id=0,
                    metadata={"hidden": torch.ones(1)},
                    transport=TransportKind.SHM,
                )
            with pytest.raises(RuntimeError, match="control publication failed"):
                await send_coro
            await _wait_until(lambda: all(op.waited for op in relay.ops))

            assert all(isinstance(op.failed, RuntimeError) for op in relay.ops)
            assert engine._pending == {}
        finally:
            engine.close()

    asyncio.run(_run())


def test_comm_engine_threads_endpoint_identity_to_stream_metadata_ops() -> None:
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

        await engine.send_stream_chunk(
            relay=relay,
            control_plane=control_plane,
            request_id="req-1",
            data=torch.ones(2),
            target_stage="receiver",
            target_endpoint="inproc://receiver",
            from_stage="sender",
            chunk_id=0,
            metadata={"hidden": torch.ones(1)},
            transport=TransportKind.SHM,
        )

        assert relay.receiver_ids == ["inproc://receiver", "inproc://receiver"]
        _, _, msg = control_plane.sent_to_stage[0]
        data_ref = DataRef.from_dict(msg.data_ref)
        completion_task = engine._pending[data_ref.object_id].task
        assert completion_task is not None

        engine.ack_transfer(
            DataAckMessage(
                request_id="req-1",
                from_stage="receiver",
                to_stage="sender",
                object_id=data_ref.object_id,
            )
        )
        await completion_task
        assert all(op.acked and op.waited for op in relay.ops)

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
        "metadata_tensors": [],
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
