# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc
import logging
import multiprocessing as mp
import pickle
import traceback

import pytest
import torch

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataRef, TransportKind
from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.pipeline.local_dispatch import LocalStageDispatcher
from sglang_omni.pipeline.stage.input import AggregatedInput
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage.stream_queue import StreamQueue
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, _construct_stage
from sglang_omni.proto import DataAckMessage, DataReadyMessage
from tests.unit_test.fixtures.pipeline_fakes import (
    EventLog,
    FakeRelay,
    FakeScheduler,
    RecordingStageControlPlane,
    collect_event_names,
    fake_factory_path,
    make_noop_projector,
    make_result_message,
    make_stage_payload,
    make_stream_message,
    make_tensor_payload,
    tensor_equal,
)
from tests.unit_test.pipeline.helpers import make_stage


class _CloseAwareControlPlane(RecordingStageControlPlane):
    async def recv(self):
        while not self.closed:
            await asyncio.sleep(0)
        raise RuntimeError("control plane closed")


_DIRECT_IPC_PROCESS_TIMEOUT_S = 60.0


def _receive_direct_ipc_stream_chunk(
    data_ref: dict,
    metadata_kind: str,
    result_conn,
) -> None:
    try:
        torch.cuda.set_device(0)
        data, metadata = stage_io.deserialize_direct_cuda_ipc_stream_chunk(data_ref)

        assert data.is_cuda
        assert torch.equal(data, torch.arange(4, device=data.device))
        assert metadata is not None
        if metadata_kind == "cpu_tensor":
            assert metadata["stats"].device.type == "cpu"
            assert torch.equal(metadata["stats"], torch.ones(1))
        else:
            assert metadata == {"transcript": "x" * (128 * 1024)}

        torch.cuda.synchronize(data.device)
        del data, metadata
        gc.collect()
        torch.cuda.ipc_collect()
        result_conn.send(("ok", None))
    except BaseException:
        result_conn.send(("error", traceback.format_exc()))
        raise
    finally:
        result_conn.close()


def _receive_direct_ipc_payload(
    data_ref: dict,
    payload_kind: str,
    result_conn,
) -> None:
    try:
        torch.cuda.set_device(0)
        payload = stage_io.deserialize_direct_cuda_ipc_payload(data_ref)

        if payload_kind == "large_header":
            assert payload.request.inputs == "x" * (128 * 1024)
            gpu_tensor = payload.data["encoder_out"]
            assert torch.equal(
                gpu_tensor,
                torch.arange(4, device=gpu_tensor.device),
            )
        else:
            gpu_tensor = payload.data["gpu"]
            assert torch.equal(
                gpu_tensor,
                torch.arange(2, device=gpu_tensor.device),
            )
            assert payload.data["cpu"].device.type == "cpu"
            assert torch.equal(payload.data["cpu"], torch.ones(1))

        torch.cuda.synchronize(gpu_tensor.device)
        del gpu_tensor, payload
        gc.collect()
        torch.cuda.ipc_collect()
        result_conn.send(("ok", None))
    except BaseException:
        result_conn.send(("error", traceback.format_exc()))
        raise
    finally:
        result_conn.close()


def _run_direct_ipc_receiver(target, *args) -> None:
    ctx = mp.get_context("spawn")
    result_conn, child_result_conn = ctx.Pipe(duplex=False)
    process = ctx.Process(target=target, args=(*args, child_result_conn))
    try:
        process.start()
        child_result_conn.close()
        process.join(timeout=_DIRECT_IPC_PROCESS_TIMEOUT_S)
        if process.is_alive():
            process.terminate()
            process.join()
            raise AssertionError("direct CUDA IPC receiver timed out")

        status, detail = (
            result_conn.recv()
            if result_conn.poll()
            else ("error", "direct CUDA IPC receiver returned no result")
        )
        assert process.exitcode == 0, detail
        assert status == "ok", detail
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        result_conn.close()
        child_result_conn.close()
        process.close()


def test_aggregated_input_waits_per_request_without_cross_talk() -> None:
    """Preserves per-request fan-in isolation when requests interleave."""
    handler = AggregatedInput(
        {"preprocess", "image"},
        lambda payloads: make_stage_payload(data={"sources": sorted(payloads)}),
    )

    assert handler.receive("req-1", "preprocess", make_stage_payload()) is None
    assert handler.receive("req-2", "preprocess", make_stage_payload()) is None
    req2 = handler.receive("req-2", "image", make_stage_payload())
    req1 = handler.receive("req-1", "image", make_stage_payload())

    assert req2.data == {"sources": ["image", "preprocess"]}
    assert req1.data == {"sources": ["image", "preprocess"]}


def test_aggregated_input_supports_request_dynamic_source_sets() -> None:
    """Preserves early-arriving payloads while narrowing fan-in per request."""

    def _expected_sources(request_id, from_stage, payload):
        del request_id
        if from_stage != "preprocess":
            return None
        return payload.data["expected"]

    handler = AggregatedInput(
        {"preprocess", "image", "audio"},
        lambda payloads: make_stage_payload(data={"sources": sorted(payloads)}),
        expected_sources_fn=_expected_sources,
    )

    assert handler.receive("req-audio", "audio", make_stage_payload()) is None
    audio = handler.receive(
        "req-audio",
        "preprocess",
        make_stage_payload(data={"expected": ["preprocess", "audio"]}),
    )
    assert audio.data == {"sources": ["audio", "preprocess"]}

    text = handler.receive(
        "req-text",
        "preprocess",
        make_stage_payload(data={"expected": ["preprocess"]}),
    )
    assert text.data == {"sources": ["preprocess"]}


def test_aggregated_input_rejects_dynamic_sources_outside_static_fanin() -> None:
    def _invalid_sources(request_id, from_stage, payload):
        del request_id, from_stage, payload
        return ["preprocess", "audio"]

    handler = AggregatedInput(
        {"preprocess", "image"},
        lambda payloads: make_stage_payload(data={"sources": sorted(payloads)}),
        expected_sources_fn=_invalid_sources,
    )

    with pytest.raises(ValueError, match="outside static wait_for"):
        handler.receive("req-1", "preprocess", make_stage_payload())


def test_stage_routes_results_streams_and_clears_abort_state() -> None:
    """Preserves result routing, stream forwarding, and abort cleanup."""

    async def _run() -> None:
        relay = FakeRelay()
        scheduler = FakeScheduler()
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode", "talker": "inproc://talker"},
            project_payload={"decode": make_noop_projector("decode-only")},
            stream_targets=["talker"],
            relay=relay,
            scheduler=scheduler,
            control_plane=control_plane,
        )
        stage_obj._active_requests.add("req-1")
        scheduler.outbox.put(make_stream_message("req-1", data=torch.tensor([7])))
        scheduler.outbox.put(make_result_message("req-1", data={"answer": 1}))

        await stage_obj._drain_outbox()

        decode_msg = next(
            msg for target, _, msg in control_plane.sent_to_stage if target == "decode"
        )
        restored = await stage_io.read_payload(
            relay, "req-1", DataRef.from_dict(decode_msg.data_ref)
        )
        assert restored.data == {"marker": "decode-only", "data": {"answer": 1}}
        stream_msg = next(
            msg
            for target, _, msg in control_plane.sent_to_stage
            if target == "talker" and msg.chunk_id == 0
        )
        assert stream_msg.chunk_id == 0

        stage_obj._stream_queue = StreamQueue()
        stage_obj._stream_queue.open("req-1")
        stage_obj._on_abort("req-1")

        assert "req-1" in stage_obj._aborted
        assert relay.cleaned[-1] == "req-1"
        assert scheduler.aborted == ["req-1"]
        assert not stage_obj._stream_queue.has("req-1")

    asyncio.run(_run())


def test_stage_process_rejects_dynamic_targets_outside_static_topology() -> None:
    spec = StageLaunchConfig(
        stage_name="thinker",
        factory=fake_factory_path("make_scheduler"),
        next_stages=["decode"],
        route_fn=fake_factory_path("route_to_undeclared_talker"),
        stream_targets=["decode"],
        stream_done_to_fn=fake_factory_path("stream_done_to_undeclared_talker"),
        recv_endpoint="inproc://thinker",
        coordinator_endpoint="inproc://coordinator",
        abort_endpoint="inproc://abort",
        stage_endpoints={
            "decode": "inproc://decode",
            "talker": "inproc://talker",
        },
        comm_config={"slot_size_mb": 1},
    )
    stage_obj = _construct_stage(spec, logging.getLogger(__name__))
    payload = make_stage_payload()

    with pytest.raises(ValueError, match="route_fn.*outside the static topology"):
        stage_obj.get_next("req-1", payload)

    with pytest.raises(
        ValueError, match="stream_done_to_fn.*outside the static topology"
    ):
        stage_obj.get_stream_done_targets("req-1", payload)


def test_stage_process_rejects_dynamic_wait_sources_outside_static_fanin() -> None:
    spec = StageLaunchConfig(
        stage_name="aggregate",
        factory=fake_factory_path("make_scheduler"),
        next_stages="decode",
        wait_for=["preprocess", "thinker"],
        wait_for_fn=fake_factory_path("wait_sources_to_undeclared_stage"),
        merge_fn=fake_factory_path("merge_payloads"),
        recv_endpoint="inproc://aggregate",
        coordinator_endpoint="inproc://coordinator",
        abort_endpoint="inproc://abort",
        stage_endpoints={"decode": "inproc://decode"},
        comm_config={"slot_size_mb": 1},
    )
    stage_obj = _construct_stage(spec, logging.getLogger(__name__))

    with pytest.raises(ValueError, match="outside static wait_for"):
        stage_obj.input_handler.receive("req-1", "preprocess", make_stage_payload())


def test_stage_process_accepts_iterable_dynamic_wait_sources() -> None:
    spec = StageLaunchConfig(
        stage_name="aggregate",
        factory=fake_factory_path("make_scheduler"),
        next_stages="decode",
        wait_for=["preprocess", "thinker"],
        wait_for_fn=fake_factory_path("tuple_wait_sources"),
        merge_fn=fake_factory_path("merge_payloads"),
        recv_endpoint="inproc://aggregate",
        coordinator_endpoint="inproc://coordinator",
        abort_endpoint="inproc://abort",
        stage_endpoints={"decode": "inproc://decode"},
        comm_config={"slot_size_mb": 1},
    )
    stage_obj = _construct_stage(spec, logging.getLogger(__name__))

    assert (
        stage_obj.input_handler.receive("req-1", "preprocess", make_stage_payload())
        is None
    )
    merged = stage_obj.input_handler.receive("req-1", "thinker", make_stage_payload())

    assert merged is not None
    assert merged.data["merged_sources"] == ["preprocess", "thinker"]


def test_stage_run_raises_when_scheduler_thread_crashes() -> None:
    async def _run() -> None:
        scheduler = FakeScheduler(fail_start=RuntimeError("boom"))
        stage_obj = make_stage(
            scheduler=scheduler,
            control_plane=_CloseAwareControlPlane(),
        )

        with pytest.raises(RuntimeError, match="Scheduler thread"):
            await asyncio.wait_for(stage_obj.run(), timeout=2.0)

        assert scheduler.stopped is True

    asyncio.run(_run())


def test_relay_payload_and_cross_gpu_stream_contracts() -> None:
    """Preserves tensor payload round-trips and stream control-before-wait ordering."""

    async def _run() -> None:
        relay = FakeRelay()
        payload = make_tensor_payload()
        data_ref, op = await stage_io.write_payload(
            relay,
            payload.request_id,
            payload,
            transport=TransportKind.SHM,
        )
        await op.wait_for_completion()
        restored = await stage_io.read_payload(relay, payload.request_id, data_ref)
        assert tensor_equal(restored.data, payload.data)

        log = EventLog()
        stream_relay = FakeRelay(log=log)
        control_plane = RecordingStageControlPlane()
        control_plane.log = log
        stream_ref, stream_ops = await stage_io.write_stream_chunk(
            stream_relay,
            request_id="req-1",
            data=torch.tensor([1, 2, 3]),
            target_stage="talker",
            from_stage="thinker",
            chunk_id=0,
            metadata={"token_id": 1, "hidden": torch.tensor([4])},
            transport=TransportKind.SHM,
        )
        await control_plane.send_to_stage(
            "talker",
            "inproc://talker",
            DataReadyMessage(
                request_id="req-1",
                from_stage="thinker",
                to_stage="talker",
                data_ref=stream_ref.to_dict(),
                chunk_id=0,
            ),
        )
        for op in stream_ops:
            op.mark_receiver_done()
            await op.wait_for_completion()

        names = collect_event_names(log)
        assert names.index("stage_cp_send_to_stage") < names.index("op_wait")
        msg = control_plane.sent_to_stage[0][2]
        stream_ref = DataRef.from_dict(msg.data_ref)
        assert stream_ref.metadata["token_id"] == 1
        assert [ref.path for ref in stream_ref.metadata_tensors] == ["hidden"]

    asyncio.run(_run())


@pytest.mark.parametrize("request_field", ["inputs", "params", "metadata"])
def test_payload_request_validation_rejects_tensor_fields(request_field: str) -> None:
    payload = make_stage_payload(data={"value": torch.ones(1)})
    setattr(payload.request, request_field, {"tensor": torch.ones(1)})

    with pytest.raises(ValueError, match="request control metadata"):
        stage_io.validate_payload_request(payload)


_EXTENDED_TRANSFER_DTYPES = [
    getattr(torch, name)
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "uint16",
        "uint32",
        "uint64",
    )
    if isinstance(getattr(torch, name, None), torch.dtype)
]


@pytest.mark.parametrize("dtype", _EXTENDED_TRANSFER_DTYPES, ids=str)
def test_raw_tensor_codec_round_trips_extended_dtypes(dtype) -> None:
    async def _run() -> None:
        relay = FakeRelay()
        source = torch.empty((), dtype=dtype)
        source.reshape(-1).view(torch.uint8).reshape(-1).copy_(
            torch.arange(source.numel() * source.element_size(), dtype=torch.uint8)
        )

        data_ref, op = await stage_io.write_tensor(
            relay,
            "extended-dtype",
            source,
            transport=TransportKind.SHM,
        )
        restored = await stage_io.read_tensor(relay, data_ref)
        op.mark_receiver_done()
        await op.wait_for_completion()

        assert restored.dtype is dtype
        assert restored.shape == source.shape
        assert torch.equal(
            restored.reshape(-1).view(torch.uint8),
            source.reshape(-1).view(torch.uint8),
        )

    asyncio.run(_run())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "dtype",
    [dtype for dtype in _EXTENDED_TRANSFER_DTYPES if "float8" in str(dtype)],
    ids=str,
)
def test_cuda_raw_tensor_codec_round_trips_fp8(dtype) -> None:
    async def _run() -> None:
        relay = FakeRelay(device="cuda:0")
        source = torch.empty(8, dtype=dtype, device="cuda:0")
        source.view(torch.uint8).copy_(
            torch.arange(8, dtype=torch.uint8, device="cuda:0")
        )

        data_ref, op = await stage_io.write_tensor(
            relay,
            "cuda-fp8",
            source,
            transport=TransportKind.CUDA_IPC,
        )
        restored = await stage_io.read_tensor(relay, data_ref)
        op.mark_receiver_done()
        await op.wait_for_completion()

        assert restored.dtype is dtype
        assert torch.equal(restored.view(torch.uint8), source.view(torch.uint8))

    asyncio.run(_run())


@pytest.mark.parametrize("tensor_kind", ["quantized", "sparse"])
def test_raw_tensor_codec_rejects_unreconstructable_layouts(tensor_kind) -> None:
    async def _run() -> None:
        relay = FakeRelay()
        tensor = (
            torch.quantize_per_tensor(
                torch.tensor([1.0, 2.0]),
                scale=0.1,
                zero_point=10,
                dtype=torch.qint8,
            )
            if tensor_kind == "quantized"
            else torch.eye(2).to_sparse()
        )

        with pytest.raises(ValueError, match="tensor byte transport"):
            await stage_io.write_tensor(
                relay,
                "unsupported-tensor",
                tensor,
                transport=TransportKind.SHM,
            )
        assert relay.storage == {}

    asyncio.run(_run())


def test_packed_tensor_codec_preserves_mixed_dtype_alignment() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        byte_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
        byte_tensor = torch.empty(1, dtype=byte_dtype)
        byte_tensor.view(torch.uint8).fill_(7)
        wide_tensor = torch.empty(3, dtype=torch.uint64)
        wide_tensor.view(torch.uint8).copy_(torch.arange(24, dtype=torch.uint8))
        payload = make_stage_payload(data={"byte": byte_tensor, "wide": wide_tensor})

        data_ref, op = await stage_io.write_payload(
            relay,
            payload.request_id,
            payload,
            transport=TransportKind.SHM,
        )
        restored = await stage_io.read_payload(relay, payload.request_id, data_ref)
        op.mark_receiver_done()
        await op.wait_for_completion()

        entries = {entry.path: entry for entry in data_ref.tensors}
        assert entries["wide"].offset % wide_tensor.element_size() == 0
        assert torch.equal(
            restored.data["byte"].view(torch.uint8),
            byte_tensor.view(torch.uint8),
        )
        assert torch.equal(
            restored.data["wide"].view(torch.uint8),
            wide_tensor.view(torch.uint8),
        )

    asyncio.run(_run())


@pytest.mark.parametrize("dtype_name", sorted(stage_io._UNSUPPORTED_QUANTIZED_DTYPES))
def test_tensor_dtype_codec_rejects_quantized_metadata(dtype_name) -> None:
    with pytest.raises(ValueError, match="unsupported tensor dtype metadata"):
        stage_io._torch_dtype(dtype_name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_payload_round_trip_preserves_cpu_tensor_devices() -> None:
    async def _run() -> None:
        relay = FakeRelay(device="cuda:0")
        payload = make_stage_payload(
            request_id="req-mixed-devices",
            data={
                "embeds": torch.arange(4, device="cuda:0"),
                "grid": torch.ones(1, dtype=torch.long),
            },
        )

        data_ref, _ = await stage_io.write_payload(
            relay,
            payload.request_id,
            payload,
            transport=TransportKind.CUDA_IPC,
        )
        restored = await stage_io.read_payload(relay, payload.request_id, data_ref)

        assert restored.data["embeds"].device.type == "cuda"
        assert restored.data["grid"].device.type == "cpu"
        assert torch.equal(restored.data["grid"], torch.ones(1, dtype=torch.long))

    asyncio.run(_run())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_stream_round_trip_preserves_cpu_metadata_devices() -> None:
    async def _run() -> None:
        relay = FakeRelay(device="cuda:0")
        data_ref, ops = await stage_io.write_stream_chunk(
            relay,
            request_id="req-mixed-metadata",
            data=torch.arange(4, device="cuda:0"),
            target_stage="receiver",
            from_stage="sender",
            chunk_id=0,
            metadata={"stats": torch.ones(2)},
            transport=TransportKind.CUDA_IPC,
        )

        data, metadata = await stage_io.read_stream_chunk(relay, data_ref)
        for op in ops:
            op.mark_receiver_done()
            await op.wait_for_completion()

        assert data.device.type == "cuda"
        assert metadata is not None
        assert metadata["stats"].device.type == "cpu"
        assert torch.equal(metadata["stats"], torch.ones(2))
        assert data_ref.metadata_tensors[0].ref.device == "cpu"

    asyncio.run(_run())


def test_stage_relay_read_failure_completes_with_error() -> None:
    """Preserves failure reporting when a stage cannot read its relay payload."""

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(
            relay=relay,
            control_plane=control_plane,
            endpoints={"upstream": "inproc://upstream"},
        )
        payload = make_stage_payload(request_id="req-1")
        data_ref, _ = await stage_io.write_payload(
            relay,
            "req-1",
            payload,
            transport=TransportKind.SHM,
        )
        relay.fail_get = RuntimeError("read failed")

        await stage_obj._on_data_ready(
            DataReadyMessage("req-1", "upstream", "stage", data_ref.to_dict())
        )

        assert control_plane.completions[0].success is False
        assert "relay read failed" in control_plane.completions[0].error
        assert relay.cleaned[-1] == "req-1"

    asyncio.run(_run())


def test_stage_uses_dynamic_route_and_stream_done_targets() -> None:
    async def _run() -> None:
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(
            control_plane=control_plane,
            endpoints={"decode": "inproc://decode", "talker": "inproc://talker"},
            get_next=lambda request_id, output: output.request.metadata["next"],
            stream_targets=["talker", "decode"],
            get_stream_done_targets=lambda request_id, output: output.request.metadata[
                "stream_targets"
            ],
        )
        payload = make_stage_payload(request_id="req-1")
        payload.request.metadata["next"] = "decode"
        payload.request.metadata["stream_targets"] = ["decode"]
        stage_obj._active_requests.add("req-1")

        await stage_obj._route_result("req-1", payload)

        stream_done_target, _, stream_done_msg = control_plane.sent_to_stage[0]
        routed_target, _, routed_msg = control_plane.sent_to_stage[1]
        assert stream_done_target == "decode"
        assert isinstance(stream_done_msg, DataReadyMessage)
        assert stream_done_msg.is_done
        assert routed_target == "decode"
        assert isinstance(routed_msg, DataReadyMessage)
        assert not routed_msg.is_done

    asyncio.run(_run())


def test_stage_sends_same_process_payload_as_local_object(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.stage.runtime._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(name="decode", scheduler=receiver_scheduler)
        sender = make_stage(
            name="thinker",
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        tensor = torch.arange(4)
        payload = make_stage_payload(request_id="req-local", data={"tensor": tensor})

        await sender._send_to_stage(
            "req-local",
            "decode",
            payload,
            allow_local_object=True,
        )

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.type == "new_request"
        assert queued.data is payload
        assert queued.data.data["tensor"] is tensor

    asyncio.run(_run())

    hop_events = [event for event in events if event["event_name"] == "stage_hop_sent"]
    assert hop_events == [
        {
            "request_id": "req-local",
            "stage": "thinker",
            "event_name": "stage_hop_sent",
            "metadata": {"to_stage": "decode", "transport": "local_object"},
        }
    ]


def test_stage_applies_projector_before_local_object_send() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(name="decode", scheduler=receiver_scheduler)
        sender = make_stage(
            name="thinker",
            endpoints={"decode": "inproc://decode"},
            project_payload={"decode": make_noop_projector("decode-only")},
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        await sender._send_to_stage(
            "req-local",
            "decode",
            make_stage_payload(request_id="req-local", data={"answer": 7}),
            allow_local_object=True,
        )

        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.data.data == {
            "marker": "decode-only",
            "data": {"answer": 7},
        }

    asyncio.run(_run())


def test_stage_local_object_preserves_fan_in_semantics() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(
            name="aggregate",
            scheduler=receiver_scheduler,
            input_handler=AggregatedInput(
                {"preprocess", "thinker"},
                lambda payloads: make_stage_payload(
                    request_id="req-local",
                    data={
                        "sources": sorted(payloads),
                        "values": {
                            name: payload.data for name, payload in payloads.items()
                        },
                    },
                ),
            ),
        )
        preprocess = make_stage(
            name="preprocess",
            endpoints={"aggregate": "inproc://aggregate"},
            same_process_targets={"aggregate"},
            local_dispatcher=dispatcher,
        )
        thinker = make_stage(
            name="thinker",
            endpoints={"aggregate": "inproc://aggregate"},
            same_process_targets={"aggregate"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register(receiver)

        await preprocess._send_to_stage(
            "req-local",
            "aggregate",
            make_stage_payload(request_id="req-local", data={"p": 1}),
            allow_local_object=True,
        )
        assert receiver_scheduler.inbox.empty()

        await thinker._send_to_stage(
            "req-local",
            "aggregate",
            make_stage_payload(request_id="req-local", data={"t": 2}),
            allow_local_object=True,
        )

        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.type == "new_request"
        assert queued.data.data["sources"] == ["preprocess", "thinker"]
        assert queued.data.data["values"] == {
            "preprocess": {"p": 1},
            "thinker": {"t": 2},
        }

    asyncio.run(_run())


def test_stage_fan_out_payloads_materialize_when_local_object_is_unsafe() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            same_process_targets={"decode", "archive"},
        )

        await sender._route_result(
            "req-fanout",
            make_stage_payload(request_id="req-fanout", data={"answer": 7}),
        )

        assert [target for target, _, _ in control_plane.sent_to_stage] == [
            "decode",
            "archive",
        ]
        assert control_plane.sent_to_stage[0][2].chunk_id is None
        assert control_plane.sent_to_stage[1][2].chunk_id is None

    asyncio.run(_run())


def test_stage_projected_fan_out_payloads_use_local_object_when_isolated() -> None:
    def _isolated_projector(marker):
        def _project(payload):
            return make_stage_payload(
                request_id=payload.request_id,
                inputs=payload.request.inputs,
                params=payload.request.params,
                data={"marker": marker, "data": dict(payload.data)},
            )

        return _project

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        decode_scheduler = FakeScheduler()
        archive_scheduler = FakeScheduler()
        decode = make_stage(name="decode", scheduler=decode_scheduler)
        archive = make_stage(name="archive", scheduler=archive_scheduler)
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            project_payload={
                "decode": _isolated_projector("decode-only"),
                "archive": _isolated_projector("archive-only"),
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, decode, archive])

        await sender._route_result(
            "req-fanout",
            make_stage_payload(request_id="req-fanout", data={"answer": 7}),
        )

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        decode_msg = decode_scheduler.inbox.get_nowait()
        archive_msg = archive_scheduler.inbox.get_nowait()
        assert decode_msg.data.data == {
            "marker": "decode-only",
            "data": {"answer": 7},
        }
        assert archive_msg.data.data == {
            "marker": "archive-only",
            "data": {"answer": 7},
        }

    asyncio.run(_run())


def test_stage_projected_fan_out_requires_isolated_data_container() -> None:
    def _shared_data_projector(payload):
        return make_stage_payload(
            request_id=payload.request_id,
            inputs=payload.request.inputs,
            params=payload.request.params,
            data=payload.data,
        )

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            project_payload={
                "decode": _shared_data_projector,
                "archive": _shared_data_projector,
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=LocalStageDispatcher(),
        )

        await sender._route_result(
            "req-fanout",
            make_stage_payload(request_id="req-fanout", data={"answer": 7}),
        )

        assert [target for target, _, _ in control_plane.sent_to_stage] == [
            "decode",
            "archive",
        ]
        assert relay.storage

    asyncio.run(_run())


def test_stage_projected_fan_out_rejects_nested_mutable_aliases() -> None:
    def _shallow_copy_projector(payload):
        return make_stage_payload(
            request_id=payload.request_id,
            inputs=payload.request.inputs,
            params=payload.request.params,
            data={"projected": dict(payload.data)},
        )

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        decode = make_stage(name="decode", scheduler=FakeScheduler())
        archive = make_stage(name="archive", scheduler=FakeScheduler())
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            project_payload={
                "decode": _shallow_copy_projector,
                "archive": _shallow_copy_projector,
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, decode, archive])

        await sender._route_result(
            "req-fanout",
            make_stage_payload(
                request_id="req-fanout",
                data={"nested": {"tokens": [1, 2, 3]}, "answer": 7},
            ),
        )

        assert [target for target, _, _ in control_plane.sent_to_stage] == [
            "decode",
            "archive",
        ]
        assert relay.storage

    asyncio.run(_run())


def test_stage_projected_fan_out_rejects_wrapped_original_data() -> None:
    def _wrapped_data_projector(payload):
        return make_stage_payload(
            request_id=payload.request_id,
            inputs=payload.request.inputs,
            params=payload.request.params,
            data={"projected": payload.data},
        )

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            relay=relay,
            control_plane=control_plane,
            project_payload={
                "decode": _wrapped_data_projector,
                "archive": _wrapped_data_projector,
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=LocalStageDispatcher(),
        )

        await sender._route_result(
            "req-fanout",
            make_stage_payload(request_id="req-fanout", data={"answer": 7}),
        )

        assert [target for target, _, _ in control_plane.sent_to_stage] == [
            "decode",
            "archive",
        ]
        assert relay.storage

    asyncio.run(_run())


def test_stage_projected_fan_out_allows_tensor_leaf_sharing() -> None:
    def _tensor_leaf_projector(payload):
        return make_stage_payload(
            request_id=payload.request_id,
            inputs=payload.request.inputs,
            params=payload.request.params,
            data={"tensor": payload.data["tensor"], "target_only": []},
        )

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        decode_scheduler = FakeScheduler()
        decode = make_stage(name="decode", scheduler=decode_scheduler)
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            project_payload={"decode": _tensor_leaf_projector},
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, decode])
        tensor = torch.arange(4)

        await sender._route_result(
            "req-tensor-leaf",
            make_stage_payload(
                request_id="req-tensor-leaf",
                data={"tensor": tensor, "scratch": []},
            ),
        )

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        queued = decode_scheduler.inbox.get_nowait()
        assert queued.data.data["tensor"] is tensor

    asyncio.run(_run())


def test_stage_projected_fan_out_requires_stage_payload_projection() -> None:
    def _invalid_projector(payload):
        del payload
        return {"not": "a-stage-payload"}

    async def _run() -> None:
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: ["decode", "archive"],
            endpoints={
                "decode": "inproc://decode",
                "archive": "inproc://archive",
            },
            project_payload={
                "decode": _invalid_projector,
                "archive": _invalid_projector,
            },
            same_process_targets={"decode", "archive"},
            local_dispatcher=LocalStageDispatcher(),
        )

        with pytest.raises(
            TypeError,
            match="projectors to return StagePayload",
        ):
            await sender._route_result(
                "req-fanout",
                make_stage_payload(request_id="req-fanout", data={"answer": 7}),
            )

    asyncio.run(_run())


def test_stage_sends_same_process_stream_chunk_as_local_object(monkeypatch) -> None:
    events: list[dict] = []
    monkeypatch.setattr(
        "sglang_omni.pipeline.stage.runtime._emit_event",
        lambda **kwargs: events.append(kwargs),
    )

    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(
            name="talker",
            scheduler=receiver_scheduler,
            can_accept_stream_before_payload=True,
        )
        receiver._stream_queue = StreamQueue()
        sender = make_stage(
            name="thinker",
            endpoints={"talker": "inproc://talker"},
            relay=relay,
            control_plane=control_plane,
            same_process_targets={"talker"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        chunk = torch.arange(4)
        metadata = {"modality": "audio"}

        await sender._send_stream_to_target(
            "req-stream-local",
            chunk,
            "talker",
            metadata,
        )

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.type == "stream_chunk"
        assert queued.data.chunk_id == 0
        assert queued.data.data is chunk
        assert queued.data.metadata is metadata

    asyncio.run(_run())

    receive_events = [
        event
        for event in events
        if event["event_name"] == "stage_stream_chunk_received"
    ]
    assert receive_events == [
        {
            "request_id": "req-stream-local",
            "stage": "talker",
            "event_name": "stage_stream_chunk_received",
            "metadata": {"from_stage": "thinker", "chunk_id": 0},
        }
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stage_sends_same_gpu_stream_chunk_as_direct_cuda_ipc(monkeypatch) -> None:
    monkeypatch.setattr(
        stage_io,
        "try_serialize_direct_cuda_ipc_stream_chunk",
        lambda data, metadata: {
            "_type": "TorchCudaIpcStreamChunk",
            "version": 1,
            "tensor_bytes": b"handle",
            "metadata": metadata,
        },
    )

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = Stage(
            name="talker_ar",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=0,
            endpoints={"code2wav": "inproc://code2wav"},
            control_plane=control_plane,
            relay=relay,
            scheduler=FakeScheduler(),
            gpu_stage_names={"code2wav"},
            stage_gpu_ids={"code2wav": (0,)},
        )

        data = torch.arange(4, device="cuda:0")
        await sender._send_stream_to_target(
            "req-same-gpu",
            data,
            "code2wav",
            {"modality": "audio_codes"},
        )

        assert relay.storage == {}
        target, endpoint, msg = control_plane.sent_to_stage[0]
        assert target == "code2wav"
        assert endpoint == "inproc://code2wav"
        assert msg.data_ref["_type"] == "TorchCudaIpcStreamChunk"
        assert msg.chunk_id == 0

    asyncio.run(_run())


def test_stage_falls_back_for_ineligible_same_gpu_stream_chunk(monkeypatch) -> None:
    monkeypatch.setattr(
        stage_io,
        "try_serialize_direct_cuda_ipc_stream_chunk",
        lambda data, metadata: None,
    )

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = Stage(
            name="talker_ar",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=0,
            endpoints={"code2wav": "inproc://code2wav"},
            control_plane=control_plane,
            relay=relay,
            scheduler=FakeScheduler(),
            gpu_stage_names={"code2wav"},
            stage_gpu_ids={"code2wav": (0,)},
        )

        await sender._send_stream_to_target(
            "req-fallback",
            torch.arange(4),
            "code2wav",
            {"stats": torch.ones(1)},
        )

        assert relay.storage
        target, endpoint, msg = control_plane.sent_to_stage[0]
        assert target == "code2wav"
        assert endpoint == "inproc://code2wav"
        data_ref = DataRef.from_dict(msg.data_ref)
        assert data_ref.transport is TransportKind.SHM
        assert data_ref.object_id in sender._comm._pending

        completion_task = sender._comm._pending[data_ref.object_id].task
        assert completion_task is not None
        sender._comm.ack_transfer(
            DataAckMessage(
                request_id="req-fallback",
                from_stage="code2wav",
                to_stage="talker_ar",
                object_id=data_ref.object_id,
            )
        )
        await completion_task
        sender._comm.close()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "metadata",
    [
        {"stats": torch.ones(1)},
        {"transcript": "x" * (128 * 1024)},
    ],
)
def test_direct_stream_serializes_inline_metadata(
    monkeypatch,
    metadata,
) -> None:
    data = object()
    monkeypatch.setattr(
        stage_io,
        "_contains_cuda_tensor",
        lambda value, seen=None: value is data,
    )

    data_ref = stage_io.try_serialize_direct_cuda_ipc_stream_chunk(data, metadata)

    assert data_ref is not None
    message = DataReadyMessage(
        request_id="req-inline-metadata",
        from_stage="sender",
        to_stage="receiver",
        data_ref=data_ref,
        chunk_id=0,
    )
    decoded = deserialize_message(serialize_message(message))
    _, restored = stage_io.deserialize_direct_cuda_ipc_stream_chunk(decoded.data_ref)
    assert restored is not None
    if "stats" in metadata:
        assert torch.equal(restored["stats"], metadata["stats"])
        assert restored["stats"].device.type == "cpu"
    else:
        assert restored == metadata


def test_direct_inline_cpu_tensor_compacts_backing_storage(monkeypatch) -> None:
    data = object()
    base = torch.arange(1_000_000, dtype=torch.float32)
    view = base[:1]
    monkeypatch.setattr(
        stage_io,
        "_contains_cuda_tensor",
        lambda value, seen=None: value is data,
    )

    data_ref = stage_io.try_serialize_direct_cuda_ipc_stream_chunk(
        data,
        {"view": view},
    )

    assert data_ref is not None
    tensor_pickle = data_ref["metadata"]["view"]["_ipc_tensor"]
    assert len(tensor_pickle) < 1024
    restored = stage_io.deserialize_direct_ipc_metadata(data_ref["metadata"])
    assert torch.equal(restored["view"], view)


def test_direct_payload_header_compacts_shared_cpu_storage_views() -> None:
    base = torch.arange(1_000_000, dtype=torch.float32)
    view = base[:1]

    header, tensors = stage_io.extract_cuda_tensors({"first": view, "again": view})

    assert tensors == {}
    assert header["first"] is header["again"]
    assert (
        header["first"].untyped_storage().nbytes() == view.numel() * view.element_size()
    )
    assert torch.equal(header["first"], view)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("metadata_kind", ["cpu_tensor", "large_inline"])
def test_same_gpu_cuda_stream_keeps_direct_for_inline_metadata(
    metadata_kind,
) -> None:
    async def _run() -> None:
        relay = FakeRelay(device="cuda:0")
        control_plane = RecordingStageControlPlane()
        sender = Stage(
            name="talker_ar",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=0,
            endpoints={"code2wav": "inproc://code2wav"},
            control_plane=control_plane,
            relay=relay,
            scheduler=FakeScheduler(),
            gpu_stage_names={"code2wav"},
            stage_gpu_ids={"code2wav": (0,)},
        )
        metadata = (
            {"stats": torch.ones(1)}
            if metadata_kind == "cpu_tensor"
            else {"transcript": "x" * (128 * 1024)}
        )
        source = torch.arange(4, device="cuda:0")

        await sender._send_stream_to_target(
            f"req-{metadata_kind}",
            source,
            "code2wav",
            metadata,
        )

        target, endpoint, msg = control_plane.sent_to_stage[0]
        assert target == "code2wav"
        assert endpoint == "inproc://code2wav"
        assert msg.data_ref["_type"] == "TorchCudaIpcStreamChunk"
        assert relay.storage == {}
        _run_direct_ipc_receiver(
            _receive_direct_ipc_stream_chunk,
            msg.data_ref,
            metadata_kind,
        )
        sender._comm.close()

    asyncio.run(_run())


def test_stage_sends_same_gpu_cuda_payload_as_direct_cuda_ipc(monkeypatch) -> None:
    monkeypatch.setattr(
        stage_io,
        "try_serialize_direct_cuda_ipc_payload",
        lambda payload: {
            "_type": "TorchCudaIpcPayload",
            "version": 1,
            "header": b"payload",
            "tensors": [],
        },
    )

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = Stage(
            name="encoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=0,
            endpoints={"mm_aggregate": "inproc://mm"},
            control_plane=control_plane,
            relay=relay,
            scheduler=FakeScheduler(),
            gpu_stage_names={"mm_aggregate"},
            stage_gpu_ids={"mm_aggregate": (0,)},
        )

        payload = make_stage_payload(request_id="req-same-gpu", data={"x": "cuda"})
        await sender._send_to_stage("req-same-gpu", "mm_aggregate", payload)

        assert relay.storage == {}
        target, endpoint, msg = control_plane.sent_to_stage[0]
        assert target == "mm_aggregate"
        assert endpoint == "inproc://mm"
        assert msg.data_ref["_type"] == "TorchCudaIpcPayload"
        assert msg.chunk_id is None

    asyncio.run(_run())


def test_stage_can_disable_same_gpu_direct_cuda_payload(monkeypatch) -> None:
    def _unexpected_direct_payload(payload):
        raise AssertionError("direct payload serializer should not be called")

    monkeypatch.setattr(
        stage_io,
        "try_serialize_direct_cuda_ipc_payload",
        _unexpected_direct_payload,
    )

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = Stage(
            name="mm_aggregate",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=0,
            endpoints={"thinker": "inproc://thinker"},
            control_plane=control_plane,
            relay=relay,
            scheduler=FakeScheduler(),
            gpu_stage_names={"thinker"},
            stage_gpu_ids={"thinker": (0,)},
            disable_direct_cuda_ipc_payload=True,
        )

        payload = make_tensor_payload(request_id="req-direct-disabled")
        await sender._send_to_stage("req-direct-disabled", "thinker", payload)

        target, endpoint, msg = control_plane.sent_to_stage[0]
        assert target == "thinker"
        assert endpoint == "inproc://thinker"
        assert msg.data_ref["_type"] == "DataRef"
        assert relay.storage

    asyncio.run(_run())


def test_stage_uses_relay_when_direct_cuda_payload_is_reexported(monkeypatch) -> None:
    def _raise_reexport(payload):
        raise RuntimeError(
            "Attempted to send CUDA tensor received from another process"
        )

    monkeypatch.setattr(
        stage_io,
        "try_serialize_direct_cuda_ipc_payload",
        _raise_reexport,
    )

    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = Stage(
            name="mm_aggregate",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=0,
            endpoints={"talker_ar": "inproc://talker"},
            control_plane=control_plane,
            relay=relay,
            scheduler=FakeScheduler(),
            gpu_stage_names={"talker_ar"},
            stage_gpu_ids={"talker_ar": (0,)},
        )

        payload = make_tensor_payload(request_id="req-reexport")
        await sender._send_to_stage("req-reexport", "talker_ar", payload)

        target, endpoint, msg = control_plane.sent_to_stage[0]
        assert target == "talker_ar"
        assert endpoint == "inproc://talker"
        assert msg.data_ref["_type"] == "DataRef"
        assert relay.storage

    asyncio.run(_run())


def test_large_direct_payload_header_is_not_rerouted(monkeypatch) -> None:
    payload = make_stage_payload(
        data={"blob": "x" * (128 * 1024)},
        inputs="request",
    )
    monkeypatch.setattr(
        stage_io,
        "extract_cuda_tensors",
        lambda data: (data, {"gpu": object()}),
    )
    monkeypatch.setattr(stage_io, "_ipc_pickle", lambda value: b"cuda-handle")

    data_ref = stage_io.try_serialize_direct_cuda_ipc_payload(payload)

    assert data_ref is not None
    assert data_ref["_type"] == "TorchCudaIpcPayload"
    message = DataReadyMessage(
        request_id="req-large-header",
        from_stage="sender",
        to_stage="receiver",
        data_ref=data_ref,
    )
    decoded = deserialize_message(serialize_message(message))
    assert len(decoded.data_ref["header"]) > 128 * 1024
    assert pickle.loads(decoded.data_ref["header"]).request.inputs == "request"


@pytest.mark.parametrize("local_object", [False, True])
def test_stage_rejects_tensor_in_request_control_metadata(local_object: bool) -> None:
    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        dispatcher = LocalStageDispatcher() if local_object else None
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(name="mm_aggregate", scheduler=receiver_scheduler)
        sender = Stage(
            name="encoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=0,
            endpoints={"mm_aggregate": "inproc://mm"},
            control_plane=control_plane,
            relay=relay,
            scheduler=FakeScheduler(),
            gpu_stage_names={"mm_aggregate"},
            stage_gpu_ids={"mm_aggregate": (0,)},
            same_process_targets={"mm_aggregate"} if local_object else set(),
            local_dispatcher=dispatcher,
        )
        if dispatcher is not None:
            dispatcher.register_many([sender, receiver])
        payload = make_stage_payload(
            request_id="req-request-tensor",
            data={"encoder_out": "value"},
            inputs={"tensor": torch.ones(1)},
        )

        with pytest.raises(ValueError, match="request control metadata"):
            await sender._send_to_stage(
                payload.request_id,
                "mm_aggregate",
                payload,
                allow_local_object=local_object,
            )

        assert control_plane.sent_to_stage == []
        assert relay.storage == {}
        assert receiver_scheduler.inbox.empty()
        sender._comm.close()

    asyncio.run(_run())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_same_gpu_cuda_payload_keeps_direct_for_large_header() -> None:
    async def _run() -> None:
        relay = FakeRelay(device="cuda:0")
        control_plane = RecordingStageControlPlane()
        sender = Stage(
            name="encoder",
            role="single",
            get_next=lambda request_id, output: None,
            gpu_id=0,
            endpoints={"mm_aggregate": "inproc://mm"},
            control_plane=control_plane,
            relay=relay,
            scheduler=FakeScheduler(),
            gpu_stage_names={"mm_aggregate"},
            stage_gpu_ids={"mm_aggregate": (0,)},
        )
        payload = make_stage_payload(
            request_id="req-large-header",
            data={"encoder_out": torch.arange(4, device="cuda:0")},
            inputs="x" * (128 * 1024),
        )

        await sender._send_to_stage(payload.request_id, "mm_aggregate", payload)

        target, endpoint, msg = control_plane.sent_to_stage[0]
        assert target == "mm_aggregate"
        assert endpoint == "inproc://mm"
        assert msg.data_ref["_type"] == "TorchCudaIpcPayload"
        assert relay.storage == {}
        _run_direct_ipc_receiver(
            _receive_direct_ipc_payload,
            msg.data_ref,
            "large_header",
        )
        sender._comm.close()

    asyncio.run(_run())


def test_stage_receives_same_gpu_direct_cuda_ipc_payload(monkeypatch) -> None:
    payload = make_stage_payload(request_id="req-direct", data={"answer": 7})
    monkeypatch.setattr(
        stage_io,
        "deserialize_direct_cuda_ipc_payload",
        lambda data_ref: payload,
    )

    async def _run() -> None:
        control_plane = RecordingStageControlPlane()
        scheduler = FakeScheduler()
        receiver = make_stage(
            name="mm_aggregate",
            scheduler=scheduler,
            control_plane=control_plane,
        )

        await receiver._on_data_ready(
            DataReadyMessage(
                request_id="req-direct",
                from_stage="encoder",
                to_stage="mm_aggregate",
                data_ref={
                    "_type": "TorchCudaIpcPayload",
                    "version": 1,
                    "header": b"payload",
                    "tensors": [],
                },
            )
        )

        queued = scheduler.inbox.get_nowait()
        assert queued.type == "new_request"
        assert queued.data is payload
        assert control_plane.sent_to_stage == []

    asyncio.run(_run())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_direct_cuda_ipc_payload_preserves_inline_cpu_tensors() -> None:
    payload = make_stage_payload(
        data={
            "gpu": torch.arange(2, device="cuda:0"),
            "cpu": torch.ones(1),
        }
    )

    ref = stage_io.serialize_direct_cuda_ipc_payload(payload)
    header = pickle.loads(ref["header"])

    assert header.data["gpu"]["_tensor_placeholder"] == "gpu"
    assert not header.data["cpu"].is_cuda
    assert torch.equal(header.data["cpu"], torch.ones(1))
    assert [entry["path"] for entry in ref["tensors"]] == ["gpu"]
    _run_direct_ipc_receiver(
        _receive_direct_ipc_payload,
        ref,
        "cpu_tensor",
    )


def test_direct_cuda_ipc_payload_rejects_cpu_only_payloads() -> None:
    payload = make_stage_payload(data={"x": torch.ones(1)})

    with pytest.raises(ValueError, match="at least one CUDA tensor"):
        stage_io.serialize_direct_cuda_ipc_payload(payload)


def test_stage_sends_same_process_stream_done_and_final_payload_locally() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(
            name="decode",
            scheduler=receiver_scheduler,
            can_accept_stream_before_payload=True,
        )
        receiver._stream_queue = StreamQueue()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            stream_targets=["decode"],
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        payload = make_stage_payload(request_id="req-stream-local", data={"answer": 7})
        await sender._route_result("req-stream-local", payload)

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        stream_done = receiver_scheduler.inbox.get_nowait()
        full_payload = receiver_scheduler.inbox.get_nowait()
        assert stream_done.type == "stream_done"
        assert full_payload.type == "new_request"
        assert full_payload.data is payload

    asyncio.run(_run())


def test_stage_allows_local_payload_when_static_stream_target_is_inactive() -> None:
    async def _run() -> None:
        dispatcher = LocalStageDispatcher()
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        receiver_scheduler = FakeScheduler()
        receiver = make_stage(name="decode", scheduler=receiver_scheduler)
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            get_stream_done_targets=lambda request_id, output: None,
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            stream_targets=["decode"],
            same_process_targets={"decode"},
            local_dispatcher=dispatcher,
        )
        dispatcher.register_many([sender, receiver])

        payload = make_stage_payload(request_id="req-no-stream", data={"answer": 7})
        await sender._route_result("req-no-stream", payload)

        assert relay.storage == {}
        assert control_plane.sent_to_stage == []
        queued = receiver_scheduler.inbox.get_nowait()
        assert queued.type == "new_request"
        assert queued.data is payload

    asyncio.run(_run())


def test_stage_preserves_relay_order_when_target_also_receives_stream() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        sender = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode"},
            relay=relay,
            control_plane=control_plane,
            stream_targets=["decode"],
        )

        await sender._route_result(
            "req-streamed",
            make_stage_payload(request_id="req-streamed", data={"answer": 7}),
        )

        assert [msg.is_done for _, _, msg in control_plane.sent_to_stage] == [
            True,
            False,
        ]
        assert control_plane.sent_to_stage[1][2].chunk_id is None
        assert relay.storage

    asyncio.run(_run())


def test_stage_payload_send_requires_endpoint() -> None:
    async def _run() -> None:
        sender = make_stage(name="thinker", endpoints={})

        with pytest.raises(RuntimeError, match="no endpoint configured"):
            await sender._send_to_stage(
                "req-1",
                "decode",
                make_stage_payload(request_id="req-1"),
            )

    asyncio.run(_run())


def test_stage_local_object_requires_registered_target() -> None:
    async def _run() -> None:
        sender = make_stage(
            name="thinker",
            endpoints={"decode": "inproc://decode"},
            same_process_targets={"decode"},
            local_dispatcher=LocalStageDispatcher(),
        )

        with pytest.raises(RuntimeError, match="not registered"):
            await sender._send_to_stage(
                "req-local",
                "decode",
                make_stage_payload(request_id="req-local"),
                allow_local_object=True,
            )

    asyncio.run(_run())
