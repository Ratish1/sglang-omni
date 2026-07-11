#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-process H200 gate for the intra-node Stage transports.

The harness deliberately calls the production Stage send/receive methods,
StageControlPlane, CommEngine, stage_io codecs, and CudaIpcRelay.  The only
instrumented classes count relay calls and control ACKs while delegating the
actual work to those production implementations.

Examples:
    python benchmarks/communication/stage_transport.py \
        --case direct-payload --src-gpu 0 --dst-gpus 0
    python benchmarks/communication/stage_transport.py \
        --case pooled-stream --src-gpu 0 --dst-gpus 1 --window 8
    python benchmarks/communication/stage_transport.py \
        --case fanout --src-gpu 0 --dst-gpus 1,1
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sglang_omni.comm import stage_io
from sglang_omni.comm.data_ref import DataRef
from sglang_omni.pipeline.control_plane import (
    ControlPlaneContext,
    StageControlPlane,
    serialize_message,
)
from sglang_omni.pipeline.stage.runtime import Stage
from sglang_omni.pipeline.stage.stream_queue import StreamQueue
from sglang_omni.proto import (
    DataAckMessage,
    DataReadyMessage,
    OmniRequest,
    StagePayload,
)
from sglang_omni.relay.cuda_ipc import CudaIpcRelay

_PROCESS_TIMEOUT_S = 180.0
_POLL_INTERVAL_S = 0.001


@dataclass(frozen=True)
class CaseConfig:
    case: str
    src_gpu: int
    dst_gpus: tuple[int, ...]
    tensor_bytes: int
    metadata_bytes: int
    header_bytes: int
    cpu_view_backing_bytes: int
    warmups: int
    count: int
    window: int
    pool_size_mb: int
    timeout_s: float
    profile: bool

    @property
    def is_stream(self) -> bool:
        return "stream" in self.case

    @property
    def is_direct(self) -> bool:
        return self.case.startswith("direct")

    @property
    def is_abort(self) -> bool:
        return "abort" in self.case

    @property
    def is_dtype_suite(self) -> bool:
        return self.case == "pooled-dtypes"

    @property
    def dtype_names(self) -> tuple[str, ...]:
        return tuple(stage_io._BYTE_VIEW_DTYPES)

    @property
    def warmup_transfers_per_target(self) -> int:
        multiplier = len(self.dtype_names) if self.is_dtype_suite else 1
        return self.warmups * multiplier

    @property
    def transfers_per_target(self) -> int:
        multiplier = len(self.dtype_names) if self.is_dtype_suite else 1
        return (self.warmups + self.count) * multiplier


@dataclass(frozen=True)
class EndpointConfig:
    sender: str
    receivers: dict[str, str]
    coordinator: str
    abort: str


class CountingCudaIpcRelay(CudaIpcRelay):
    """Count calls without replacing any CUDA relay mechanics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.put_calls = 0
        self.get_calls = 0

    async def put_async(self, *args: Any, **kwargs: Any):
        self.put_calls += 1
        return await super().put_async(*args, **kwargs)

    async def get_async(self, *args: Any, **kwargs: Any):
        self.get_calls += 1
        return await super().get_async(*args, **kwargs)


class CountingStageControlPlane(StageControlPlane):
    """Count ACKs while retaining the production ZMQ/msgpack path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sent_acks = 0

    async def send_to_stage(self, next_stage: str, endpoint: str, msg: Any) -> None:
        if isinstance(msg, DataAckMessage):
            self.sent_acks += 1
        await super().send_to_stage(next_stage, endpoint, msg)


class HarnessScheduler:
    def __init__(self) -> None:
        self.inbox: queue.Queue = queue.Queue()
        self.outbox: queue.Queue = queue.Queue()
        self.aborted: list[str] = []

    def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)


def _nvtx_push(enabled: bool, label: str) -> None:
    if enabled:
        torch.cuda.nvtx.range_push(label)


def _nvtx_pop(enabled: bool) -> None:
    if enabled:
        torch.cuda.nvtx.range_pop()


def _sequence_checksum(numel: int, offset: int = 0) -> int:
    cycles, remainder = divmod(numel, 251)
    cycle_sum = 250 * 251 // 2
    tail = sum((index + offset) % 251 for index in range(remainder))
    if offset:
        cycle_sum = sum((index + offset) % 251 for index in range(251))
    return cycles * cycle_sum + tail


def _make_tensor(numel: int, device: torch.device, offset: int = 0) -> torch.Tensor:
    values = torch.arange(numel, dtype=torch.int32, device=device)
    if offset:
        values.add_(offset)
    return values.remainder_(251).to(torch.uint8)


def _make_dtype_tensor(
    num_bytes: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if num_bytes % dtype.itemsize:
        raise ValueError(
            f"{num_bytes} bytes cannot represent a whole number of {dtype} values"
        )
    return _make_tensor(num_bytes, device).view(dtype)


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _wire_name(msg: DataReadyMessage) -> str:
    ref = msg.data_ref
    if not isinstance(ref, dict):
        return type(ref).__name__
    direct_type = ref.get("_type")
    if direct_type != "DataRef":
        return str(direct_type)
    data_ref = DataRef.from_dict(ref)
    return f"DataRef:{data_ref.transport.value}:{data_ref.layout.value}"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[index])


def _latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": _percentile(latencies_ms, 0.50),
        "p95_ms": _percentile(latencies_ms, 0.95),
        "p99_ms": _percentile(latencies_ms, 0.99),
        "mean_ms": float(statistics.fmean(latencies_ms)),
    }


def _logical_gbps(logical_bytes: int, p50_ms: float) -> float:
    if p50_ms <= 0:
        return 0.0
    return logical_bytes / (p50_ms / 1000.0) / 1_000_000_000.0


def _stage(
    *,
    name: str,
    gpu: int,
    endpoints: dict[str, str],
    control_plane: StageControlPlane,
    relay: CudaIpcRelay,
    scheduler: HarnessScheduler,
    stage_gpu_ids: dict[str, tuple[int, ...]],
    stream_receiver: bool = False,
) -> Stage:
    stage = Stage(
        name=name,
        role="single",
        get_next=lambda request_id, output: None,
        gpu_id=gpu,
        endpoints=endpoints,
        control_plane=control_plane,
        relay=relay,
        comm_config={
            "ack_timeout_s": 60.0,
            "send_queue_size": 1024,
        },
        scheduler=scheduler,
        gpu_stage_names=set(stage_gpu_ids),
        stage_gpu_ids=stage_gpu_ids,
        can_accept_stream_before_payload=stream_receiver,
    )
    if stream_receiver:
        stage._stream_queue = StreamQueue(max_pending=4096)
    return stage


async def _receive_acks(stage: Stage, expected: int, timeout_s: float) -> int:
    for _ in range(expected):
        msg = await asyncio.wait_for(stage.control_plane.recv(), timeout=timeout_s)
        if not isinstance(msg, DataAckMessage):
            raise TypeError(f"sender expected DataAckMessage, got {type(msg).__name__}")
        await stage._handle_message(msg)
    return expected


async def _wait_pending_empty(stage: Stage, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while stage._comm._pending:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"pending transfers did not drain: {sorted(stage._comm._pending)}"
            )
        await asyncio.sleep(_POLL_INTERVAL_S)


async def _next_completion(completion_q: mp.Queue, timeout_s: float) -> tuple[str, int]:
    return await asyncio.to_thread(completion_q.get, True, timeout_s)


async def _sender_run(
    config: CaseConfig,
    endpoints: EndpointConfig,
    completion_q: mp.Queue,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{config.src_gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = _memory_snapshot(device)

    control_plane = CountingStageControlPlane(
        stage_name="sender",
        recv_endpoint=endpoints.sender,
        coordinator_endpoint=endpoints.coordinator,
        abort_endpoint=endpoints.abort,
    )
    await control_plane.start()
    relay = CountingCudaIpcRelay(
        engine_id="h200-sender",
        device=str(device),
        pool_size_mb=config.pool_size_mb,
        slot_size_kb=64,
    )
    stage_gpu_ids = {"sender": (config.src_gpu,)}
    for index, (target, _) in enumerate(endpoints.receivers.items()):
        stage_gpu_ids[target] = (config.dst_gpus[index],)
    stage = _stage(
        name="sender",
        gpu=config.src_gpu,
        endpoints=endpoints.receivers,
        control_plane=control_plane,
        relay=relay,
        scheduler=HarnessScheduler(),
        stage_gpu_ids=stage_gpu_ids,
    )

    primary = (
        None if config.is_dtype_suite else _make_tensor(config.tensor_bytes, device)
    )
    layer_hidden = (
        _make_tensor(config.metadata_bytes, device, offset=17)
        if config.is_stream
        else None
    )
    cpu_base = (
        torch.arange(config.cpu_view_backing_bytes, dtype=torch.uint8)
        if config.cpu_view_backing_bytes
        else None
    )
    cpu_view = cpu_base[:1] if cpu_base is not None else None
    total_messages = config.transfers_per_target * len(endpoints.receivers)
    ack_task = None
    if not config.is_direct:
        ack_task = asyncio.create_task(
            _receive_acks(stage, total_messages, config.timeout_s)
        )

    publication_ms: list[float] = []
    completed: set[tuple[str, int]] = set()
    payload = None
    primary_for_send = None
    metadata = None
    sends: list[asyncio.Task] = []
    acks_received = 0
    result: dict[str, Any] | None = None
    _nvtx_push(config.profile, f"comm_case:{config.case}:sender")
    try:
        for batch_start in range(0, config.transfers_per_target, config.window):
            batch_stop = min(
                batch_start + config.window,
                config.transfers_per_target,
            )
            sends = []
            send_started: dict[tuple[str, int], int] = {}
            for sequence in range(batch_start, batch_stop):
                primary_for_send = primary
                if config.is_dtype_suite:
                    dtype_name = config.dtype_names[sequence % len(config.dtype_names)]
                    primary_for_send = _make_dtype_tensor(
                        config.tensor_bytes,
                        stage_io._BYTE_VIEW_DTYPES[dtype_name],
                        device,
                    )
                for target in endpoints.receivers:
                    request_id = f"h200-{target}-{sequence}"
                    sent_ns = time.perf_counter_ns()
                    send_started[(target, sequence)] = sent_ns
                    if config.is_stream:
                        metadata = {
                            "token_id": sequence,
                            "sent_ns": sent_ns,
                            "layer_hidden": layer_hidden,
                        }
                        if config.header_bytes:
                            metadata["transcript"] = "x" * config.header_bytes
                        if cpu_view is not None:
                            metadata["cpu_view"] = cpu_view
                        sends.append(
                            asyncio.create_task(
                                stage._send_stream_to_target(
                                    request_id,
                                    primary_for_send,
                                    target,
                                    metadata,
                                )
                            )
                        )
                    else:
                        payload = StagePayload(
                            request_id=request_id,
                            request=OmniRequest(
                                inputs=(
                                    "x" * config.header_bytes
                                    if config.header_bytes
                                    else "h200-stage-transport"
                                ),
                                metadata={"sequence": sequence, "sent_ns": sent_ns},
                            ),
                            data={
                                "primary": primary_for_send,
                                **(
                                    {"cpu_view": cpu_view}
                                    if cpu_view is not None
                                    else {}
                                ),
                            },
                        )
                        sends.append(
                            asyncio.create_task(
                                stage._send_to_stage(request_id, target, payload)
                            )
                        )
            await asyncio.gather(*sends)
            published_ns = time.perf_counter_ns()
            for started_ns in send_started.values():
                publication_ms.append((published_ns - started_ns) / 1_000_000.0)

            expected_completions = len(send_started)
            for _ in range(expected_completions):
                target, sequence = await _next_completion(
                    completion_q, config.timeout_s
                )
                key = (target, sequence)
                if key not in send_started:
                    raise RuntimeError(f"unexpected receiver completion {key!r}")
                if key in completed:
                    raise RuntimeError(f"duplicate receiver completion {key!r}")
                completed.add(key)

        if ack_task is not None:
            acks_received = await asyncio.wait_for(ack_task, timeout=config.timeout_s)
        await _wait_pending_empty(stage, config.timeout_s)
        result = {
            "role": "sender",
            "completed": len(completed),
            "publication": _latency_summary(
                publication_ms[
                    config.warmup_transfers_per_target * len(endpoints.receivers) :
                ]
            ),
            "memory_before": before,
            "memory_peak": _memory_snapshot(device),
        }
    finally:
        _nvtx_pop(config.profile)
        if ack_task is not None and not ack_task.done():
            ack_task.cancel()
            with suppress(asyncio.CancelledError):
                await ack_task
        relay_put_calls = relay.put_calls
        pending = len(stage._comm._pending)
        stage._comm.close()
        await asyncio.sleep(0)
        control_plane.close()
        ControlPlaneContext.close()
        payload = None
        metadata = None
        sends.clear()
        primary = None
        primary_for_send = None
        layer_hidden = None
        cpu_view = None
        cpu_base = None
        stage = None
        relay = None
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        after = _memory_snapshot(device)

    if result is None:
        raise RuntimeError("sender completed without a result")
    result.update(
        relay_put_calls=relay_put_calls,
        ack_messages=acks_received,
        pending=pending,
        memory_after=after,
    )
    return result


def _sender_process(
    config: CaseConfig,
    endpoints: EndpointConfig,
    ready_q: mp.Queue,
    completion_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    try:
        ready_q.put(("sender", os.getpid()))
        result = asyncio.run(_sender_run(config, endpoints, completion_q))
        result_q.put(("sender", "ok", result))
    except BaseException:
        detail = traceback.format_exc()
        result_q.put(("sender", "error", detail))
        raise


async def _receiver_run(
    *,
    receiver_name: str,
    receiver_index: int,
    config: CaseConfig,
    endpoints: EndpointConfig,
    ready_q: mp.Queue,
    completion_q: mp.Queue,
) -> dict[str, Any]:
    gpu = config.dst_gpus[receiver_index]
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = _memory_snapshot(device)

    control_plane = CountingStageControlPlane(
        stage_name=receiver_name,
        recv_endpoint=endpoints.receivers[receiver_name],
        coordinator_endpoint=endpoints.coordinator,
        abort_endpoint=endpoints.abort,
    )
    await control_plane.start()
    relay = CountingCudaIpcRelay(
        engine_id=f"h200-{receiver_name}",
        device=str(device),
        pool_size_mb=config.pool_size_mb,
        slot_size_kb=64,
    )
    scheduler = HarnessScheduler()
    stage_gpu_ids = {
        "sender": (config.src_gpu,),
        receiver_name: (gpu,),
    }
    stage = _stage(
        name=receiver_name,
        gpu=gpu,
        endpoints={"sender": endpoints.sender},
        control_plane=control_plane,
        relay=relay,
        scheduler=scheduler,
        stage_gpu_ids=stage_gpu_ids,
        stream_receiver=config.is_stream,
    )
    ready_q.put((receiver_name, os.getpid()))

    expected_primary = _sequence_checksum(config.tensor_bytes)
    expected_layer = _sequence_checksum(config.metadata_bytes, offset=17)
    latencies_ms: list[float] = []
    wire_types: set[str] = set()
    received_dtypes: set[str] = set()
    pool_refcounter_offsets: set[int] = set()
    control_bytes_sample = 0
    incoming = None
    layer_hidden = None
    metadata = None
    msg = None
    payload = None
    primary = None
    stream_item = None
    cpu_view = None
    result: dict[str, Any] | None = None
    _nvtx_push(config.profile, f"comm_case:{config.case}:{receiver_name}")
    try:
        for sequence in range(config.transfers_per_target):
            payload = None
            stream_item = None
            metadata = None
            msg = await asyncio.wait_for(control_plane.recv(), timeout=config.timeout_s)
            if not isinstance(msg, DataReadyMessage):
                raise TypeError(
                    f"receiver expected DataReadyMessage, got {type(msg).__name__}"
                )
            wire_types.add(_wire_name(msg))
            if (
                isinstance(msg.data_ref, dict)
                and msg.data_ref.get("_type") == "DataRef"
            ):
                backend = DataRef.from_dict(msg.data_ref).buffer.info
                pool_storage = backend.get("cuda_ipc", {}).get("pool_storage", {})
                offset = pool_storage.get("ref_counter_offset")
                if offset is not None:
                    pool_refcounter_offsets.add(int(offset))
            if config.is_abort:
                stage._aborted.add(msg.request_id)
                if config.is_stream:
                    await stage._on_stream_chunk(msg)
                else:
                    await stage._on_data_ready(msg)
                if not scheduler.inbox.empty():
                    raise AssertionError("aborted direct value reached the scheduler")
                if sequence == config.warmups:
                    control_bytes_sample = len(serialize_message(msg))
                msg = None
                completion_q.put((receiver_name, sequence))
                continue
            if config.is_stream:
                await stage._on_stream_chunk(msg)
                incoming = scheduler.inbox.get(timeout=config.timeout_s)
                stream_item = incoming.data
                primary = stream_item.data
                metadata = stream_item.metadata or {}
                received_sequence = int(metadata["token_id"])
                sent_ns = int(metadata["sent_ns"])
                layer_hidden = metadata["layer_hidden"]
                cpu_view = metadata.get("cpu_view")
                if (
                    config.header_bytes
                    and len(metadata["transcript"]) != config.header_bytes
                ):
                    raise AssertionError("stream transcript length changed in transit")
            else:
                await stage._on_data_ready(msg)
                incoming = scheduler.inbox.get(timeout=config.timeout_s)
                payload = incoming.data
                primary = payload.data["primary"]
                received_sequence = int(payload.request.metadata["sequence"])
                sent_ns = int(payload.request.metadata["sent_ns"])
                layer_hidden = None
                cpu_view = payload.data.get("cpu_view")
                expected_input_bytes = (
                    config.header_bytes
                    if config.header_bytes
                    else len("h200-stage-transport")
                )
                if len(payload.request.inputs) != expected_input_bytes:
                    raise AssertionError("payload header length changed in transit")

            if received_sequence != sequence:
                raise AssertionError(
                    f"{receiver_name} got sequence {received_sequence}, expected {sequence}"
                )
            if primary.device != device:
                raise AssertionError(
                    f"{receiver_name} primary is on {primary.device}, expected {device}"
                )
            if config.is_dtype_suite:
                expected_dtype = config.dtype_names[sequence % len(config.dtype_names)]
                if str(primary.dtype) != expected_dtype:
                    raise AssertionError(
                        f"{receiver_name} got dtype {primary.dtype}, "
                        f"expected {expected_dtype}"
                    )
                received_dtypes.add(str(primary.dtype))
            _nvtx_push(config.profile, "harness_checksum")
            try:
                checksum = int(
                    primary.contiguous().view(torch.uint8).sum(dtype=torch.int64).item()
                )
                if layer_hidden is not None:
                    if layer_hidden.device != device:
                        raise AssertionError(
                            f"{receiver_name} metadata is on {layer_hidden.device}, "
                            f"expected {device}"
                        )
                    checksum += int(layer_hidden.sum(dtype=torch.int64).item())
            finally:
                _nvtx_pop(config.profile)
            expected = expected_primary + (
                expected_layer if layer_hidden is not None else 0
            )
            if checksum != expected:
                raise AssertionError(
                    f"{receiver_name} checksum {checksum} != expected {expected}"
                )
            if cpu_view is not None:
                if cpu_view.device.type != "cpu":
                    raise AssertionError(
                        f"{receiver_name} CPU view reconstructed on {cpu_view.device}"
                    )
                if int(cpu_view.item()) != 0:
                    raise AssertionError("CPU view value changed in transit")
                if cpu_view.untyped_storage().nbytes() != cpu_view.element_size():
                    raise AssertionError(
                        "CPU view retained a larger-than-logical backing storage"
                    )

            received_ns = time.perf_counter_ns()
            if sequence >= config.warmup_transfers_per_target:
                latencies_ms.append((received_ns - sent_ns) / 1_000_000.0)
            if sequence == config.warmup_transfers_per_target:
                # One post-latency sample records envelope size without adding a
                # second msgpack pass to every measured transfer.
                control_bytes_sample = len(serialize_message(msg))
            incoming = None
            primary = None
            layer_hidden = None
            metadata = None
            payload = None
            stream_item = None
            cpu_view = None
            msg = None
            completion_q.put((receiver_name, sequence))
        latency = _latency_summary(latencies_ms)
        logical_bytes = config.tensor_bytes + (
            config.metadata_bytes if config.is_stream else 0
        )
        result = {
            "role": receiver_name,
            "wire_types": sorted(wire_types),
            "dtypes": sorted(received_dtypes),
            "pool_refcounter_offsets": sorted(pool_refcounter_offsets),
            "control_bytes_sample": control_bytes_sample,
            "logical_bytes": logical_bytes,
            "logical_gbps_p50": _logical_gbps(logical_bytes, latency["p50_ms"]),
            "latency": latency,
            "memory_before": before,
            "memory_peak": _memory_snapshot(device),
        }
    finally:
        _nvtx_pop(config.profile)
        relay_get_calls = relay.get_calls
        ack_messages = control_plane.sent_acks
        stage._comm.close()
        await asyncio.sleep(0)
        control_plane.close()
        ControlPlaneContext.close()
        incoming = None
        layer_hidden = None
        metadata = None
        msg = None
        payload = None
        primary = None
        stream_item = None
        cpu_view = None
        stage = None
        relay = None
        gc.collect()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        after = _memory_snapshot(device)

    if result is None:
        raise RuntimeError(f"{receiver_name} completed without a result")
    result.update(
        relay_get_calls=relay_get_calls,
        ack_messages=ack_messages,
        memory_after=after,
    )
    return result


def _receiver_process(
    receiver_name: str,
    receiver_index: int,
    config: CaseConfig,
    endpoints: EndpointConfig,
    ready_q: mp.Queue,
    completion_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    try:
        result = asyncio.run(
            _receiver_run(
                receiver_name=receiver_name,
                receiver_index=receiver_index,
                config=config,
                endpoints=endpoints,
                ready_q=ready_q,
                completion_q=completion_q,
            )
        )
        result_q.put((receiver_name, "ok", result))
    except BaseException:
        detail = traceback.format_exc()
        result_q.put((receiver_name, "error", detail))
        raise


def _command_output(command: list[str]) -> str:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def _environment() -> dict[str, Any]:
    return {
        "revision": _command_output(["git", "rev-parse", "HEAD"]),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": os.sys.version,
        "allocator_config": {
            "PYTORCH_ALLOC_CONF": os.getenv("PYTORCH_ALLOC_CONF"),
            "PYTORCH_CUDA_ALLOC_CONF": os.getenv("PYTORCH_CUDA_ALLOC_CONF"),
        },
        "nvidia_smi": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,name,driver_version,memory.used",
                "--format=csv,noheader",
            ]
        ),
        "topology": _command_output(["nvidia-smi", "topo", "-m"]),
    }


def _build_endpoints(root: Path, receiver_count: int) -> EndpointConfig:
    receivers = {
        f"receiver-{index}": f"ipc://{root}/r{index}.sock"
        for index in range(receiver_count)
    }
    return EndpointConfig(
        sender=f"ipc://{root}/sender.sock",
        receivers=receivers,
        coordinator=f"ipc://{root}/coordinator.sock",
        abort=f"ipc://{root}/abort.sock",
    )


def _collect_results(
    processes: dict[str, mp.Process],
    result_q: mp.Queue,
    timeout_s: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    deadline = time.monotonic() + timeout_s
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    while len(results) + len(errors) < len(processes) and time.monotonic() < deadline:
        try:
            role, status, detail = result_q.get(
                timeout=min(0.25, max(0.0, deadline - time.monotonic()))
            )
        except queue.Empty:
            if all(not process.is_alive() for process in processes.values()):
                break
            continue
        if status == "ok":
            results[role] = detail
        else:
            errors[role] = detail

    for process in processes.values():
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for role, process in processes.items():
        if process.is_alive():
            process.terminate()
            process.join(timeout=30)
            errors.setdefault(role, "process timed out and was terminated")
        if process.exitcode != 0:
            errors.setdefault(role, f"process exited with code {process.exitcode}")
    return results, errors


def _validate_transport_contract(
    config: CaseConfig,
    results: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    sender = results.get("sender")
    receiver_names = sorted(role for role in results if role != "sender")
    if sender is None:
        return ["sender returned no result"]

    transfers_per_target = config.transfers_per_target
    total_transfers = transfers_per_target * len(config.dst_gpus)
    expected_wire = None
    if config.is_direct:
        expected_wire = (
            "TorchCudaIpcStreamChunk" if config.is_stream else "TorchCudaIpcPayload"
        )
    expected_relay_calls = 0 if config.is_direct else total_transfers
    if sender["relay_put_calls"] != expected_relay_calls:
        errors.append(
            f"sender relay puts={sender['relay_put_calls']} expected={expected_relay_calls}"
        )
    expected_acks = 0 if config.is_direct else total_transfers
    if sender["ack_messages"] != expected_acks:
        errors.append(f"sender ACKs={sender['ack_messages']} expected={expected_acks}")
    if sender["pending"] != 0:
        errors.append(f"sender retained {sender['pending']} pending transfers")
    if sender["completed"] != total_transfers:
        errors.append(
            f"sender completions={sender['completed']} expected={total_transfers}"
        )

    fanout_offsets: list[int] = []
    for receiver_name in receiver_names:
        receiver = results[receiver_name]
        wire_types = receiver["wire_types"]
        if expected_wire is not None:
            if wire_types != [expected_wire]:
                errors.append(
                    f"{receiver_name} wire_types={wire_types!r} expected={[expected_wire]!r}"
                )
        elif not wire_types or any(
            not wire.startswith("DataRef:cuda_ipc:") for wire in wire_types
        ):
            errors.append(f"{receiver_name} used unexpected pooled wire {wire_types!r}")

        expected_receiver_calls = 0 if config.is_direct else transfers_per_target
        if receiver["relay_get_calls"] != expected_receiver_calls:
            errors.append(
                f"{receiver_name} relay gets={receiver['relay_get_calls']} "
                f"expected={expected_receiver_calls}"
            )
        expected_receiver_acks = 0 if config.is_direct else transfers_per_target
        if receiver["ack_messages"] != expected_receiver_acks:
            errors.append(
                f"{receiver_name} ACKs={receiver['ack_messages']} "
                f"expected={expected_receiver_acks}"
            )
        after = receiver["memory_after"]
        if after["allocated"] != 0 or after["reserved"] != 0:
            errors.append(f"{receiver_name} retained CUDA allocator memory: {after}")

        offsets = receiver["pool_refcounter_offsets"]
        if config.is_direct:
            if offsets:
                errors.append(
                    f"{receiver_name} direct path unexpectedly mapped pool offsets {offsets}"
                )
        elif len(offsets) != 1:
            errors.append(
                f"{receiver_name} expected one stable pool token, got {offsets}"
            )
        else:
            fanout_offsets.append(offsets[0])

    sender_after = sender["memory_after"]
    if sender_after["allocated"] != 0 or sender_after["reserved"] != 0:
        errors.append(f"sender retained CUDA allocator memory: {sender_after}")
    if len(config.dst_gpus) > 1 and len(set(fanout_offsets)) != len(fanout_offsets):
        errors.append(
            f"fan-out receivers shared CUDA IPC refcounter offsets {fanout_offsets}"
        )
    return errors


def run_case(config: CaseConfig) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("stage transport harness requires CUDA")
    if config.is_direct and any(gpu != config.src_gpu for gpu in config.dst_gpus):
        raise ValueError("direct cases require sender and receiver on the same GPU")
    if not config.is_direct and any(gpu == config.src_gpu for gpu in config.dst_gpus):
        raise ValueError("pooled/fanout cases require a different destination GPU")

    environment_before = _environment()
    context = mp.get_context("spawn")
    ready_q = context.Queue()
    completion_q = context.Queue()
    result_q = context.Queue()
    with tempfile.TemporaryDirectory(prefix="sgo-c-", dir="/tmp") as temp_dir:
        endpoints = _build_endpoints(Path(temp_dir), len(config.dst_gpus))
        processes: dict[str, mp.Process] = {}
        try:
            for index, receiver_name in enumerate(endpoints.receivers):
                process = context.Process(
                    name=receiver_name,
                    target=_receiver_process,
                    args=(
                        receiver_name,
                        index,
                        config,
                        endpoints,
                        ready_q,
                        completion_q,
                        result_q,
                    ),
                )
                process.start()
                processes[receiver_name] = process

            for _ in endpoints.receivers:
                ready_q.get(timeout=config.timeout_s)

            sender = context.Process(
                name="sender",
                target=_sender_process,
                args=(config, endpoints, ready_q, completion_q, result_q),
            )
            sender.start()
            processes["sender"] = sender
            ready_q.get(timeout=config.timeout_s)
            results, errors = _collect_results(
                processes,
                result_q,
                timeout_s=max(config.timeout_s, _PROCESS_TIMEOUT_S),
            )
        finally:
            for process in processes.values():
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=30)

    environment_after = _environment()
    invariant_errors = _validate_transport_contract(config, results)
    if invariant_errors:
        errors["invariants"] = "\n".join(invariant_errors)
    return {
        "ok": not errors and results.keys() == processes.keys(),
        "config": asdict(config),
        "environment_before": environment_before,
        "environment_after": environment_after,
        "process_exitcodes": {
            role: process.exitcode for role, process in processes.items()
        },
        "results": results,
        "errors": errors,
    }


def _parse_gpu_list(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU ids must be comma-separated integers"
        ) from exc
    if not result or any(gpu < 0 for gpu in result):
        raise argparse.ArgumentTypeError("at least one non-negative GPU id is required")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "direct-payload",
            "direct-stream",
            "direct-payload-metadata",
            "direct-stream-metadata",
            "direct-abort-payload",
            "direct-abort-stream",
            "pooled-dtypes",
            "pooled-payload",
            "pooled-stream",
            "fanout",
        ),
    )
    parser.add_argument("--src-gpu", type=int, default=0)
    parser.add_argument("--dst-gpus", type=_parse_gpu_list, default=(0,))
    parser.add_argument("--tensor-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--metadata-bytes", type=int, default=16 * 1024)
    parser.add_argument("--header-bytes", type=int, default=0)
    parser.add_argument("--cpu-view-backing-bytes", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--pool-size-mb", type=int, default=128)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.case == "fanout" and len(args.dst_gpus) < 2:
        parser.error("fanout requires at least two --dst-gpus entries")
    if args.case != "fanout" and len(args.dst_gpus) != 1:
        parser.error(f"{args.case} requires exactly one destination GPU")
    for name in ("tensor_bytes", "metadata_bytes", "count", "window", "pool_size_mb"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.header_bytes < 0 or args.cpu_view_backing_bytes < 0:
        parser.error("metadata boundary sizes must be non-negative")
    return args


def main() -> int:
    args = parse_args()
    config = CaseConfig(
        case=args.case,
        src_gpu=args.src_gpu,
        dst_gpus=args.dst_gpus,
        tensor_bytes=args.tensor_bytes,
        metadata_bytes=args.metadata_bytes,
        header_bytes=args.header_bytes,
        cpu_view_backing_bytes=args.cpu_view_backing_bytes,
        warmups=args.warmups,
        count=args.count,
        window=args.window,
        pool_size_mb=args.pool_size_mb,
        timeout_s=args.timeout_s,
        profile=args.profile,
    )
    result = run_case(config)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
