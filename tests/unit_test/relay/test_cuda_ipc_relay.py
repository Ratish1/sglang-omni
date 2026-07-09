# SPDX-License-Identifier: Apache-2.0
"""GPU-direct round-trip tests for the CUDA-IPC relay.

The relay shares a CUDA buffer across processes via an IPC handle; the receiver
opens it and copies into its own buffer (a peer/NVLink copy when the GPUs
differ). These run two real processes because a process cannot open its own CUDA
IPC handle.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue
import time
import traceback
from contextlib import nullcontext

import pytest
import torch

import sglang_omni.relay.cuda_ipc as cuda_ipc_module
from sglang_omni.relay.cuda_ipc import (
    CudaIpcPutOperation,
    CudaIpcRelay,
    _ContiguousSlotAllocator,
)

_N = 1024 * 1024  # 1 MiB payload
_REUSE_COUNT = int(os.getenv("SGLANG_OMNI_CUDA_IPC_TEST_TRANSFERS", "3"))
_PROCESS_TIMEOUT_S = 120.0
if _REUSE_COUNT < 1:
    raise ValueError("SGLANG_OMNI_CUDA_IPC_TEST_TRANSFERS must be positive")


def _expected(n: int, transfer_index: int = 0) -> torch.Tensor:
    return ((torch.arange(n, dtype=torch.int64) + transfer_index * 17) % 251).to(
        torch.uint8
    )


def test_cuda_ipc_put_timeout_fails_relay_without_releasing_slot() -> None:
    released = False
    failed: list[BaseException] = []

    def release() -> None:
        nonlocal released
        released = True

    async def run() -> None:
        op = CudaIpcPutOperation(
            metadata={},
            ready_event=object(),  # type: ignore[arg-type]
            source_tensor=object(),  # type: ignore[arg-type]
            slot_index=0,
            request_id="r",
            size=1,
            release_cb=release,
            fail_cb=failed.append,
        )
        with pytest.raises(TimeoutError):
            await op.wait_for_completion(timeout=0.0)

    asyncio.run(run())

    assert released is False
    assert len(failed) == 1
    assert isinstance(failed[0], TimeoutError)


def test_cuda_ipc_relay_failure_wakes_blocked_slot_acquire() -> None:
    class BlockingAllocator:
        def __init__(self) -> None:
            self.released: list[tuple[int, int]] = []

        async def acquire_async(
            self, num_slots: int, *, capture_layout: bool = False
        ) -> int:
            await asyncio.Event().wait()
            return 0

        def release(self, offset: int, num_slots: int) -> None:
            self.released.append((offset, num_slots))

    async def run() -> BlockingAllocator:
        relay = CudaIpcRelay(engine_id="sender", device="cuda:0")
        allocator = BlockingAllocator()
        task = asyncio.create_task(relay._acquire_slots(allocator, 2))
        await asyncio.sleep(0)
        relay._mark_failed(TimeoutError("ack timeout"))
        with pytest.raises(RuntimeError, match="cuda_ipc relay failed"):
            _ = await task
        return allocator

    allocator = asyncio.run(run())
    assert allocator.released == []


def test_cuda_ipc_put_fails_fast_after_relay_failure() -> None:
    async def run() -> None:
        relay = CudaIpcRelay(engine_id="sender", device="cuda:0")
        relay._mark_failed(TimeoutError("ack timeout"))
        with pytest.raises(RuntimeError, match="cuda_ipc relay failed"):
            await relay.put_async(torch.zeros(1, dtype=torch.uint8))

    asyncio.run(run())


def test_cuda_ipc_put_requires_receiver_identity_before_pool_allocation() -> None:
    class FakeCudaTensor:
        is_cuda = True
        device = torch.device("cuda:0")

    async def run() -> None:
        relay = CudaIpcRelay(engine_id="sender", device="cuda:0")
        try:
            with pytest.raises(ValueError, match="stable receiver_id"):
                await relay.put_async(FakeCudaTensor())  # type: ignore[arg-type]
            assert relay._pool_tensor is None
        finally:
            relay.close()

    asyncio.run(run())


def test_cuda_ipc_default_pool_uses_small_slots() -> None:
    relay = CudaIpcRelay(engine_id="sender", device="cuda:0")
    assert relay.slot_size == 64 * 1024
    assert relay.pool_size == 1024 * 1024 * 1024
    assert relay.slot_count == 16 * 1024


def test_cuda_ipc_pool_size_and_slot_size_are_configurable() -> None:
    relay = CudaIpcRelay(
        engine_id="sender",
        device="cuda:0",
        pool_size_mb=1,
        slot_size_kb=256,
    )
    assert relay.slot_size == 256 * 1024
    assert relay.pool_size == 1024 * 1024
    assert relay.slot_count == 4


def test_contiguous_slot_allocator_waits_for_contiguous_range() -> None:
    async def run() -> None:
        allocator = _ContiguousSlotAllocator(slot_count=4, slot_size=8)
        first = (await allocator.acquire_async(1)).offset
        middle = (await allocator.acquire_async(1)).offset
        tail = (await allocator.acquire_async(1)).offset
        assert (first, middle, tail) == (0, 8, 16)

        allocator.release(middle, 1)
        blocked = asyncio.create_task(allocator.acquire_async(2, capture_layout=True))
        await asyncio.sleep(0)
        assert blocked.done() is False

        allocator.release(tail, 1)
        allocation = await asyncio.wait_for(blocked, timeout=1.0)
        assert allocation.offset == 8
        assert allocation.wait_rounds == 1
        assert allocation.last_failed_free_slots == 2
        assert allocation.last_failed_largest_free_run == 1
        allocator.release(first, 1)
        allocator.release(8, 2)

    asyncio.run(run())


def test_contiguous_slot_allocator_rejects_double_release() -> None:
    async def run() -> None:
        allocator = _ContiguousSlotAllocator(slot_count=2, slot_size=8)
        offset = (await allocator.acquire_async(2)).offset
        allocator.release(offset, 2)
        with pytest.raises(RuntimeError, match="released twice"):
            allocator.release(offset, 2)

    asyncio.run(run())


def test_cuda_ipc_pool_token_is_cached_per_receiver_without_reallocating(
    monkeypatch,
) -> None:
    class FakeTensor:
        is_cuda = True

        def contiguous(self):
            return self

        def view(self, *args):
            del args
            return self

        def reshape(self, *args):
            del args
            return self

        def numel(self) -> int:
            return 1

        def __getitem__(self, key):
            del key
            return self

        def copy_(self, source, *, non_blocking: bool = False):
            del source, non_blocking
            return self

    class FakeEvent:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def record(self, stream) -> None:
            del stream

        def ipc_handle(self) -> bytes:
            return b"ready-event"

    async def run() -> tuple[int, list[dict], list[dict]]:
        relay = CudaIpcRelay(
            engine_id="sender",
            device="cuda:0",
            pool_size_mb=1,
        )
        pool = FakeTensor()
        allocator = _ContiguousSlotAllocator(
            slot_count=1,
            slot_size=relay.slot_size,
        )
        allocations = 0
        descriptors: list[dict] = []

        def ensure_local_pool() -> None:
            nonlocal allocations
            if relay._pool_tensor is not None:
                return
            allocations += 1
            relay._pool_tensor = pool  # type: ignore[assignment]
            relay._pool_id = "pool"
            relay._allocator = allocator

        def dump_pool(tensor) -> dict:
            assert tensor is pool
            descriptor = {
                "storage_handle": b"same-allocation",
                "ref_counter_offset": len(descriptors),
            }
            descriptors.append(descriptor)
            return descriptor

        relay._ensure_local_pool = ensure_local_pool  # type: ignore[method-assign]
        monkeypatch.setattr(cuda_ipc_module, "_dump_cuda_storage_handle", dump_pool)
        monkeypatch.setattr(cuda_ipc_module, "_comm_trace_enabled", lambda: False)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
        monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
        monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

        try:
            published = []
            publications = (
                ("first", "receiver-a"),
                ("second", "receiver-a"),
                ("third", "receiver-b"),
            )
            for request_id, receiver_id in publications:
                op = await relay.put_async(
                    FakeTensor(),
                    request_id=request_id,
                    receiver_id=receiver_id,
                )
                published.append(op.metadata["cuda_ipc"]["pool_storage"])
                op.mark_receiver_done()
                await op.wait_for_completion()
            return allocations, descriptors, published
        finally:
            relay.close()

    allocations, descriptors, published = asyncio.run(run())

    assert allocations == 1
    assert len(descriptors) == 2
    assert published[0] is published[1]
    assert published[0]["storage_handle"] == published[2]["storage_handle"]
    assert published[0]["ref_counter_offset"] != published[2]["ref_counter_offset"]


def test_cuda_ipc_remote_pool_rejects_descriptor_change_after_import(
    monkeypatch,
) -> None:
    relay = CudaIpcRelay(engine_id="receiver", device="cuda:0")
    pool = object()
    loads: list[dict] = []

    def load_pool(storage_meta, *, device):
        assert device == torch.device("cuda:0")
        loads.append(storage_meta)
        return pool

    monkeypatch.setattr(cuda_ipc_module, "_load_cuda_storage_handle", load_pool)

    def metadata(ref_counter_offset: int) -> dict:
        return {
            "cuda_ipc": {
                "pool_id": "sender-pool",
                "pool_storage": {
                    "storage_handle": b"allocation",
                    "ref_counter_handle": b"receiver-token",
                    "ref_counter_offset": ref_counter_offset,
                },
            }
        }

    first = metadata(1)
    try:
        assert relay._get_remote_pool(first, device=torch.device("cuda:0")) is pool
        assert relay._get_remote_pool(first, device=torch.device("cuda:0")) is pool
        with pytest.raises(ValueError, match="descriptor changed"):
            relay._get_remote_pool(metadata(2), device=torch.device("cuda:0"))
    finally:
        relay.close()

    assert loads == [first["cuda_ipc"]["pool_storage"]]


def _sender(
    src_gpu: int,
    meta_q: mp.Queue,
    ack_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    relay = None
    try:
        torch.cuda.set_device(src_gpu)
        relay = CudaIpcRelay(
            engine_id="sender",
            device=f"cuda:{src_gpu}",
            pool_size_mb=1,
        )

        async def run() -> None:
            for index in range(_REUSE_COUNT):
                request_id = f"r-{index}"
                buf = _expected(_N, index).to(f"cuda:{src_gpu}")
                op = await relay.put_async(
                    buf,
                    request_id=request_id,
                    receiver_id="receiver",
                )
                meta_q.put((request_id, op.metadata))

                ack_request_id, ack_error = ack_q.get(timeout=60)
                if ack_request_id != request_id:
                    raise RuntimeError(
                        f"received ack for {ack_request_id!r}, expected {request_id!r}"
                    )
                if ack_error is None:
                    op.mark_receiver_done()
                else:
                    op.mark_receiver_failed(RuntimeError(ack_error))
                await op.wait_for_completion(timeout=60)

        asyncio.run(run())
        result_q.put(("sender", "ok", None))
    except BaseException:
        result_q.put(("sender", "err", traceback.format_exc()))
        raise
    finally:
        if relay is not None:
            relay.close()


def _receiver(
    dst_gpu: int,
    meta_q: mp.Queue,
    ack_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    relay = None
    request_id = None
    try:
        torch.cuda.set_device(dst_gpu)
        relay = CudaIpcRelay(engine_id="receiver", device=f"cuda:{dst_gpu}")
        for index in range(_REUSE_COUNT):
            request_id, metadata = meta_q.get(timeout=60)
            assert request_id == f"r-{index}"

            async def run() -> torch.Tensor:
                size = metadata["transfer_info"]["size"]
                dest = torch.empty(size, dtype=torch.uint8, device=f"cuda:{dst_gpu}")
                op = await relay.get_async(metadata, dest, request_id=request_id)
                await op.wait_for_completion(timeout=60)
                return dest

            dest = asyncio.run(run())
            expected = _expected(_N, index).to(f"cuda:{dst_gpu}")
            assert torch.equal(dest, expected), f"payload mismatch for {request_id}"

            # Drop the imported pool before the final ACK permits the exporter to
            # release its pool and exit. Earlier ACKs intentionally exercise slot
            # reuse through the receiver's cached IPC mapping.
            if index == _REUSE_COUNT - 1:
                relay.close()
                relay = None
            ack_q.put((request_id, None))

        result_q.put(("receiver", "ok", None))
    except BaseException:
        error = traceback.format_exc()
        if relay is not None:
            relay.close()
            relay = None
        ack_q.put((request_id, error))
        result_q.put(("receiver", "err", error))
        raise
    finally:
        if relay is not None:
            relay.close()


def _fanout_sender(
    src_gpu: int,
    meta_queues: list[mp.Queue],
    ack_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    relay = None
    try:
        torch.cuda.set_device(src_gpu)
        relay = CudaIpcRelay(
            engine_id="fanout-sender",
            device=f"cuda:{src_gpu}",
            pool_size_mb=2,
        )

        async def run() -> None:
            operations = {}
            descriptors = []
            for index, meta_q in enumerate(meta_queues):
                request_id = f"fanout-{index}"
                op = await relay.put_async(
                    _expected(_N, index).to(f"cuda:{src_gpu}"),
                    request_id=request_id,
                    receiver_id=f"receiver-{index}",
                )
                operations[request_id] = op
                descriptors.append(op.metadata["cuda_ipc"]["pool_storage"])
                meta_q.put((request_id, index, op.metadata))

            assert len({item["storage_handle"] for item in descriptors}) == 1
            assert len(
                {
                    (item["ref_counter_handle"], item["ref_counter_offset"])
                    for item in descriptors
                }
            ) == len(meta_queues)

            for _ in meta_queues:
                request_id, error = ack_q.get(timeout=60)
                op = operations[request_id]
                if error is None:
                    op.mark_receiver_done()
                else:
                    op.mark_receiver_failed(RuntimeError(error))
                await op.wait_for_completion(timeout=60)

        asyncio.run(run())
        result_q.put(("sender", "ok", None))
    except BaseException:
        result_q.put(("sender", "err", traceback.format_exc()))
        raise
    finally:
        if relay is not None:
            relay.close()


def _fanout_receiver(
    dst_gpu: int,
    role: str,
    meta_q: mp.Queue,
    ack_q: mp.Queue,
    result_q: mp.Queue,
) -> None:
    relay = None
    request_id = None
    try:
        torch.cuda.set_device(dst_gpu)
        relay = CudaIpcRelay(engine_id=role, device=f"cuda:{dst_gpu}")
        request_id, index, metadata = meta_q.get(timeout=60)

        async def run() -> torch.Tensor:
            size = metadata["transfer_info"]["size"]
            dest = torch.empty(size, dtype=torch.uint8, device=f"cuda:{dst_gpu}")
            op = await relay.get_async(metadata, dest, request_id=request_id)
            await op.wait_for_completion(timeout=60)
            return dest

        dest = asyncio.run(run())
        expected = _expected(_N, index).to(f"cuda:{dst_gpu}")
        assert torch.equal(dest, expected), f"payload mismatch for {request_id}"
        relay.close()
        relay = None
        ack_q.put((request_id, None))
        result_q.put((role, "ok", None))
    except BaseException:
        error = traceback.format_exc()
        if relay is not None:
            relay.close()
            relay = None
        ack_q.put((request_id, error))
        result_q.put((role, "err", error))
        raise


def _assert_processes_succeed(
    processes: list[mp.Process],
    result_q: mp.Queue,
    expected_roles: set[str],
) -> None:
    deadline = time.monotonic() + _PROCESS_TIMEOUT_S
    results: dict[str, tuple[str, str | None]] = {}
    while len(results) < len(expected_roles) and time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            role, status, detail = result_q.get(timeout=min(0.25, remaining))
            results[role] = (status, detail)
        except queue.Empty:
            if not any(process.is_alive() for process in processes):
                break

    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join()

    details = "\n".join(
        f"{role}: {status}\n{detail or ''}".rstrip()
        for role, (status, detail) in sorted(results.items())
    )
    for process in processes:
        assert process.exitcode == 0, details
    assert results.keys() == expected_roles, details
    assert all(status == "ok" for status, _ in results.values()), details


def _run_case(src_gpu: int, dst_gpu: int) -> None:
    ctx = mp.get_context("spawn")
    meta_q, ack_q, result_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    sender = ctx.Process(target=_sender, args=(src_gpu, meta_q, ack_q, result_q))
    receiver = ctx.Process(target=_receiver, args=(dst_gpu, meta_q, ack_q, result_q))
    sender.start()
    receiver.start()

    _assert_processes_succeed([sender, receiver], result_q, {"sender", "receiver"})


def _run_fanout_case(gpu: int) -> None:
    ctx = mp.get_context("spawn")
    meta_queues = [ctx.Queue(), ctx.Queue()]
    ack_q, result_q = ctx.Queue(), ctx.Queue()
    sender = ctx.Process(
        target=_fanout_sender,
        args=(gpu, meta_queues, ack_q, result_q),
    )
    receivers = [
        ctx.Process(
            target=_fanout_receiver,
            args=(gpu, f"receiver-{index}", meta_q, ack_q, result_q),
        )
        for index, meta_q in enumerate(meta_queues)
    ]
    processes = [sender, *receivers]
    for process in processes:
        process.start()

    _assert_processes_succeed(
        processes,
        result_q,
        {"sender", "receiver-0", "receiver-1"},
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ipc_same_gpu_round_trip() -> None:
    _run_case(0, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_ipc_same_gpu_fanout_uses_one_token_per_receiver() -> None:
    _run_fanout_case(0)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires >= 2 GPUs for cross-GPU transfer"
)
def test_cuda_ipc_cross_gpu_round_trip() -> None:
    _run_case(0, 1)
