# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from multiprocessing import shared_memory

import pytest
import torch

from sglang_omni.relay.shm import ShmRelay


def _shared_memory_exists(name: str) -> bool:
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    shm.close()
    return True


@pytest.mark.parametrize(
    ("shape", "expected_bytes"),
    [
        ((11,), 44),
        ((0,), 0),
        ((2, 0, 4), 0),
    ],
)
def test_shm_round_trip_preserves_logical_length(shape, expected_bytes) -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="round-trip", device="cpu")
        source = (
            torch.arange(11, dtype=torch.float32)
            if shape == (11,)
            else torch.empty(shape)
        )
        put_op = await relay.put_async(source, request_id="r")

        assert put_op.metadata["transfer_info"]["size"] == expected_bytes

        destination = torch.empty_like(source)
        get_op = await relay.get_async(put_op.metadata, destination, request_id="r")
        await get_op.wait_for_completion()
        put_op.mark_receiver_done()
        await put_op.wait_for_completion()

        assert destination.shape == source.shape
        assert torch.equal(destination, source)

    asyncio.run(run())


def test_shm_put_timeout_unlinks_block_and_releases_credit() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", device="cpu", credits=1)
        tensor = torch.arange(16, dtype=torch.uint8)
        op = await relay.put_async(tensor, request_id="r0")
        shm_name = op.metadata["transfer_info"]["shm_name"]

        assert _shared_memory_exists(shm_name)
        with pytest.raises(TimeoutError, match="was not consumed in time"):
            await op.wait_for_completion(timeout=0.0)
        assert not _shared_memory_exists(shm_name)

        # The timeout path released the semaphore credit; another put should not
        # block even though the first transfer failed.
        op2 = await asyncio.wait_for(
            relay.put_async(tensor, request_id="r1"),
            timeout=1.0,
        )
        shm_name2 = op2.metadata["transfer_info"]["shm_name"]
        try:
            assert _shared_memory_exists(shm_name2)
        finally:
            with pytest.raises(TimeoutError):
                await op2.wait_for_completion(timeout=0.0)
            assert not _shared_memory_exists(shm_name2)

    asyncio.run(run())
