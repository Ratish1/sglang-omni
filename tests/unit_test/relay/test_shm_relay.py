# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from multiprocessing import shared_memory

import pytest
import torch

from sglang_omni.relay.shm import ShmRelay


def test_shm_put_timeout_unlinks_block_and_releases_credit() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", device="cpu", credits=1)
        tensor = torch.arange(16, dtype=torch.uint8)
        op = await relay.put_async(tensor, request_id="r0")
        shm_name = op.metadata["transfer_info"]["shm_name"]
        probe = shared_memory.SharedMemory(name=shm_name)
        probe.close()

        with pytest.raises(TimeoutError, match="was not consumed in time"):
            await op.wait_for_completion(timeout=0.0)
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=shm_name)

        # The timeout path released the semaphore credit; another put should not
        # block even though the first transfer failed.
        op2 = await asyncio.wait_for(
            relay.put_async(tensor, request_id="r1"),
            timeout=1.0,
        )
        shm_name2 = op2.metadata["transfer_info"]["shm_name"]
        try:
            probe = shared_memory.SharedMemory(name=shm_name2)
            probe.close()
        finally:
            with pytest.raises(TimeoutError):
                await op2.wait_for_completion(timeout=0.0)
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=shm_name2)

    asyncio.run(run())


def test_shm_round_trip_preserves_logical_transfer_length() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="round-trip", device="cpu", credits=1)
        tensors = (
            torch.empty(0, dtype=torch.uint8),
            torch.arange(17, dtype=torch.uint8),
        )

        for index, source in enumerate(tensors):
            op = await relay.put_async(source, request_id=f"r{index}")
            transfer_info = op.metadata["transfer_info"]
            logical_size = source.view(torch.uint8).numel()
            assert transfer_info["size"] == logical_size

            probe = shared_memory.SharedMemory(name=transfer_info["shm_name"])
            try:
                assert probe.size >= max(logical_size, 1)
            finally:
                probe.close()

            destination = torch.empty(logical_size, dtype=torch.uint8)
            get_op = await relay.get_async(
                op.metadata,
                destination,
                request_id=f"r{index}",
            )
            await get_op.wait_for_completion()
            op.mark_receiver_done()
            await op.wait_for_completion()

            assert torch.equal(destination, source.view(torch.uint8))
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=transfer_info["shm_name"])

    asyncio.run(run())
