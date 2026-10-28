# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import os
import sys

import pytest
import torch

from sglang_omni.relay.shm import ShmRelay


def test_shm_put_timeout_unlinks_block_and_releases_credit() -> None:
    async def run() -> None:
        relay = ShmRelay(engine_id="sender", device="cpu", credits=1)
        tensor = torch.arange(16, dtype=torch.uint8)
        op = await relay.put_async(tensor, request_id="r0")
        shm_name = op.metadata["transfer_info"]["shm_name"]
        shm_path = f"/dev/shm/{shm_name}"

        assert os.path.exists(shm_path)
        with pytest.raises(TimeoutError, match="was not consumed in time"):
            await op.wait_for_completion(timeout=0.0)
        assert not os.path.exists(shm_path)

        # The timeout path released the semaphore credit; another put should not
        # block even though the first transfer failed.
        op2 = await asyncio.wait_for(
            relay.put_async(tensor, request_id="r1"),
            timeout=1.0,
        )
        shm_name2 = op2.metadata["transfer_info"]["shm_name"]
        shm_path2 = f"/dev/shm/{shm_name2}"
        try:
            assert os.path.exists(shm_path2)
        finally:
            with pytest.raises(TimeoutError):
                await op2.wait_for_completion(timeout=0.0)
            assert not os.path.exists(shm_path2)

    asyncio.run(run())


@pytest.mark.skipif(sys.platform != "linux", reason="blocks are mapped on Linux")
@pytest.mark.parametrize("element_count", [0, 1001])
def test_shm_mapped_read_returns_writable_block_bytes_and_unlinks_block(
    element_count: int,
) -> None:
    async def scenario() -> None:
        relay = ShmRelay(engine_id="e", device="cpu")
        source = torch.randn(element_count, dtype=torch.float32)
        put_operation = await relay.put_async(source, request_id="r0")
        transfer_info = put_operation.metadata["transfer_info"]

        received = relay.read_mapped(
            put_operation.metadata, transfer_info["size"]
        ).view(torch.float32)

        assert torch.equal(received, source)
        assert not os.path.exists(f"/dev/shm/{transfer_info['shm_name']}")
        received.add_(1.0)
        assert torch.equal(received, source + 1.0)

    asyncio.run(scenario())


def test_shm_round_trips_zero_byte_tensor() -> None:
    async def scenario() -> None:
        relay = ShmRelay(engine_id="e", device="cpu")
        put = await relay.put_async(torch.empty(0, dtype=torch.long), request_id="r0")
        assert put.metadata["transfer_info"]["size"] == 0
        dest = torch.empty(0, dtype=torch.long)
        get = await relay.get_async(put.metadata, dest)
        await get.wait_for_completion(timeout=1.0)
        assert dest.numel() == 0

    asyncio.run(scenario())
