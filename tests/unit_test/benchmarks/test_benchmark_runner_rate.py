# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig


@pytest.mark.asyncio
async def test_poisson_arrival_seed_is_reproducible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def collect_intervals(seed: int) -> list[int]:
        async def fake_sleep(_interval: float) -> None:
            return None

        async def send(_session, sample):
            await asyncio.sleep(0)
            return RequestResult(request_id=str(sample), is_success=True)

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        runner = BenchmarkRunner(
            RunConfig(
                max_concurrency=2,
                request_rate=10.0,
                request_rate_seed=seed,
                disable_tqdm=True,
            )
        )
        results = await runner._dispatch(None, [1, 2, 3], send)
        arrivals = [int(result.client_scheduled_arrival_ns) for result in results]
        return [
            arrivals[0],
            arrivals[1] - arrivals[0],
            arrivals[2] - arrivals[1],
        ][1:]

    first = await collect_intervals(7)
    second = await collect_intervals(7)
    third = await collect_intervals(8)

    assert first == second
    assert first != third


@pytest.mark.asyncio
async def test_dispatch_records_client_queue_and_send_boundaries() -> None:
    gate = asyncio.Event()

    async def send(_session, sample):
        if sample == 1:
            await gate.wait()
        return RequestResult(request_id=str(sample), is_success=True)

    runner = BenchmarkRunner(RunConfig(max_concurrency=1, disable_tqdm=True))
    task = asyncio.create_task(runner._dispatch(None, [1, 2], send))
    await asyncio.sleep(0)
    gate.set()
    results = await task

    for result in results:
        assert result.client_scheduled_arrival_ns is not None
        assert result.client_task_created_ns is not None
        assert result.client_permit_wait_start_ns is not None
        assert result.client_permit_acquired_ns is not None
        assert result.client_send_invoked_ns is not None
        assert result.client_response_complete_ns is not None
        assert (
            result.client_permit_wait_start_ns
            <= result.client_permit_acquired_ns
            <= result.client_send_invoked_ns
            <= result.client_response_complete_ns
        )
    assert results[1].client_permit_acquired_ns > results[1].client_permit_wait_start_ns
