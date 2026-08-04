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
    async def collect_intervals(seed: int) -> list[float]:
        intervals: list[float] = []

        async def fake_sleep(interval: float) -> None:
            intervals.append(interval)

        async def send(_session, sample):  # noqa: ANN001, ANN202
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
        await runner._dispatch(None, [1, 2, 3], send)
        return intervals

    first = await collect_intervals(7)
    second = await collect_intervals(7)
    third = await collect_intervals(8)

    assert first == second
    assert first != third
