# SPDX-License-Identifier: Apache-2.0
"""BenchmarkRunner: warmup + concurrent dispatch with semaphore and rate limiting."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

from benchmarks.benchmarker.data import RequestResult

logger = logging.getLogger(__name__)

SendFn = Callable[[aiohttp.ClientSession, Any], Coroutine[Any, Any, RequestResult]]


@dataclass
class RunConfig:
    max_concurrency: int = 1
    request_rate: float = float("inf")
    warmup: int = 1
    disable_tqdm: bool = False
    timeout_s: int = 300
    request_rate_seed: int | None = None


class BenchmarkRunner:
    """Support concurrent requests sending in a single benchmark run.

    Note (chenyang):
    max_concurrency is default to 1, thus all the requests are runs sequentially.

    TODO (chenyang):
    Current concurrency implementation of models are not fully supported.
    https://github.com/sgl-project/sglang-omni/issues/229
    https://github.com/sgl-project/sglang-omni/issues/228
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.wall_clock_s: float = 0.0

    async def run(self, samples: list, send_fn: SendFn) -> list[RequestResult]:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.config.warmup > 0:
                await self._warmup(session, samples, send_fn)

            logger.info(
                "Benchmarking %d requests (max_concurrency=%s)...",
                len(samples),
                self.config.max_concurrency,
            )
            t0 = time.perf_counter()
            results = await self._dispatch(session, samples, send_fn)
            self.wall_clock_s = time.perf_counter() - t0
        return results

    async def _warmup(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> None:
        count = min(self.config.warmup, len(samples))
        logger.info("Warmup (%d requests)...", count)
        for i in range(count):
            result = await send_fn(session, samples[i])
            status = "ok" if result.is_success else result.error
            logger.info("  warmup %d/%d: %s", i + 1, count, status)

    async def _dispatch(
        self,
        session: aiohttp.ClientSession,
        samples: list,
        send_fn: SendFn,
    ) -> list[RequestResult]:
        semaphore = (
            asyncio.Semaphore(self.config.max_concurrency)
            if self.config.max_concurrency
            else None
        )
        pbar = tqdm(total=len(samples), disable=self.config.disable_tqdm)

        async def _limited(
            sample: Any,
            *,
            scheduled_arrival_ns: int,
            task_created_ns: int,
        ) -> RequestResult:
            permit_wait_start_ns = time.perf_counter_ns()
            if semaphore:
                async with semaphore:
                    permit_acquired_ns = time.perf_counter_ns()
                    send_invoked_ns = time.perf_counter_ns()
                    result = await send_fn(session, sample)
            else:
                permit_acquired_ns = time.perf_counter_ns()
                send_invoked_ns = time.perf_counter_ns()
                result = await send_fn(session, sample)
            response_complete_ns = time.perf_counter_ns()
            result.client_scheduled_arrival_ns = scheduled_arrival_ns
            result.client_task_created_ns = task_created_ns
            result.client_permit_wait_start_ns = permit_wait_start_ns
            result.client_permit_acquired_ns = permit_acquired_ns
            result.client_send_invoked_ns = send_invoked_ns
            result.client_response_complete_ns = response_complete_ns
            pbar.update(1)
            return result

        try:
            tasks: list[asyncio.Task] = []
            rng = np.random.default_rng(self.config.request_rate_seed)
            dispatch_origin_ns = time.perf_counter_ns()
            scheduled_arrival_ns = dispatch_origin_ns
            for sample in samples:
                if self.config.request_rate != float("inf"):
                    interval = rng.exponential(1.0 / self.config.request_rate)
                    scheduled_arrival_ns += int(interval * 1e9)
                    delay_s = max(
                        0.0,
                        (scheduled_arrival_ns - time.perf_counter_ns()) / 1e9,
                    )
                    await asyncio.sleep(delay_s)
                task_created_ns = time.perf_counter_ns()
                tasks.append(
                    asyncio.create_task(
                        _limited(
                            sample,
                            scheduled_arrival_ns=scheduled_arrival_ns,
                            task_created_ns=task_created_ns,
                        )
                    )
                )

            results: list[RequestResult] = list(await asyncio.gather(*tasks))
        finally:
            pbar.close()
        return results
