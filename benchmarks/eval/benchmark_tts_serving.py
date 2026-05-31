# SPDX-License-Identifier: Apache-2.0
"""TTS serving benchmark.

The harness follows the benchmark platform contract:

- read /etc/benchmark/spec.json by default
- write outputs under /var/benchmark/out by default
- exit 0 when the harness ran, even when the server/model failed
- exit non-zero only for harness infrastructure failures

Docker:
    docker build -f benchmarks/tts_serving/Dockerfile \
      -t sglang-omni-tts-serving-benchmark .
    docker run --rm \
      -v "$PWD/spec.json:/etc/benchmark/spec.json:ro" \
      -v "$PWD/out:/var/benchmark/out" \
      sglang-omni-tts-serving-benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from pathlib import Path

import aiohttp

from benchmarks.tts_serving.artifacts import (
    ArtifactError,
    prepare_output_dir,
    write_artifacts,
    write_harness_log,
)
from benchmarks.tts_serving.assets import DEFAULT_OUT_DIR, DEFAULT_SPEC_PATH
from benchmarks.tts_serving.http_client import run_http_scenario
from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.report import build_results_report
from benchmarks.tts_serving.scenarios import Scenario, build_scenarios
from benchmarks.tts_serving.spec import BenchmarkSpec, LoadStage, SpecError, load_spec
from benchmarks.tts_serving.ws_client import run_ws_scenario


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TTS serving benchmark harness.")
    parser.add_argument("--spec", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    return parser


async def _run_benchmark(
    spec: BenchmarkSpec,
    scenarios: list[Scenario],
    harness_log: list[str],
) -> list[ScenarioResult]:
    timeout = aiohttp.ClientTimeout(total=spec.params.timeout_s)
    headers = _auth_headers(spec)
    connector = aiohttp.TCPConnector(limit=max(spec.params.max_concurrency * 2, 8))
    async with aiohttp.ClientSession(
        timeout=timeout,
        headers=headers,
        connector=connector,
    ) as session:
        results: list[ScenarioResult] = []
        for stage in spec.params.load_stages:
            stage_scenarios = [
                scenario for scenario in scenarios if scenario.stage_id == stage.id
            ]
            results.extend(
                await _run_stage(
                    session,
                    spec,
                    stage,
                    stage_scenarios,
                    harness_log,
                )
            )
        return results


async def _run_stage(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    stage: LoadStage,
    scenarios: list[Scenario],
    harness_log: list[str],
) -> list[ScenarioResult]:
    if stage.mode == "closed_loop":
        return await _run_closed_loop_stage(
            session, spec, stage, scenarios, harness_log
        )
    return await _run_scheduled_stage(session, spec, stage, scenarios, harness_log)


async def _run_closed_loop_stage(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    stage: LoadStage,
    scenarios: list[Scenario],
    harness_log: list[str],
) -> list[ScenarioResult]:
    scenario_iter = iter(scenarios)
    results: list[ScenarioResult] = []
    started = time.perf_counter()

    async def worker() -> None:
        for scenario in scenario_iter:
            actual_start = time.perf_counter()
            result = await _run_one_scenario(session, spec, scenario)
            _attach_schedule_metadata(
                result,
                stage=stage,
                planned_start=actual_start,
                actual_start=actual_start,
            )
            results.append(result)

    await asyncio.gather(
        *(worker() for _ in range(min(stage.max_concurrency, len(scenarios))))
    )
    harness_log.append(
        f"stage={stage.id} mode={stage.mode} completed {len(results)} scenarios "
        f"at concurrency={stage.max_concurrency} in {time.perf_counter() - started:.3f}s"
    )
    return results


async def _run_scheduled_stage(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    stage: LoadStage,
    scenarios: list[Scenario],
    harness_log: list[str],
) -> list[ScenarioResult]:
    semaphore = asyncio.Semaphore(stage.max_concurrency)
    stage_start = time.perf_counter()
    offsets = _planned_offsets(stage, len(scenarios), seed=spec.seed)

    async def run_planned(scenario: Scenario, offset: float) -> ScenarioResult:
        planned_start = stage_start + offset
        await asyncio.sleep(max(0.0, planned_start - time.perf_counter()))
        async with semaphore:
            actual_start = time.perf_counter()
            result = await _run_one_scenario(session, spec, scenario)
            _attach_schedule_metadata(
                result,
                stage=stage,
                planned_start=planned_start,
                actual_start=actual_start,
            )
            return result

    started = time.perf_counter()
    results: list[ScenarioResult] = []
    pending: set[asyncio.Task[ScenarioResult]] = set()
    max_pending = max(stage.max_concurrency * 4, 128)
    for scenario, offset in zip(scenarios, offsets, strict=True):
        pending.add(asyncio.create_task(run_planned(scenario, offset)))
        if len(pending) >= max_pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            results.extend(task.result() for task in done)
    if pending:
        results.extend(await asyncio.gather(*pending))
    harness_log.append(
        f"stage={stage.id} mode={stage.mode} completed {len(results)} scenarios "
        f"at concurrency={stage.max_concurrency} in {time.perf_counter() - started:.3f}s"
    )
    return results


async def _run_one_scenario(
    session: aiohttp.ClientSession,
    spec: BenchmarkSpec,
    scenario: Scenario,
) -> ScenarioResult:
    if scenario.method == "WS":
        return await run_ws_scenario(session, spec, scenario)
    return await run_http_scenario(session, spec, scenario)


def _attach_schedule_metadata(
    result: ScenarioResult,
    *,
    stage: LoadStage,
    planned_start: float,
    actual_start: float,
) -> None:
    result.stage_id = stage.id
    result.load_mode = stage.mode
    result.load_concurrency = stage.max_concurrency
    result.planned_start_s = planned_start
    result.actual_start_s = actual_start
    result.queue_wait_s = max(0.0, actual_start - planned_start)


def _planned_offsets(stage: LoadStage, request_count: int, *, seed: int) -> list[float]:
    if request_count <= 0:
        return []
    if stage.mode == "burst":
        return [0.0] * request_count
    if stage.mode == "ramp":
        return _ramp_offsets(stage, request_count)
    if stage.mode == "soak" and stage.duration_s:
        if request_count == 1:
            return [0.0]
        step = stage.duration_s / float(request_count - 1)
        return [index * step for index in range(request_count)]
    if stage.arrival_distribution == "poisson":
        rng = random.Random(f"{seed}:{stage.id}:arrival")
        elapsed = 0.0
        offsets: list[float] = []
        for _ in range(request_count):
            offsets.append(elapsed)
            elapsed += rng.expovariate(stage.request_rate)
        return offsets
    return [index / stage.request_rate for index in range(request_count)]


def _ramp_offsets(stage: LoadStage, request_count: int) -> list[float]:
    start_rate = stage.start_request_rate or stage.request_rate
    end_rate = stage.request_rate
    elapsed = 0.0
    offsets: list[float] = []
    for index in range(request_count):
        offsets.append(elapsed)
        position = index / max(request_count - 1, 1)
        current_rate = start_rate + (end_rate - start_rate) * position
        elapsed += 1.0 / current_rate
    return offsets


def _auth_headers(spec: BenchmarkSpec) -> dict[str, str]:
    if not spec.auth.api_key_env:
        return {}
    token = os.environ.get(spec.auth.api_key_env)
    if not token:
        raise RuntimeError(
            f"auth environment variable is not set: {spec.auth.api_key_env}"
        )
    return {"Authorization": f"Bearer {token}"}


def main() -> int:
    args = _build_arg_parser().parse_args()
    harness_log: list[str] = []
    try:
        spec = load_spec(args.spec)
        out_dir = prepare_output_dir(args.out)
    except (SpecError, ArtifactError) as exc:
        print(f"benchmark harness failed: {exc}")
        return 2

    scenarios = build_scenarios(spec)
    stage_request_total = sum(stage.request_count for stage in spec.params.load_stages)
    harness_log.append(
        f"loaded spec={Path(args.spec)} profile={spec.params.profile} "
        f"stage_requests={stage_request_total} scenarios={len(scenarios)} "
        f"load_stages={[stage.id for stage in spec.params.load_stages]}"
    )
    try:
        results = asyncio.run(_run_benchmark(spec, scenarios, harness_log))
        report = build_results_report(spec, results, scenarios=scenarios)
        write_artifacts(out_dir, spec, scenarios, results, report)
        write_harness_log(out_dir, harness_log)
    except ArtifactError as exc:
        print(f"benchmark harness failed: {exc}")
        return 2
    except Exception as exc:
        harness_log.append(f"unhandled harness error: {exc.__class__.__name__}: {exc}")
        report = build_results_report(
            spec,
            [],
            scenarios=scenarios,
            harness_status="error",
            harness_error=f"{exc.__class__.__name__}: {exc}",
        )
        try:
            write_artifacts(out_dir, spec, scenarios, [], report)
            write_harness_log(out_dir, harness_log)
        except ArtifactError:
            pass
        print(f"benchmark harness failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
