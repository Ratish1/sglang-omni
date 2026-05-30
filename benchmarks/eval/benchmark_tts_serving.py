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
from benchmarks.tts_serving.spec import BenchmarkSpec, SpecError, load_spec
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
        semaphore = asyncio.Semaphore(spec.params.max_concurrency)

        async def run_one(scenario: Scenario) -> ScenarioResult:
            async with semaphore:
                if scenario.method == "WS":
                    return await run_ws_scenario(session, spec, scenario)
                return await run_http_scenario(session, spec, scenario)

        tasks: list[asyncio.Task[ScenarioResult]] = []
        rng = random.Random(spec.seed)
        for scenario in scenarios:
            if spec.params.request_rate != float("inf"):
                await asyncio.sleep(rng.expovariate(spec.params.request_rate))
            tasks.append(asyncio.create_task(run_one(scenario)))
        started = time.perf_counter()
        results = list(await asyncio.gather(*tasks))
        harness_log.append(
            f"completed {len(results)} scenarios in {time.perf_counter() - started:.3f}s"
        )
        return results


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
    harness_log.append(
        f"loaded spec={Path(args.spec)} profile={spec.params.profile} "
        f"traffic_requests={spec.params.total_requests} scenarios={len(scenarios)}"
    )
    try:
        results = asyncio.run(_run_benchmark(spec, scenarios, harness_log))
        report = build_results_report(spec, results)
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
