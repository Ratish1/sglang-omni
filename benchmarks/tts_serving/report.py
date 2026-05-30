# SPDX-License-Identifier: Apache-2.0
"""Report aggregation for the TTS serving benchmark."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.spec import BenchmarkSpec


def build_results_report(
    spec: BenchmarkSpec,
    results: list[ScenarioResult],
    *,
    harness_status: str = "ok",
    harness_error: str | None = None,
) -> dict[str, Any]:
    total = len(results)
    traffic_total = sum(
        1 for result in results if result.category != "capability_probe"
    )
    capability_total = total - traffic_total
    succeeded = sum(1 for result in results if _result_passed(spec, result))
    failed = total - succeeded
    capabilities = _capabilities(results)
    category_counts = Counter(result.category for result in results)
    status_counts = Counter(result.status for result in results)
    latencies = [r.latency_s for r in results if r.latency_s > 0]
    ttfas = [r.ttfa_s for r in results if r.ttfa_s is not None]
    rtfs = [r.rtf for r in results if r.rtf > 0]
    passed = harness_status == "ok" and _is_benchmark_passed(
        spec, results, capabilities
    )
    return {
        "schema_version": 1,
        "harness_status": harness_status,
        "harness_error": harness_error,
        "overall": {
            "passed": passed,
            "total": total,
            "traffic_total": traffic_total,
            "capability_probe_total": capability_total,
            "succeeded": succeeded,
            "failed": failed,
        },
        "config": {
            "test_type": spec.test_type,
            "base_url": spec.base_url,
            "model_name": spec.model_name,
            "run_id": spec.run_id,
            "seed": spec.seed,
            "profile": spec.params.profile,
            "total_requests": spec.params.total_requests,
            "max_concurrency": spec.params.max_concurrency,
            "request_rate": (
                "inf"
                if spec.params.request_rate == float("inf")
                else spec.params.request_rate
            ),
            "timeout_s": spec.params.timeout_s,
            "enabled_endpoints": list(spec.params.enabled_endpoints),
        },
        "capabilities": capabilities,
        "metrics": {
            "latency_s": _summary(latencies),
            "ttfa_s": _summary(ttfas),
            "rtf": _summary(rtfs),
            "status_counts": dict(status_counts),
            "category_counts": dict(category_counts),
            "by_category": _by_category(spec, results),
        },
        "failures": [
            result.to_json() for result in results if not _result_passed(spec, result)
        ][:100],
        "missing_capabilities": [
            result.to_json() for result in results if result.status == "missing"
        ],
    }


def _is_benchmark_passed(
    spec: BenchmarkSpec,
    results: list[ScenarioResult],
    capabilities: dict[str, str],
) -> bool:
    return all(_result_passed(spec, result) for result in results) and all(
        status != "fail" for status in capabilities.values()
    )


def _result_passed(spec: BenchmarkSpec, result: ScenarioResult) -> bool:
    if result.status == "missing" and spec.params.allow_missing_optional_endpoints:
        return True
    if result.expected_success:
        return result.success
    return result.status == "expected_error"


def _capabilities(results: list[ScenarioResult]) -> dict[str, str]:
    caps: dict[str, str] = {}
    for result in results:
        if result.capability:
            caps[result.endpoint] = result.capability
    return caps


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    values_sorted = sorted(values)
    return {
        "mean": statistics.fmean(values_sorted),
        "p50": _percentile(values_sorted, 0.50),
        "p95": _percentile(values_sorted, 0.95),
        "max": max(values_sorted),
    }


def _percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    idx = min(round((len(values) - 1) * pct), len(values) - 1)
    return values[idx]


def _by_category(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for result in results:
        grouped[result.category]["total"] += 1
        grouped[result.category][
            "succeeded" if _result_passed(spec, result) else "failed"
        ] += 1
    return {key: dict(value) for key, value in grouped.items()}
