# SPDX-License-Identifier: Apache-2.0
"""Report aggregation for the TTS serving benchmark."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from benchmarks.tts_serving.metrics import ScenarioResult
from benchmarks.tts_serving.scenarios import (
    SCENARIO_SCHEMA_VERSION,
    VOICE_UPLOAD_SUCCESS_FORMATS,
    Scenario,
    scenario_set_hash,
)
from benchmarks.tts_serving.spec import BenchmarkSpec


def build_results_report(
    spec: BenchmarkSpec,
    results: list[ScenarioResult],
    *,
    scenarios: list[Scenario] | None = None,
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
    operation_capabilities = _operation_capabilities(results)
    capabilities = _endpoint_capabilities(results, operation_capabilities)
    category_counts = Counter(result.category for result in results)
    status_counts = Counter(result.status for result in results)
    latencies = [r.latency_s for r in results if r.latency_s > 0]
    ttfas = [r.ttfa_s for r in results if r.ttfa_s is not None]
    rtfs = [r.rtf for r in results if r.rtf > 0]
    queue_waits = [r.queue_wait_s for r in results if r.queue_wait_s is not None]
    generator_lags = [
        r.generator_lag_s for r in results if r.generator_lag_s is not None
    ]
    load_generation_valid = not any(result.load_generator_lagged for result in results)
    passed = (
        harness_status == "ok"
        and load_generation_valid
        and _is_benchmark_passed(spec, results, capabilities)
    )
    return {
        "schema_version": 2,
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
        "scenario_set_hash": scenario_set_hash(scenarios) if scenarios else None,
        "harness_status": harness_status,
        "harness_error": harness_error,
        "overall": {
            "passed": passed,
            "total": total,
            "traffic_total": traffic_total,
            "capability_probe_total": capability_total,
            "succeeded": succeeded,
            "failed": failed,
            "load_generation_valid": load_generation_valid,
        },
        "config": {
            "test_type": spec.test_type,
            "base_url": spec.base_url,
            "model_name": spec.model_name,
            "run_id": spec.run_id,
            "seed": spec.seed,
            "provider_label": spec.params.provider_label,
            "implementation_label": spec.params.implementation_label,
            "profile": spec.params.profile,
            "total_requests": spec.params.total_requests,
            "stage_request_total": sum(
                stage.request_count for stage in spec.params.load_stages
            ),
            "max_concurrency": spec.params.max_concurrency,
            "concurrency_levels": list(
                spec.params.concurrency_levels or (spec.params.max_concurrency,)
            ),
            "request_rate": (
                "inf"
                if spec.params.request_rate == float("inf")
                else spec.params.request_rate
            ),
            "timeout_s": spec.params.timeout_s,
            "enabled_endpoints": list(spec.params.enabled_endpoints),
            "voice_cache_eviction_count": spec.params.voice_cache_eviction_count,
            "voice_speaker_cap_count": spec.params.voice_speaker_cap_count,
            "voice_upload_coverage": _voice_upload_coverage(scenarios or []),
            "load_stages": [stage.to_json() for stage in spec.params.load_stages],
        },
        "capabilities": capabilities,
        "operation_capabilities": operation_capabilities,
        "metrics": {
            "latency_s": _summary(latencies),
            "ttfa_s": _summary(ttfas),
            "queue_wait_s": _summary(queue_waits),
            "generator_lag_s": _summary(generator_lags),
            "peak_inflight": max(
                (result.peak_inflight for result in results if result.peak_inflight),
                default=None,
            ),
            "peak_pending_tasks": max(
                (
                    result.peak_pending_tasks
                    for result in results
                    if result.peak_pending_tasks
                ),
                default=None,
            ),
            "load_generator_lagged": any(
                result.load_generator_lagged for result in results
            ),
            "load_generation_error": (
                "scheduled arrivals lagged beyond benchmark threshold"
                if not load_generation_valid
                else None
            ),
            "rtf": _summary(rtfs),
            "rtf_sampled_formats": ["wav", "pcm"],
            "rtf_unsupported_format_counts": _rtf_unsupported_format_counts(results),
            "rtf_note": (
                "RTF is computed only when audio duration can be derived from WAV "
                "headers or raw PCM byte counts; compressed responses are excluded."
            ),
            "status_counts": dict(status_counts),
            "http_status_counts": dict(
                Counter(
                    str(result.http_status)
                    for result in results
                    if result.http_status is not None
                )
            ),
            "admission_status_counts": _admission_status_counts(results),
            "error_class_counts": dict(
                Counter(result.error_class for result in results if result.error_class)
            ),
            "category_counts": dict(category_counts),
            "endpoint_mix": dict(Counter(result.endpoint for result in results)),
            "by_category": _by_category(spec, results),
            "by_stage": _by_stage(spec, results),
            "by_endpoint": _by_endpoint(spec, results),
            "by_operation": _by_operation(spec, results),
            "by_configured_concurrency": _by_configured_concurrency(spec, results),
            "by_peak_inflight": _by_peak_inflight(spec, results),
        },
        "failures": [
            result.to_json() for result in results if not _result_passed(spec, result)
        ][:100],
        "unsupported_contracts": [
            result.to_json()
            for result in results
            if result.status == "unsupported_contract"
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
    if result.expected_success:
        return result.success
    return result.status == "expected_error"


def _operation_capabilities(results: list[ScenarioResult]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for result in results:
        if result.capability:
            key = result.capability_key or result.endpoint
            grouped[key].append(result.capability)
    return {
        key: _roll_up_capability(statuses) for key, statuses in sorted(grouped.items())
    }


def _endpoint_capabilities(
    results: list[ScenarioResult],
    operation_capabilities: dict[str, str],
) -> dict[str, str]:
    endpoint_operations: dict[str, set[str]] = defaultdict(set)
    for result in results:
        key = result.capability_key or result.endpoint
        if key in operation_capabilities:
            endpoint_operations[result.endpoint].add(key)
    return {
        endpoint: _roll_up_endpoint_capability(
            [operation_capabilities[key] for key in sorted(keys)]
        )
        for endpoint, keys in sorted(endpoint_operations.items())
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "p99_9": None,
            "max": None,
        }
    values_sorted = sorted(values)
    return {
        "mean": statistics.fmean(values_sorted),
        "p50": _percentile(values_sorted, 0.50),
        "p95": _percentile(values_sorted, 0.95),
        "p99": _percentile(values_sorted, 0.99),
        "p99_9": _percentile(values_sorted, 0.999),
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


def _by_stage(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[result.stage_id or "unknown"].append(result)
    return {
        stage_id: _result_group_summary(spec, stage_results)
        for stage_id, stage_results in sorted(grouped.items())
    }


def _by_endpoint(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[result.endpoint].append(result)
    return {
        endpoint: _result_group_summary(spec, endpoint_results)
        for endpoint, endpoint_results in sorted(grouped.items())
    }


def _by_operation(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        grouped[result.capability_key or result.endpoint].append(result)
    return {
        operation: _result_group_summary(spec, operation_results)
        for operation, operation_results in sorted(grouped.items())
    }


def _by_configured_concurrency(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        if result.load_concurrency is not None:
            grouped[result.load_concurrency].append(result)

    summaries: dict[str, dict[str, Any]] = {}
    for level, level_results in sorted(grouped.items()):
        summaries[str(level)] = _result_group_summary(spec, level_results)
    return summaries


def _by_peak_inflight(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, dict[str, Any]]:
    grouped: dict[int, list[ScenarioResult]] = defaultdict(list)
    for result in results:
        if result.peak_inflight is not None:
            grouped[result.peak_inflight].append(result)

    summaries: dict[str, dict[str, Any]] = {}
    for level, level_results in sorted(grouped.items()):
        summaries[str(level)] = _result_group_summary(spec, level_results)
    return summaries


def _voice_upload_coverage(scenarios: list[Scenario]) -> dict[str, Any]:
    successful_formats = {
        str(scenario.planned_metadata.get("upload_format"))
        for scenario in scenarios
        if scenario.capability_key == "voices.upload"
        and scenario.expect_success
        and scenario.planned_metadata.get("upload_case") == "format"
    }
    near_limit_formats = {
        str(scenario.planned_metadata.get("upload_format"))
        for scenario in scenarios
        if scenario.capability_key == "voices.upload"
        and scenario.planned_metadata.get("upload_case") == "near_limit"
    }
    cache_eviction_formats = {
        str(scenario.planned_metadata.get("upload_format"))
        for scenario in scenarios
        if scenario.capability_key == "voices.upload"
        and scenario.planned_metadata.get("upload_case") == "cache_eviction"
    }
    speaker_cap_cases = sum(
        1
        for scenario in scenarios
        if scenario.capability_key == "voices.upload"
        and scenario.planned_metadata.get("upload_case") == "speaker_cap"
    )
    configured_formats = [
        upload_format for upload_format, _ in VOICE_UPLOAD_SUCCESS_FORMATS
    ]
    near_limit_missing_formats = sorted(
        set(configured_formats) - set(near_limit_formats)
    )
    return {
        "accepted_format_cases": sorted(successful_formats),
        "configured_accepted_formats": configured_formats,
        "near_limit_formats": sorted(near_limit_formats),
        "near_limit_missing_formats": near_limit_missing_formats,
        "near_limit_contract_complete": not near_limit_missing_formats,
        "cache_eviction_formats": sorted(cache_eviction_formats),
        "speaker_cap_cases": speaker_cap_cases,
        "near_limit_pr1_gap": (
            (
                "Valid just-under-10MB fixtures are currently generated only for WAV; "
                "non-WAV near-limit formats are reported as an explicit coverage gap "
                "instead of using invalid padded containers."
            )
            if near_limit_missing_formats
            else None
        ),
    }


def _result_group_summary(
    spec: BenchmarkSpec, results: list[ScenarioResult]
) -> dict[str, Any]:
    latencies = [result.latency_s for result in results if result.latency_s > 0]
    ttfas = [result.ttfa_s for result in results if result.ttfa_s is not None]
    rtfs = [result.rtf for result in results if result.rtf > 0]
    queue_waits = [
        result.queue_wait_s for result in results if result.queue_wait_s is not None
    ]
    generator_lags = [
        result.generator_lag_s
        for result in results
        if result.generator_lag_s is not None
    ]
    planned_starts = [
        result.planned_start_s
        for result in results
        if result.planned_start_s is not None
    ]
    first_start = min(
        (result.actual_start_s for result in results if result.actual_start_s),
        default=None,
    )
    last_completion = max(
        (result.completed_s for result in results if result.completed_s),
        default=None,
    )
    wall_time_s = (
        last_completion - first_start
        if first_start is not None and last_completion is not None
        else None
    )
    planned_window_s = (
        max(planned_starts) - min(planned_starts) if len(planned_starts) > 1 else None
    )
    succeeded = sum(1 for result in results if _result_passed(spec, result))
    failed = len(results) - succeeded
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "wall_time_s": wall_time_s,
        "planned_window_s": planned_window_s,
        "configured_max_concurrency": sorted(
            {
                result.configured_max_concurrency
                for result in results
                if result.configured_max_concurrency is not None
            }
        ),
        "peak_inflight": max(
            (result.peak_inflight for result in results if result.peak_inflight),
            default=None,
        ),
        "peak_pending_tasks": max(
            (
                result.peak_pending_tasks
                for result in results
                if result.peak_pending_tasks
            ),
            default=None,
        ),
        "scheduled_task_count": max(
            (
                result.scheduled_task_count
                for result in results
                if result.scheduled_task_count
            ),
            default=None,
        ),
        "load_generator_lagged": any(
            result.load_generator_lagged for result in results
        ),
        "offered_rps": (
            len(results) / planned_window_s
            if planned_window_s and planned_window_s > 0
            else None
        ),
        "achieved_rps": (
            len(results) / wall_time_s if wall_time_s and wall_time_s > 0 else None
        ),
        "latency_s": _summary(latencies),
        "ttfa_s": _summary(ttfas),
        "queue_wait_s": _summary(queue_waits),
        "generator_lag_s": _summary(generator_lags),
        "rtf": _summary(rtfs),
        "status_counts": dict(Counter(result.status for result in results)),
        "http_status_counts": dict(
            Counter(
                str(result.http_status)
                for result in results
                if result.http_status is not None
            )
        ),
        "admission_status_counts": _admission_status_counts(results),
        "error_class_counts": dict(
            Counter(result.error_class for result in results if result.error_class)
        ),
        "category_counts": dict(Counter(result.category for result in results)),
    }


def _roll_up_endpoint_capability(statuses: list[str]) -> str:
    return _roll_up_capability(statuses)


def _admission_status_counts(results: list[ScenarioResult]) -> dict[str, int]:
    counts = Counter()
    for result in results:
        if result.http_status in {429, 503}:
            counts[str(result.http_status)] += 1
        if result.status == "transport_error":
            counts["transport_error"] += 1
        if result.error_type and "Timeout" in result.error_type:
            counts["timeout"] += 1
    return dict(counts)


def _rtf_unsupported_format_counts(results: list[ScenarioResult]) -> dict[str, int]:
    counts = Counter()
    for result in results:
        response_format = (result.response_format or "").lower()
        if (
            response_format
            and response_format not in {"wav", "pcm"}
            and result.audio_bytes > 0
            and result.rtf == 0
        ):
            counts[response_format] += 1
    return dict(counts)


def _roll_up_capability(statuses: list[str]) -> str:
    if not statuses:
        return "missing"
    if "fail" in statuses:
        return "fail"
    unique = set(statuses)
    if len(unique) == 1:
        return statuses[0]
    return "partial"
