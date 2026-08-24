# SPDX-License-Identifier: Apache-2.0
"""Summarize the local Qwen3-TTS Phase-A vocoder qualification.

This script deliberately separates:

* artifact/correctness evidence (fixed-code differential and request results),
* mechanical trace evidence (transfer launches and host waits), and
* range timing (descriptive only; not an end-to-end speedup claim).

Run ``analyze_cuda_sync_trace.py`` first for each trace.  The optional baseline
is a mechanism reference only when its model/revision differs from the
candidate; the report never treats such a pair as a serving A/B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_cuda_sync_trace import iter_trace_events

_CANDIDATE_RANGES = (
    "qwen3_tts.vocoder.direct_decode.total",
    "qwen3_tts.vocoder.direct_decode",
    "qwen3_tts.vocoder.codes.h2d",
    "qwen3_tts.vocoder.waveform.publish",
)
_BASELINE_RANGES = ("qwen3_tts.vocoder.tokenizer.decode",)
_WAIT_APIS = {
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
}
_LIFETIME_APIS = {"cudaEventQuery", "cudaEventRecord"}
_PACKED_SEQUENCE_FINGERPRINT = (
    "aten::diff",
    "aten::ne",
    "aten::cumsum",
    "aten::eq",
    "aten::all",
)
_SELECTED_CPU_OPS = {
    "aten::_local_scalar_dense",
    "aten::copy_",
    "aten::is_nonzero",
    "aten::item",
    *_PACKED_SEQUENCE_FINGERPRINT,
}
_CAPACITY_RE = re.compile(
    r"Qwen3-TTS non-streaming code staging capacity: (?P<bytes>\d+) bytes"
)
_NORMALIZE_KEY_RE = re.compile(r"[^a-z0-9]")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FIXED_CODE_FRAME_LENGTHS = {
    1: [8],
    2: [8, 10],
    8: [8, 10, 12, 14, 16, 18, 20, 22],
}


def _normalized_args(event: dict[str, Any]) -> dict[str, Any]:
    args = event.get("args")
    if not isinstance(args, dict):
        return {}
    return {
        _NORMALIZE_KEY_RE.sub("", str(key).lower()): value
        for key, value in args.items()
    }


def _correlation_id(event: dict[str, Any]) -> str | None:
    args = _normalized_args(event)
    for key in ("correlation", "correlationid", "linkedcorrelationid"):
        value = args.get(key)
        if value not in (None, "", 0, "0"):
            return str(value)
    return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _timing(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "total_us": sum(values),
        "mean_us": sum(values) / len(values) if values else None,
        "p50_us": _percentile(values, 0.50),
        "p95_us": _percentile(values, 0.95),
        "max_us": max(values) if values else None,
    }


def _is_complete(event: dict[str, Any]) -> bool:
    return (
        event.get("ph") == "X"
        and isinstance(event.get("ts"), (int, float))
        and isinstance(event.get("dur", 0), (int, float))
    )


def _scan_trace(trace: Path, expected_ranges: tuple[str, ...]) -> dict[str, Any]:
    range_set = set(expected_ranges)
    ranges_by_thread: dict[tuple[Any, Any], list[tuple[str, float, float, float]]] = (
        defaultdict(list)
    )
    selected_events_by_thread: dict[
        tuple[Any, Any], list[tuple[str, float, float, str | None]]
    ] = defaultdict(list)
    gpu_copy_direction: dict[str, str] = {}
    gpu_copy_bytes: dict[str, int] = {}
    trace_aten_events = 0

    for event in iter_trace_events(trace):
        if not _is_complete(event):
            continue
        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        start_us = float(event["ts"])
        duration_us = max(float(event.get("dur", 0.0)), 0.0)
        end_us = start_us + duration_us
        thread = (event.get("pid"), event.get("tid"))

        if category == "user_annotation" and name in range_set:
            ranges_by_thread[thread].append((name, start_us, end_us, duration_us))
            continue
        if name.startswith("aten::"):
            trace_aten_events += 1
        if name in _WAIT_APIS | _LIFETIME_APIS | _SELECTED_CPU_OPS | {
            "cudaMemcpyAsync"
        }:
            selected_events_by_thread[thread].append(
                (name, start_us, end_us, _correlation_id(event))
            )

        correlation_id = _correlation_id(event)
        if correlation_id is None or "memcpy" not in name.lower():
            continue
        lowered = name.lower()
        if "htod" in lowered or "host -> device" in lowered:
            gpu_copy_direction[correlation_id] = "HtoD"
        elif "dtoh" in lowered or "device -> host" in lowered:
            gpu_copy_direction[correlation_id] = "DtoH"
        else:
            continue
        args = _normalized_args(event)
        for key in ("bytes", "size", "numbytes"):
            value = args.get(key)
            if isinstance(value, (int, float)):
                gpu_copy_bytes[correlation_id] = int(value)
                break

    range_durations: dict[str, list[float]] = defaultdict(list)
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    transfer_bytes: dict[str, Counter[str]] = defaultdict(Counter)
    instances_by_thread: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for thread, thread_ranges in ranges_by_thread.items():
        for name, start_us, end_us, duration_us in thread_ranges:
            range_durations[name].append(duration_us)
            instances_by_thread[thread].append(
                {
                    "name": name,
                    "start_us": start_us,
                    "end_us": end_us,
                    "duration_us": duration_us,
                    "events": Counter(),
                    "transfer_bytes": Counter(),
                }
            )

    for thread, events in selected_events_by_thread.items():
        thread_instances = instances_by_thread.get(thread, ())
        for event_name, start_us, end_us, correlation_id in events:
            for instance in thread_instances:
                if instance["start_us"] <= start_us and end_us <= instance["end_us"]:
                    range_name = instance["name"]
                    counters[range_name][event_name] += 1
                    instance["events"][event_name] += 1
                    if event_name == "cudaMemcpyAsync" and correlation_id:
                        direction = gpu_copy_direction.get(correlation_id, "unknown")
                        counters[range_name][f"cudaMemcpyAsync.{direction}"] += 1
                        transfer_bytes[range_name][direction] += gpu_copy_bytes.get(
                            correlation_id, 0
                        )
                        instance["events"][f"cudaMemcpyAsync.{direction}"] += 1
                        instance["transfer_bytes"][direction] += gpu_copy_bytes.get(
                            correlation_id, 0
                        )

    return {
        "trace": str(trace.resolve()),
        "trace_aten_events": trace_aten_events,
        "ranges": {
            name: {
                "timing": _timing(range_durations[name]),
                "events": dict(sorted(counters[name].items())),
                "transfer_bytes": dict(sorted(transfer_bytes[name].items())),
                "instances": sorted(
                    (
                        {
                            "start_us": instance["start_us"],
                            "end_us": instance["end_us"],
                            "duration_us": instance["duration_us"],
                            "events": dict(sorted(instance["events"].items())),
                            "transfer_bytes": dict(
                                sorted(instance["transfer_bytes"].items())
                            ),
                        }
                        for thread_instances in instances_by_thread.values()
                        for instance in thread_instances
                        if instance["name"] == name
                    ),
                    key=lambda instance: instance["start_us"],
                ),
            }
            for name in expected_ranges
        },
    }


def _load_occurrences(analysis_dir: Path) -> list[dict[str, Any]]:
    path = analysis_dir / "cuda_sync_occurrences.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["occurrences"])


def _occurrence_groups(
    occurrences: list[dict[str, Any]], range_prefix: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in occurrences:
        semantic_range = str(row.get("semantic_range") or "unscoped")
        if not semantic_range.startswith(range_prefix):
            continue
        sync = row.get("sync") or {}
        transfer = row.get("transfer") or {}
        key = (
            semantic_range,
            str(sync.get("name") or "unknown"),
            str(row.get("parent_cpu_op") or "direct_wait"),
            str(transfer.get("direction") or "none"),
            str(
                row.get("mechanism")
                or (row.get("blocking_copy") or {}).get("mechanism")
                or "direct_sync"
            ),
        )
        groups[key].append(row)

    result = []
    for key, rows in groups.items():
        result.append(
            {
                "semantic_range": key[0],
                "sync_name": key[1],
                "parent_cpu_op": key[2],
                "transfer_direction": key[3],
                "mechanism": key[4],
                "count": len(rows),
                "blocking_copy_count": sum(
                    row.get("blocking_copy") is not None for row in rows
                ),
                "sync_wait_total_us": sum(
                    float((row.get("metrics") or {}).get("sync_wait_us") or 0.0)
                    for row in rows
                ),
                "compound_host_block_total_us": sum(
                    float(
                        (row.get("blocking_copy") or {}).get("compound_host_block_us")
                        or 0.0
                    )
                    for row in rows
                ),
                "transfer_bytes": sum(
                    int((row.get("transfer") or {}).get("bytes") or 0) for row in rows
                ),
            }
        )
    result.sort(
        key=lambda row: (
            -row["compound_host_block_total_us"],
            -row["sync_wait_total_us"],
            row["semantic_range"],
        )
    )
    return result


def _validate_fixed_code_result(result: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(result, dict):
        return ["result is not a JSON object"]
    if result.get("schema_version") != 1:
        errors.append("schema_version is not 1")
    if result.get("status") != "pass":
        errors.append("status is not pass")
    if result.get("seed") != 20260824:
        errors.append("seed is not 20260824")
    for key in ("torch_version", "qwen_tts_version"):
        if not isinstance(result.get(key), str) or not result[key]:
            errors.append(f"{key} is missing")

    records = result.get("records")
    if not isinstance(records, list):
        return [*errors, "records is not a list"]
    batch_sizes = [
        record.get("batch_size") if isinstance(record, dict) else None
        for record in records
    ]
    if batch_sizes != [1, 2, 8]:
        errors.append("batch sizes are not exactly [1, 2, 8]")
        return errors

    for record in records:
        batch_size = record["batch_size"]
        frame_lengths = record.get("frame_lengths")
        sample_lengths = record.get("sample_lengths")
        digests = record.get("sha256")
        expected_frame_lengths = _FIXED_CODE_FRAME_LENGTHS[batch_size]
        if frame_lengths != expected_frame_lengths:
            errors.append(f"B={batch_size} frame lengths do not match the probe")
        if not isinstance(sample_lengths, list) or len(sample_lengths) != batch_size:
            errors.append(f"B={batch_size} sample lengths are malformed")
        elif sample_lengths != [
            frame_length * 1920 for frame_length in expected_frame_lengths
        ]:
            errors.append(f"B={batch_size} sample lengths violate 1,920x expansion")
        if not isinstance(digests, list) or len(digests) != batch_size:
            errors.append(f"B={batch_size} SHA-256 list is malformed")
        elif not all(
            isinstance(digest, str) and _SHA256_RE.fullmatch(digest)
            for digest in digests
        ):
            errors.append(f"B={batch_size} contains an invalid SHA-256 digest")
    return errors


def _artifact_summary(root: Path) -> dict[str, Any]:
    run = root / "q3tts-vocoder-phase-a-c16"
    fixed_log = (root / "fixed_code_differential.log").read_text(
        encoding="utf-8", errors="replace"
    )
    fixed_result_path = root / "fixed_code_differential.json"
    fixed_result = (
        json.loads(fixed_result_path.read_text(encoding="utf-8"))
        if fixed_result_path.exists()
        else None
    )
    fixed_records = (
        list(fixed_result.get("records") or [])
        if isinstance(fixed_result, dict)
        else []
    )
    fixed_validation_errors = (
        _validate_fixed_code_result(fixed_result)
        if fixed_result_path.exists()
        else ["fixed_code_differential.json is missing"]
    )
    fixed_json_pass = not fixed_validation_errors
    speed = json.loads((run / "client" / "speed_results.json").read_text())
    warmup = json.loads((root / "warmup" / "speed_results.json").read_text())
    server_log = (root / "server.log").read_text(encoding="utf-8", errors="replace")
    capacities = [
        int(match.group("bytes")) for match in _CAPACITY_RE.finditer(server_log)
    ]

    def wavs(path: Path) -> dict[str, Any]:
        rows = []
        for wav_path in sorted(path.glob("*.wav")):
            with wave.open(str(wav_path), "rb") as stream:
                rows.append(
                    (
                        stream.getframerate(),
                        stream.getnchannels(),
                        stream.getsampwidth(),
                        stream.getnframes(),
                    )
                )
        return {
            "count": len(rows),
            "sample_rates": sorted({row[0] for row in rows}),
            "channels": sorted({row[1] for row in rows}),
            "sample_widths": sorted({row[2] for row in rows}),
            "min_frames": min((row[3] for row in rows), default=None),
            "max_frames": max((row[3] for row in rows), default=None),
        }

    per_request = speed["per_request"]
    return {
        "root": str(root.resolve()),
        "fixed_code": {
            "pass_marker": (
                fixed_json_pass
                if fixed_result_path.exists()
                else "fixed-code official/direct differential: PASS" in fixed_log
            ),
            "batch_records": (
                len(fixed_records)
                if fixed_records
                else fixed_log.count("'batch_size':")
            ),
            "result_json_present": fixed_result_path.exists(),
            "result_json_sha256": (
                hashlib.sha256(fixed_result_path.read_bytes()).hexdigest()
                if fixed_result_path.exists()
                else None
            ),
            "validation_errors": fixed_validation_errors,
            "seed": (
                fixed_result.get("seed") if isinstance(fixed_result, dict) else None
            ),
            "torch_version": (
                fixed_result.get("torch_version")
                if isinstance(fixed_result, dict)
                else None
            ),
            "qwen_tts_version": (
                fixed_result.get("qwen_tts_version")
                if isinstance(fixed_result, dict)
                else None
            ),
            "records": fixed_records,
            "log_bytes": len(fixed_log.encode()),
        },
        "warmup_summary": warmup["summary"],
        "candidate_summary": speed["summary"],
        "candidate_config": speed["config"],
        "candidate_computed": {
            "rows": len(per_request),
            "successes": sum(bool(row["is_success"]) for row in per_request),
            "failures": sum(not bool(row["is_success"]) for row in per_request),
            "token_caps": sum(
                int(row.get("completion_tokens") or 0) == 2048 for row in per_request
            ),
        },
        "staging_capacity_bytes": capacities,
        "staging_capacity_max_bytes": max(capacities, default=None),
        "measured_wavs": wavs(run / "client" / "audio"),
        "warmup_wavs": wavs(root / "warmup" / "audio"),
        "server_error_scan_bytes": (run / "server_error_scan.log").stat().st_size,
    }


def _candidate_batch_summary(trace_summary: dict[str, Any]) -> dict[str, Any]:
    ranges = trace_summary["ranges"]
    totals = ranges["qwen3_tts.vocoder.direct_decode.total"]["instances"]
    directs = ranges["qwen3_tts.vocoder.direct_decode"]["instances"]
    copies = ranges["qwen3_tts.vocoder.codes.h2d"]["instances"]
    publications = ranges["qwen3_tts.vocoder.waveform.publish"]["instances"]
    rows = []
    for index, total in enumerate(totals):
        window_end = (
            totals[index + 1]["start_us"] if index + 1 < len(totals) else math.inf
        )
        nested_directs = [
            row
            for row in directs
            if total["start_us"] <= row["start_us"] and row["end_us"] <= total["end_us"]
        ]
        nested_copies = [
            row
            for row in copies
            if total["start_us"] <= row["start_us"] and row["end_us"] <= total["end_us"]
        ]
        batch_publications = [
            row
            for row in publications
            if total["end_us"] <= row["start_us"] < window_end
        ]
        direct = nested_directs[0] if len(nested_directs) == 1 else None
        copy = nested_copies[0] if len(nested_copies) == 1 else None
        rows.append(
            {
                "batch_index": index,
                "batch_size": len(batch_publications),
                "direct_instance_count": len(nested_directs),
                "copy_instance_count": len(nested_copies),
                "direct_total_us": total["duration_us"],
                "decode_us": direct["duration_us"] if direct else None,
                "scalar_reads": (
                    direct["events"].get("aten::_local_scalar_dense", 0)
                    if direct
                    else None
                ),
                "codes_h2d_bytes": (
                    copy["transfer_bytes"].get("HtoD", 0) if copy else None
                ),
                "publication_total_us": sum(
                    row["duration_us"] for row in batch_publications
                ),
                "waveform_d2h_bytes": sum(
                    row["transfer_bytes"].get("DtoH", 0) for row in batch_publications
                ),
            }
        )

    grouped: dict[int, dict[str, Any]] = {}
    for batch_size in sorted({row["batch_size"] for row in rows}):
        selected = [row for row in rows if row["batch_size"] == batch_size]
        grouped[batch_size] = {
            "batches": len(selected),
            "requests": sum(row["batch_size"] for row in selected),
            "direct_total": _timing([row["direct_total_us"] for row in selected]),
            "publication_total": _timing(
                [row["publication_total_us"] for row in selected]
            ),
            "codes_h2d_bytes": sum(row["codes_h2d_bytes"] or 0 for row in selected),
            "waveform_d2h_bytes": sum(row["waveform_d2h_bytes"] for row in selected),
        }
    return {
        "rows": rows,
        "batch_size_distribution": {
            str(batch_size): count
            for batch_size, count in sorted(
                Counter(row["batch_size"] for row in rows).items()
            )
        },
        "accounted_requests": sum(row["batch_size"] for row in rows),
        "grouped": grouped,
    }


def _build_gate(
    artifact: dict[str, Any],
    trace_summary: dict[str, Any],
    groups: list[dict[str, Any]],
    batch_summary: dict[str, Any],
) -> dict[str, Any]:
    ranges = trace_summary["ranges"]
    direct = ranges["qwen3_tts.vocoder.direct_decode"]
    h2d = ranges["qwen3_tts.vocoder.codes.h2d"]
    publish = ranges["qwen3_tts.vocoder.waveform.publish"]
    selected_groups = [
        row
        for row in groups
        if row["semantic_range"]
        in {
            "qwen3_tts.vocoder.direct_decode",
            "qwen3_tts.vocoder.codes.h2d",
        }
    ]
    fingerprint_counts = {
        name: direct["events"].get(name, 0) for name in _PACKED_SEQUENCE_FINGERPRINT
    }
    packed_checks = fingerprint_counts["aten::diff"]
    scalar_reads = direct["events"].get("aten::_local_scalar_dense", 0)
    direct_waits = sum(direct["events"].get(name, 0) for name in _WAIT_APIS)
    direct_blocking_copies = sum(row["blocking_copy_count"] for row in selected_groups)
    largest_codes_h2d = max(
        (row["codes_h2d_bytes"] or 0 for row in batch_summary["rows"]),
        default=0,
    )
    common_checks = {
        "fixed_code_parity_proven": artifact["fixed_code"]["pass_marker"]
        and artifact["fixed_code"]["batch_records"] == 3,
        "all_ranges_exercised": all(
            ranges[name]["timing"]["count"] > 0 for name in _CANDIDATE_RANGES
        ),
        "trace_aten_present": trace_summary["trace_aten_events"] > 0,
        "codes_h2d_exercised": h2d["events"].get("cudaMemcpyAsync.HtoD", 0) > 0,
        "codes_h2d_has_no_wait": sum(h2d["events"].get(name, 0) for name in _WAIT_APIS)
        == 0,
        "waveform_d2h_retained": publish["events"].get("cudaMemcpyAsync.DtoH", 0) > 0,
        "requests_complete": artifact["candidate_computed"]
        == {"rows": 64, "successes": 64, "failures": 0, "token_caps": 0},
        "audio_contract": artifact["measured_wavs"]["count"] == 64
        and artifact["measured_wavs"]["sample_rates"] == [24000]
        and artifact["measured_wavs"]["channels"] == [1]
        and artifact["measured_wavs"]["sample_widths"] == [2],
        "batch_accounting_complete": batch_summary["accounted_requests"] == 64
        and all(
            row["direct_instance_count"] == 1 and row["copy_instance_count"] == 1
            for row in batch_summary["rows"]
        ),
        "vocoder_batch_bound_respected": all(
            1 <= row["batch_size"] <= 8 for row in batch_summary["rows"]
        ),
        "staging_highwater_matches_largest_h2d": artifact["staging_capacity_max_bytes"]
        == largest_codes_h2d,
    }
    diagnostic_checks = {
        "packed_sequence_fingerprint_is_complete": len(set(fingerprint_counts.values()))
        == 1,
        "no_scalar_reads_beyond_packed_sequence_fingerprint": scalar_reads
        == packed_checks,
        "direct_decode_has_no_wait": direct_waits == 0,
        "direct_decode_has_no_scalar_read": scalar_reads == 0,
        "direct_ranges_have_no_blocking_copy": direct_blocking_copies == 0,
    }
    checks = common_checks | diagnostic_checks
    narrow_keys = [
        "all_ranges_exercised",
        "trace_aten_present",
        "codes_h2d_exercised",
        "codes_h2d_has_no_wait",
        "waveform_d2h_retained",
        "requests_complete",
        "audio_contract",
        "batch_accounting_complete",
        "vocoder_batch_bound_respected",
        "staging_highwater_matches_largest_h2d",
        "packed_sequence_fingerprint_is_complete",
        "no_scalar_reads_beyond_packed_sequence_fingerprint",
    ]
    narrow_mechanics_passed = all(checks[key] for key in narrow_keys)
    planned_exit_keys = [
        *narrow_keys,
        "direct_decode_has_no_wait",
        "direct_decode_has_no_scalar_read",
        "direct_ranges_have_no_blocking_copy",
    ]
    planned_trace_exit_passed = all(checks[key] for key in planned_exit_keys)
    return {
        "checks": checks,
        "packed_sequence_fingerprint_counts": fingerprint_counts,
        "direct_scalar_reads": scalar_reads,
        "direct_waits": direct_waits,
        "direct_blocking_copies": direct_blocking_copies,
        "largest_codes_h2d_bytes": largest_codes_h2d,
        "narrow_mechanics_passed": narrow_mechanics_passed,
        "planned_trace_exit_passed": planned_trace_exit_passed,
        "correctness_evidence_complete": common_checks["fixed_code_parity_proven"],
        "qualification_passed": common_checks["fixed_code_parity_proven"]
        and planned_trace_exit_passed,
    }


def _fmt_ms(value_us: float | int | None) -> str:
    return "-" if value_us is None else f"{float(value_us) / 1000.0:.3f}"


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    gate = summary["gate"]
    artifact = summary["artifact"]
    qualification_status = (
        "PASS"
        if gate["qualification_passed"]
        else "FAIL" if gate["correctness_evidence_complete"] else "INCOMPLETE"
    )
    lines = [
        "# Qwen3-TTS vocoder Phase-A summary",
        "",
        f"Narrow wrapper-mechanics gate: **{'PASS' if gate['narrow_mechanics_passed'] else 'FAIL'}**",
        f"Planned sync-free direct-range gate: **{'PASS' if gate['planned_trace_exit_passed'] else 'FAIL'}**",
        f"Correctness evidence: **{'COMPLETE' if gate['correctness_evidence_complete'] else 'INCOMPLETE'}**",
        f"Overall qualification: **{qualification_status}**",
        "",
        "## Gate checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, passed in gate["checks"].items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Candidate ranges",
            "",
            "| Range | Calls | Total | P50 | P95 | H2D | D2H | Waits | Scalar reads |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["candidate_trace"]["ranges"].items():
        timing = row["timing"]
        events = row["events"]
        lines.append(
            f"| `{name}` | {timing['count']} | {_fmt_ms(timing['total_us'])} ms | "
            f"{_fmt_ms(timing['p50_us'])} ms | {_fmt_ms(timing['p95_us'])} ms | "
            f"{events.get('cudaMemcpyAsync.HtoD', 0)} | "
            f"{events.get('cudaMemcpyAsync.DtoH', 0)} | "
            f"{sum(events.get(wait, 0) for wait in _WAIT_APIS)} | "
            f"{events.get('aten::_local_scalar_dense', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Candidate batch accounting",
            "",
            f"Batch-size distribution: `{summary['candidate_batches']['batch_size_distribution']}`; "
            f"accounted requests: {summary['candidate_batches']['accounted_requests']}.",
            "",
            "| Batch size | Batches | Requests | Direct mean | Direct p95 | Publish mean | H2D bytes | D2H bytes |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for batch_size, row in summary["candidate_batches"]["grouped"].items():
        lines.append(
            f"| {batch_size} | {row['batches']} | {row['requests']} | "
            f"{_fmt_ms(row['direct_total']['mean_us'])} ms | "
            f"{_fmt_ms(row['direct_total']['p95_us'])} ms | "
            f"{_fmt_ms(row['publication_total']['mean_us'])} ms | "
            f"{row['codes_h2d_bytes']} | {row['waveform_d2h_bytes']} |"
        )
    lines.extend(
        [
            "",
            "## Scalar synchronization accounting",
            "",
            f"- Direct-range scalar reads: {gate['direct_scalar_reads']}.",
            f"- Complete `diff -> ne -> cumsum -> eq -> all` fingerprints: "
            f"{gate['packed_sequence_fingerprint_counts']['aten::diff']}.",
            "- This fingerprint matches Transformers' dynamic packed-sequence "
            "check. Source attribution is an inference from the exact operator "
            "sequence because Python stacks were not captured.",
            "- Scalar reads beyond that fingerprint: "
            f"{gate['direct_scalar_reads'] - gate['packed_sequence_fingerprint_counts']['aten::diff']}.",
            "",
            "## Artifact contract",
            "",
            f"- Fixed-code PASS marker: `{artifact['fixed_code']['pass_marker']}`; "
            f"batch records: {artifact['fixed_code']['batch_records']}.",
            f"- Fixed-code artifact SHA-256: "
            f"`{artifact['fixed_code']['result_json_sha256']}`; validation errors: "
            f"`{artifact['fixed_code']['validation_errors']}`.",
            f"- Fixed-code environment: Torch "
            f"`{artifact['fixed_code']['torch_version']}`, qwen-tts "
            f"`{artifact['fixed_code']['qwen_tts_version']}`, seed "
            f"`{artifact['fixed_code']['seed']}`.",
            f"- Measured requests: {artifact['candidate_computed']}.",
            f"- Maximum logged pinned code staging: "
            f"{artifact['staging_capacity_max_bytes']} bytes.",
            f"- Largest correlated code H2D: {gate['largest_codes_h2d_bytes']} bytes.",
            f"- Client summary: `{artifact['candidate_summary']}`.",
            "",
            "## Interpretation boundary",
            "",
            "Range timings are profiler-perturbed descriptive measurements. "
            "If a baseline is present but uses a different model or code revision, "
            "it is a mechanism reference only and not an end-to-end A/B.",
            "",
        ]
    )
    if summary["baseline_trace"] is not None:
        baseline = summary["baseline_trace"]["ranges"][
            "qwen3_tts.vocoder.tokenizer.decode"
        ]
        baseline_events = baseline["events"]
        baseline_scalars = baseline_events.get("aten::_local_scalar_dense", 0)
        baseline_fingerprints = baseline_events.get("aten::diff", 0)
        lines.extend(
            [
                "## Baseline mechanism reference",
                "",
                f"- Wrapper calls: {baseline['timing']['count']}.",
                f"- Scalar reads: {baseline_scalars}; packed-sequence "
                f"fingerprints: {baseline_fingerprints}; residual wrapper "
                f"reads: {baseline_scalars - baseline_fingerprints}.",
                "- The baseline uses a different serving run/model revision. "
                "This decomposition is a mechanism reference, not a timing A/B.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("candidate_trace", type=Path)
    parser.add_argument("--candidate-analysis-dir", required=True, type=Path)
    parser.add_argument("--baseline-trace", type=Path)
    parser.add_argument("--baseline-analysis-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if (args.baseline_trace is None) != (args.baseline_analysis_dir is None):
        raise SystemExit(
            "--baseline-trace and --baseline-analysis-dir must be provided together"
        )

    artifact = _artifact_summary(args.artifact_root)
    candidate_trace = _scan_trace(args.candidate_trace, _CANDIDATE_RANGES)
    candidate_batches = _candidate_batch_summary(candidate_trace)
    candidate_groups = _occurrence_groups(
        _load_occurrences(args.candidate_analysis_dir), "qwen3_tts.vocoder."
    )
    baseline_trace = None
    baseline_groups = None
    if args.baseline_trace is not None:
        baseline_trace = _scan_trace(args.baseline_trace, _BASELINE_RANGES)
        baseline_groups = _occurrence_groups(
            _load_occurrences(args.baseline_analysis_dir),
            "qwen3_tts.vocoder.",
        )

    summary = {
        "schema_version": 1,
        "artifact": artifact,
        "candidate_trace": candidate_trace,
        "candidate_batches": candidate_batches,
        "candidate_sync_groups": candidate_groups,
        "baseline_trace": baseline_trace,
        "baseline_sync_groups": baseline_groups,
    }
    summary["gate"] = _build_gate(
        artifact,
        candidate_trace,
        candidate_groups,
        candidate_batches,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "qwen3_tts_vocoder_phase_a_summary.json"
    markdown_path = args.output_dir / "qwen3_tts_vocoder_phase_a_summary.md"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(summary, markdown_path)
    print(
        json.dumps(
            {
                "narrow_mechanics_passed": summary["gate"]["narrow_mechanics_passed"],
                "planned_trace_exit_passed": summary["gate"][
                    "planned_trace_exit_passed"
                ],
                "correctness_evidence_complete": summary["gate"][
                    "correctness_evidence_complete"
                ],
                "qualification_passed": summary["gate"]["qualification_passed"],
                "summary_json": str(json_path.resolve()),
                "summary_markdown": str(markdown_path.resolve()),
            },
            sort_keys=True,
        )
    )
    return 1 if args.strict and not summary["gate"]["qualification_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
