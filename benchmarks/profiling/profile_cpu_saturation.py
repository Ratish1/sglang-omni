# SPDX-License-Identifier: Apache-2.0
"""Direct-server Fun-ASR CPU-saturation profiling harness.

This command assumes the model server is already running.  One invocation is
one server-lifetime trial; repeat the command across fresh server restarts for
the interleaved multi-restart protocol in the accompanying README.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.eval.benchmark_asr_seedtts import run_asr_seedtts_once
from benchmarks.profiling.cpu_interferer import parse_cpu_list
from benchmarks.profiling.system_collectors import (
    CpuFrequencyCollector,
    ThreadSnapshotCollector,
    capture_command,
    collect_static_manifest,
    gpu_dmon_collector,
    parse_cpu_frequency,
    parse_gpu_dmon,
    parse_perf_stat,
    parse_thread_snapshots,
    parse_turbostat,
    perf_sched_collector,
    perf_stat_collector,
    psi_delta,
    read_process_cgroup_psi,
    read_psi,
    read_thread_snapshot,
    summarize_thread_snapshot_delta,
    turbostat_collector,
    write_json,
)
from benchmarks.tasks.asr import FUN_ASR_MODEL_PATH
from sglang_omni.profiler.integrity import (
    validate_request_lifecycle,
    validate_stop_response,
)
from sglang_omni.profiler.views import build_report

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _requested_collectors(args: argparse.Namespace) -> set[str]:
    return {item.strip() for item in args.collectors.split(",") if item.strip()}


def _wav_duration_s(path: str) -> float:
    """Read WAV duration from its header without decoding the audio."""
    with wave.open(path, "rb") as handle:
        frame_rate = handle.getframerate()
        if frame_rate <= 0:
            raise ValueError(f"invalid WAV frame rate in {path}")
        return handle.getnframes() / frame_rate


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_index(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.name == "artifact_index.json"
            or path.name.endswith(".partial")
        ):
            continue
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "root": str(root),
        "generated_wall_ns": time.time_ns(),
        "artifacts": records,
    }


def _dataset_manifest(
    samples: list[Any],
    measurement_samples: list[Any],
    *,
    source: str,
    split: str,
) -> dict[str, Any]:
    measurement_ids = {sample.sample_id for sample in measurement_samples}
    rows: list[dict[str, Any]] = []
    corpus_digest = hashlib.sha256()
    durations: list[float] = []
    for order, sample in enumerate(samples):
        path = Path(sample.ref_audio).resolve()
        duration_s = _wav_duration_s(str(path))
        audio_sha256 = _file_sha256(path)
        row = {
            "order": order,
            "sample_id": sample.sample_id,
            "audio_path": str(path),
            "audio_bytes": path.stat().st_size,
            "audio_sha256": audio_sha256,
            "duration_s": duration_s,
            "measurement": sample.sample_id in measurement_ids,
        }
        rows.append(row)
        durations.append(duration_s)
        corpus_digest.update(
            json.dumps(
                {
                    "order": order,
                    "sample_id": sample.sample_id,
                    "ref_text": sample.ref_text,
                    "target_text": sample.target_text,
                    "audio_sha256": audio_sha256,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
    ordered = sorted(durations)

    def percentile(q: float) -> float | None:
        if not ordered:
            return None
        return ordered[min(int((len(ordered) - 1) * q), len(ordered) - 1)]

    return {
        "source": source,
        "split": split,
        "samples": len(samples),
        "measurement_samples": len(measurement_samples),
        "corpus_sha256": corpus_digest.hexdigest(),
        "duration_s": {
            "min": min(ordered) if ordered else None,
            "p25": percentile(0.25),
            "p50": percentile(0.50),
            "p75": percentile(0.75),
            "p95": percentile(0.95),
            "max": max(ordered) if ordered else None,
        },
        "records": rows,
    }


def _duration_stratified_subset(samples: list[Any], count: int) -> list[Any]:
    """Choose deterministic duration quantiles while retaining corpus order."""
    if count <= 0 or count >= len(samples):
        return list(samples)
    durations: list[tuple[float, int]] = []
    for index, sample in enumerate(samples):
        try:
            duration = _wav_duration_s(sample.ref_audio)
        except (OSError, EOFError, wave.Error, ValueError):
            # The benchmark itself will report unreadable audio.  Preserve a
            # deterministic fallback here so selection never hides that error.
            duration = float(index)
        durations.append((duration, index))
    durations.sort()
    selected = {
        durations[
            min(
                ((2 * position + 1) * len(durations)) // (2 * count),
                len(durations) - 1,
            )
        ][1]
        for position in range(count)
    }
    # Integer quantiles are unique when count < len(samples), but keep a
    # deterministic fill for defensive correctness.
    if len(selected) < count:
        selected.update(index for _, index in durations if index not in selected)
    return [samples[index] for index in sorted(selected)[:count]]


def _relative_spread(values: list[float]) -> float:
    center = statistics.median(values)
    if center == 0:
        return math.inf if max(values) != min(values) else 0.0
    return (max(values) - min(values)) / abs(center)


def _is_stable(
    windows: list[dict[str, Any]],
    *,
    required_windows: int,
    tolerance: float,
) -> bool:
    if len(windows) < required_windows:
        return False
    recent = windows[-required_windows:]
    throughput = [
        float(window["speed"]["throughput_samples_per_s"]) for window in recent
    ]
    latency = [float(window["speed"]["latency_median_s"]) for window in recent]
    stable = (
        _relative_spread(throughput) <= tolerance
        and _relative_spread(latency) <= tolerance
    )
    cpu_ms = [
        window.get("profile_harness", {}).get("cpu_ms_per_request") for window in recent
    ]
    if all(value is not None for value in cpu_ms):
        stable = (
            stable and _relative_spread([float(value) for value in cpu_ms]) <= tolerance
        )
    return stable


def _process_cpu_seconds(pid: int | None) -> float | None:
    """Read Linux process user+system CPU time without launching a collector."""
    if pid is None:
        return None
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        clock_ticks = os.sysconf("SC_CLK_TCK")
        return (int(fields[13]) + int(fields[14])) / float(clock_ticks)
    except (OSError, ValueError, IndexError):
        return None


def _summarize_pass(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    speed = result["speed"]
    summarized = {
        "evaluated": summary["evaluated"],
        "total": summary["total_samples"],
        "skipped": summary["skipped"],
        "corpus_wer": summary.get("corpus_wer"),
        "wall_clock_s": result["wall_clock_s"],
        "throughput_samples_per_s": speed["throughput_samples_per_s"],
        "latency_mean_s": speed.get("latency_mean_s"),
        "latency_p50_s": speed.get("latency_median_s"),
        "latency_p95_s": speed.get("latency_p95_s"),
        "latency_p99_s": speed.get("latency_p99_s"),
        "rtf_mean": speed.get("rtf_mean"),
        "rtfx": speed.get("rtfx"),
    }
    per_sample = result.get("per_sample") or []
    http_dispatched = sum(
        1
        for row in per_sample
        if (row.get("client_timing") or {}).get("http_start_ns") is not None
    )
    http_rejected = sum(
        1
        for row in per_sample
        if isinstance(row.get("http_status"), int) and row["http_status"] >= 400
    )
    timed_out = sum(
        1 for row in per_sample if "timeout" in str(row.get("error", "")).lower()
    )
    client_queue = [
        float((row.get("client_timing") or {}).get("client_queue_s"))
        for row in per_sample
        if (row.get("client_timing") or {}).get("client_queue_s") is not None
    ]
    summarized["request_accounting"] = {
        "offered": len(per_sample),
        "http_dispatched": http_dispatched,
        "completed": int(summary["evaluated"]),
        "failed": int(summary["skipped"]),
        "http_rejected": http_rejected,
        "timed_out": timed_out,
        "max_client_queue_s": max(client_queue, default=0.0),
    }
    summarized.update(result.get("profile_harness", {}))
    return summarized


async def _post_json(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    async with session.post(url, json=payload) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"POST {url} failed with HTTP {response.status}: {body}")
        decoded = json.loads(body)
        if not isinstance(decoded, dict):
            raise TypeError(f"POST {url} returned non-object JSON")
        return decoded


async def _run_pass(
    args: argparse.Namespace,
    samples: list[Any],
    *,
    concurrency: int | None = None,
    request_rate: float | None = None,
) -> dict[str, Any]:
    pass_index = int(getattr(args, "_profile_pass_index", 0)) + 1
    args._profile_pass_index = pass_index
    run_digest = hashlib.sha256(args.run_id.encode("utf-8")).hexdigest()[:10]
    return await run_asr_seedtts_once(
        samples,
        host=args.host,
        port=args.port,
        concurrency=args.concurrency if concurrency is None else concurrency,
        request_rate=args.request_rate if request_rate is None else request_rate,
        request_rate_seed=args.request_rate_seed,
        model_path=args.model_path,
        lang=args.lang,
        disable_tqdm=args.disable_tqdm,
        stream=args.stream,
        request_id_prefix=f"profile-{run_digest}-{pass_index:04d}-",
    )


async def _warm_to_stability(
    args: argparse.Namespace,
    samples: list[Any],
    *,
    artifact_dir: Path,
) -> dict[str, Any]:
    warmup_dir = artifact_dir / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=False)
    shape_passes: list[dict[str, Any]] = []
    if args.workload_contract == "direct-steady-miss":
        shape_samples = _duration_stratified_subset(
            samples,
            args.shape_warmup_samples,
        )
        for pass_index in range(1, args.shape_warmup_passes + 1):
            result = await _run_pass(args, shape_samples)
            _require_complete(result, allow_failures=False)
            path = warmup_dir / f"shape_pass_{pass_index}.json"
            write_json(path, result)
            shape_passes.append(
                {
                    "pass": pass_index,
                    "samples": len(shape_samples),
                    "artifact": str(path),
                    "summary": _summarize_pass(result),
                }
            )

    warmup_samples = _duration_stratified_subset(samples, args.warmup_samples)
    if len(warmup_samples) < args.concurrency:
        raise ValueError("warmup sample count must be at least the target concurrency")
    windows: list[dict[str, Any]] = []
    for window_index in range(1, args.max_warmup_windows + 1):
        cpu_before = _process_cpu_seconds(args.server_pid)
        if args.server_pid is not None and cpu_before is None:
            raise RuntimeError(
                f"cannot read CPU time for --server-pid {args.server_pid}"
            )
        if args.workload_contract == "issue-reproduction":
            result = await _run_pass(
                args,
                warmup_samples,
                concurrency=1,
                request_rate=float("inf"),
            )
        else:
            result = await _run_pass(args, warmup_samples)
        cpu_after = _process_cpu_seconds(args.server_pid)
        if args.server_pid is not None and cpu_after is None:
            raise RuntimeError(f"server process {args.server_pid} exited during warmup")
        evaluated = int(result["summary"]["evaluated"])
        cpu_ms_per_request = (
            (cpu_after - cpu_before) * 1000.0 / evaluated
            if cpu_before is not None and cpu_after is not None and evaluated
            else None
        )
        result["profile_harness"] = {
            "cpu_ms_per_request": cpu_ms_per_request,
            "cpu_pid": args.server_pid,
        }
        windows.append(result)
        window_path = warmup_dir / f"stability_window_{window_index}.json"
        write_json(window_path, result)
        summary = _summarize_pass(result)
        cpu_text = (
            f" cpu_ms/req={cpu_ms_per_request:.3f}"
            if cpu_ms_per_request is not None
            else ""
        )
        print(
            f"warmup[{window_index}] qps="
            f"{summary['throughput_samples_per_s']:.3f} "
            f"p50={summary['latency_p50_s']:.4f}s"
            f"{cpu_text} evaluated={summary['evaluated']}/{summary['total']}"
        )
        if summary["evaluated"] != summary["total"]:
            raise RuntimeError("warmup had failed/skipped requests")
        if _is_stable(
            windows,
            required_windows=args.stability_windows,
            tolerance=args.stability_tolerance,
        ):
            return {
                "contract": args.workload_contract,
                "shape_passes": shape_passes,
                "stability_windows": [
                    {
                        "window": index,
                        "artifact": str(warmup_dir / f"stability_window_{index}.json"),
                        "summary": _summarize_pass(window),
                    }
                    for index, window in enumerate(windows, 1)
                ],
            }
    raise RuntimeError(
        f"server did not reach warmup stability after {args.max_warmup_windows} windows"
    )


def _metric_distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)

    def percentile(fraction: float) -> float | None:
        if not ordered:
            return None
        return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]

    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def _request_integrity(
    shape_passes: list[dict[str, Any]],
    windows: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    offered = 0
    completed = 0
    failed = 0
    rejected = 0
    timed_out = 0
    for label, summary in (
        *[(f"shape pass {row['pass']}", row["summary"]) for row in shape_passes],
        *[(f"stability window {row['window']}", row["summary"]) for row in windows],
    ):
        accounting = summary.get("request_accounting") or {}
        row_offered = int(accounting.get("offered", summary.get("total", 0)))
        row_completed = int(accounting.get("completed", summary.get("evaluated", 0)))
        row_failed = int(accounting.get("failed", summary.get("skipped", 0)))
        row_rejected = int(accounting.get("http_rejected", 0))
        row_timed_out = int(accounting.get("timed_out", 0))
        offered += row_offered
        completed += row_completed
        failed += row_failed
        rejected += row_rejected
        timed_out += row_timed_out
        if row_completed != row_offered or row_failed or row_rejected or row_timed_out:
            errors.append(
                f"{label} completed {row_completed}/{row_offered}, "
                f"failed={row_failed}, rejected={row_rejected}, "
                f"timed_out={row_timed_out}"
            )
    return {
        "valid": not errors,
        "errors": errors,
        "offered": offered,
        "completed": completed,
        "failed": failed,
        "http_rejected": rejected,
        "timed_out": timed_out,
    }


def _thread_accounting(
    *,
    process_cpu_ms: float | None,
    thread_delta: dict[str, Any],
    max_relative_error: float,
) -> dict[str, Any]:
    thread_cpu_ms = thread_delta.get("cpu_ms")
    relative_error = (
        abs(float(process_cpu_ms) - float(thread_cpu_ms)) / float(process_cpu_ms)
        if isinstance(process_cpu_ms, (int, float))
        and process_cpu_ms > 0
        and isinstance(thread_cpu_ms, (int, float))
        else None
    )
    errors: list[str] = []
    if process_cpu_ms is None:
        errors.append("stage process CPU delta is unavailable")
    if not thread_delta.get("threads_observed"):
        errors.append("no attributable native threads were observed")
    if thread_delta.get("threads_exited"):
        errors.append(f"threads exited during window: {thread_delta['threads_exited']}")
    if relative_error is None:
        errors.append("process/thread CPU accounting cannot be reconciled")
    elif relative_error > max_relative_error:
        errors.append(
            "process/thread CPU accounting relative error "
            f"{relative_error:.4f} exceeds {max_relative_error:.4f}"
        )
    return {
        "valid": not errors,
        "errors": errors,
        "process_cpu_ms": process_cpu_ms,
        "thread_cpu_ms": thread_cpu_ms,
        "relative_error": relative_error,
        "max_relative_error": max_relative_error,
    }


def _parse_continuous_system_artifacts(
    artifact_dir: Path,
    *,
    completed_requests: int,
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    perf_path = artifact_dir / "perf_stat.csv"
    if perf_path.is_file():
        perf_summary = parse_perf_stat(perf_path)
        perf_summary["completed_requests"] = completed_requests
        for event, counter in perf_summary["counters"].items():
            counter["per_completed_request"] = (
                counter["value"] / completed_requests if completed_requests else None
            )
            if event == "task-clock" and counter["unit"] == "msec":
                perf_summary["cpu_ms_per_request"] = counter["per_completed_request"]
        counters = perf_summary["counters"]
        cycles = (counters.get("cycles") or {}).get("value")
        ref_cycles = (counters.get("ref-cycles") or {}).get("value")
        instructions = (counters.get("instructions") or {}).get("value")
        perf_summary["derived"] = {
            "instructions_per_cycle": (
                instructions / cycles if instructions and cycles else None
            ),
            "cycles_per_ref_cycle": (
                cycles / ref_cycles if cycles and ref_cycles else None
            ),
        }
        parsed["perf_stat"] = perf_summary
    turbostat_path = artifact_dir / "turbostat.txt"
    if turbostat_path.is_file():
        parsed["turbostat"] = parse_turbostat(turbostat_path)
    thread_path = artifact_dir / "thread_snapshots.jsonl"
    if thread_path.is_file():
        parsed["thread_summary"] = parse_thread_snapshots(thread_path)
    gpu_path = artifact_dir / "gpu_dmon.txt"
    if gpu_path.is_file():
        parsed["gpu_dmon"] = parse_gpu_dmon(gpu_path)
    cpu_frequency_path = artifact_dir / "cpu_frequency.jsonl"
    if cpu_frequency_path.is_file():
        parsed["cpu_frequency"] = parse_cpu_frequency(cpu_frequency_path)
    return parsed


def _stability_system_integrity_errors(
    args: argparse.Namespace,
    artifact_dir: Path,
    system_result: dict[str, Any],
    windows: list[dict[str, Any]],
) -> list[str]:
    requested = _requested_collectors(args)
    errors: list[str] = []
    if "perf-stat" in requested and not (
        system_result.get("perf_stat", {}).get("counters")
    ):
        errors.append("perf-stat produced no parseable counters")
    if "perf-sched" in requested:
        path = artifact_dir / "perf_sched.data"
        if not path.is_file() or path.stat().st_size == 0:
            errors.append("perf sched did not finalize a non-empty data file")
    if "turbostat" in requested and not (
        (
            system_result.get("turbostat", {}).get("columns", {}).get("Bzy_MHz") or {}
        ).get("samples")
    ):
        errors.append("turbostat produced no parseable Bzy_MHz samples")
    if "thread-snapshot" in requested:
        thread_summary = system_result.get("thread_summary") or {}
        if int(thread_summary.get("samples", 0)) < 2:
            errors.append("thread snapshot collector produced fewer than two samples")
        if not thread_summary.get("threads"):
            errors.append("thread snapshot collector observed no native threads")
        required_comms = {
            item.strip()
            for item in getattr(args, "required_thread_comms", "").split(",")
            if item.strip()
        }
        observed_comms = {
            str(thread.get("comm"))
            for thread in thread_summary.get("threads", [])
            if thread.get("comm")
        }
        missing_comms = sorted(required_comms - observed_comms)
        if missing_comms:
            errors.append(
                f"thread snapshot collector did not observe required comms "
                f"{missing_comms}; observed={sorted(observed_comms)}"
            )
    if "gpu-dmon" in requested:
        gpu = system_result.get("gpu_dmon") or {}
        if not gpu.get("samples") or not (
            (gpu.get("columns", {}).get("sm") or {}).get("samples")
        ):
            errors.append("nvidia-smi dmon produced no parseable SM samples")
    if "cpu-frequency" in requested:
        frequency = system_result.get("cpu_frequency") or {}
        if int(frequency.get("samples", 0)) < 2:
            errors.append("CPU frequency collector produced fewer than two samples")
        if not frequency.get("cpu_samples"):
            errors.append("CPU frequency collector produced no usable frequencies")
        if frequency.get("busy_weighted_sampled_scaling_frequency_mhz") is None:
            errors.append("CPU frequency collector produced no busy-weighted value")
        expected_cpus = (
            parse_cpu_list(args.cpu_frequency_cpus) if args.cpu_frequency_cpus else []
        )
        observed_cpus = set((frequency.get("scope") or {}).get("observed_cpus") or [])
        missing_cpus = sorted(set(expected_cpus) - observed_cpus)
        if missing_cpus:
            errors.append(
                f"CPU frequency collector did not observe selected CPUs {missing_cpus}"
            )
    for window in windows:
        window_index = int(window["window"])
        pressure = window.get("pressure") or {}
        if "psi" in requested and pressure.get("cpu_psi_some_fraction") is None:
            errors.append(f"stability window {window_index} has no global CPU PSI")
        if (
            "cgroup-psi" in requested
            and pressure.get("cgroup_cpu_psi_some_fraction") is None
        ):
            errors.append(f"stability window {window_index} has no cgroup CPU PSI")
        if "thread-snapshot" in requested:
            accounting = window.get("thread_accounting") or {}
            errors.extend(
                f"stability window {window_index}: {error}"
                for error in accounting.get("errors", [])
            )
        if "cpu-frequency" in requested:
            telemetry = window.get("continuous_telemetry") or {}
            frequency = telemetry.get("cpu_frequency") or {}
            coverage = frequency.get("coverage") or {}
            if (
                int(frequency.get("samples", 0)) < 2
                or not coverage.get("brackets_start")
                or not coverage.get("brackets_stop")
            ):
                errors.append(
                    f"stability window {window_index} lacks bracketing "
                    "CPU frequency samples"
                )
    return errors


async def _run_stability_characterization(
    args: argparse.Namespace,
    samples: list[Any],
    *,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Collect fixed unprofiled windows without rejecting unstable behavior."""
    if args.server_pid is None:
        raise ValueError(
            "--server-pid is required in stability mode; stage discovery must "
            "not enable the event recorder"
        )

    collectors = _build_collectors(
        args,
        artifact_dir,
        server_pid=args.server_pid,
    )
    started: list[Any] = []
    collector_results: list[dict[str, Any]] = []
    collector_start_errors: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    shape_passes: list[dict[str, Any]] = []
    run_error: BaseException | None = None
    overall_pressure_before = _read_pressure_snapshot(
        args,
        target_pid=args.server_pid,
    )

    try:
        for collector in collectors:
            try:
                collector.start()
                started.append(collector)
            except Exception as exc:  # noqa: BLE001 - preserve partial evidence
                collector_start_errors.append(
                    {
                        "name": collector.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if args.workload_contract == "direct-steady-miss":
            shape_samples = _duration_stratified_subset(
                samples,
                args.shape_warmup_samples,
            )
            for pass_index in range(1, args.shape_warmup_passes + 1):
                result = await _run_pass(args, shape_samples)
                path = artifact_dir / "shape_warmup" / f"pass_{pass_index}.json"
                write_json(path, result)
                shape_passes.append(
                    {
                        "pass": pass_index,
                        "samples": len(shape_samples),
                        "artifact": str(path),
                        "summary": _summarize_pass(result),
                    }
                )

        window_samples = _duration_stratified_subset(samples, args.warmup_samples)
        for window_index in range(1, args.characterization_windows + 1):
            pressure_before = _read_pressure_snapshot(
                args,
                target_pid=args.server_pid,
            )
            started_monotonic_ns = time.monotonic_ns()
            cpu_before = _process_cpu_seconds(args.server_pid)
            threads_before = read_thread_snapshot(args.server_pid)
            started_wall_ns = time.time_ns()
            result = await _run_pass(args, window_samples)
            stopped_wall_ns = time.time_ns()
            threads_after = read_thread_snapshot(args.server_pid)
            cpu_after = _process_cpu_seconds(args.server_pid)
            stopped_monotonic_ns = time.monotonic_ns()
            pressure_after = _read_pressure_snapshot(
                args,
                target_pid=args.server_pid,
            )

            evaluated = int(result["summary"]["evaluated"])
            process_cpu_ms = (
                (cpu_after - cpu_before) * 1000.0
                if cpu_before is not None and cpu_after is not None
                else None
            )
            cpu_ms_per_request = (
                process_cpu_ms / evaluated
                if process_cpu_ms is not None and evaluated
                else None
            )
            result["profile_harness"] = {
                "cpu_ms_per_request": cpu_ms_per_request,
                "cpu_pid": args.server_pid,
            }
            result_path = (
                artifact_dir / "stability_windows" / f"window_{window_index:02d}.json"
            )
            write_json(result_path, result)

            thread_delta = summarize_thread_snapshot_delta(
                threads_before,
                threads_after,
            )
            thread_accounting = _thread_accounting(
                process_cpu_ms=process_cpu_ms,
                thread_delta=thread_delta,
                max_relative_error=args.max_thread_cpu_accounting_error,
            )
            for row in thread_delta["threads"]:
                row["cpu_ms_per_completed_request"] = (
                    float(row["cpu_ms"]) / evaluated if evaluated else None
                )
                row["runqueue_delay_ms_per_completed_request"] = (
                    float(row["runqueue_delay_ms"]) / evaluated if evaluated else None
                )
            system_window = {
                "window": window_index,
                "started_monotonic_ns": started_monotonic_ns,
                "stopped_monotonic_ns": stopped_monotonic_ns,
                "started_wall_ns": started_wall_ns,
                "stopped_wall_ns": stopped_wall_ns,
                "pressure": _pressure_window(pressure_before, pressure_after),
                "thread_delta": thread_delta,
                "thread_accounting": thread_accounting,
                "process_cpu_s_before": cpu_before,
                "process_cpu_s_after": cpu_after,
                "process_cpu_ms": process_cpu_ms,
                "cpu_ms_per_completed_request": cpu_ms_per_request,
            }
            system_path = (
                artifact_dir
                / "stability_windows"
                / f"window_{window_index:02d}_system.json"
            )
            write_json(system_path, system_window)

            summary = _summarize_pass(result)
            window_record = {
                "window": window_index,
                "started_monotonic_ns": system_window["started_monotonic_ns"],
                "stopped_monotonic_ns": system_window["stopped_monotonic_ns"],
                "started_wall_ns": system_window["started_wall_ns"],
                "stopped_wall_ns": system_window["stopped_wall_ns"],
                "artifact": str(result_path),
                "system_artifact": str(system_path),
                "summary": summary,
                "pressure": {
                    "cpu_psi_some_fraction": system_window["pressure"].get(
                        "cpu_psi_some_fraction"
                    ),
                    "cgroup_cpu_psi_some_fraction": system_window["pressure"].get(
                        "cgroup_cpu_psi_some_fraction"
                    ),
                },
                "thread_totals": {
                    key: thread_delta.get(key)
                    for key in (
                        "cpu_ms",
                        "runtime_ms",
                        "runqueue_delay_ms",
                        "migrations",
                        "threads_observed",
                    )
                },
                "thread_accounting": thread_accounting,
            }
            windows.append(window_record)
            write_json(
                artifact_dir / "stability_progress.json",
                {
                    "requested_windows": args.characterization_windows,
                    "completed_windows": len(windows),
                    "windows": windows,
                },
            )
            print(
                f"stability[{window_index}] "
                f"qps={summary['throughput_samples_per_s']:.3f} "
                f"p50={summary['latency_p50_s']:.4f}s "
                f"cpu_ms/req={cpu_ms_per_request:.3f} "
                f"evaluated={summary['evaluated']}/{summary['total']}"
            )
    except BaseException as exc:  # noqa: BLE001 - finalize all evidence
        run_error = exc
    finally:
        for collector in reversed(started):
            try:
                collector_results.append(collector.stop())
            except Exception as exc:  # noqa: BLE001 - retain collector failures
                collector_results.append(
                    {
                        "name": collector.name,
                        "returncode": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    overall_pressure_after = _read_pressure_snapshot(
        args,
        target_pid=args.server_pid,
    )
    completed_requests = sum(int(window["summary"]["evaluated"]) for window in windows)
    for window in windows:
        continuous_telemetry: dict[str, Any] = {}
        cpu_frequency_path = artifact_dir / "cpu_frequency.jsonl"
        if cpu_frequency_path.is_file():
            continuous_telemetry["cpu_frequency"] = parse_cpu_frequency(
                cpu_frequency_path,
                start_monotonic_ns=int(window["started_monotonic_ns"]),
                stop_monotonic_ns=int(window["stopped_monotonic_ns"]),
            )
        window["continuous_telemetry"] = continuous_telemetry
        telemetry_path = (
            artifact_dir
            / "stability_windows"
            / f"window_{int(window['window']):02d}_continuous.json"
        )
        write_json(telemetry_path, continuous_telemetry)
        window["continuous_system_artifact"] = str(telemetry_path)
    write_json(
        artifact_dir / "stability_progress.json",
        {
            "requested_windows": args.characterization_windows,
            "completed_windows": len(windows),
            "windows": windows,
        },
    )
    system_result = {
        "target_pid": args.server_pid,
        "collector_start_errors": collector_start_errors,
        "collectors": collector_results,
        "overall_pressure": _pressure_window(
            overall_pressure_before,
            overall_pressure_after,
        ),
        **_parse_continuous_system_artifacts(
            artifact_dir,
            completed_requests=completed_requests,
        ),
    }
    system_integrity_errors = [
        *[f"{row['name']}: {row['error']}" for row in collector_start_errors],
        *[
            (
                f"{row['name']}: {row.get('error')}"
                if row.get("error")
                else f"{row['name']}: returncode={row.get('returncode')}"
            )
            for row in collector_results
            if row.get("returncode") != 0 or row.get("error")
        ],
    ]
    system_integrity_errors.extend(
        _stability_system_integrity_errors(
            args,
            artifact_dir,
            system_result,
            windows,
        )
    )
    system_result["integrity_errors"] = system_integrity_errors
    system_result["valid"] = not system_integrity_errors
    write_json(artifact_dir / "system.json", system_result)

    rolling_stability: list[dict[str, Any]] = []
    raw_results = [
        json.loads(Path(window["artifact"]).read_text(encoding="utf-8"))
        for window in windows
    ]
    for end in range(args.stability_windows, len(raw_results) + 1):
        recent = raw_results[:end]
        rolling_stability.append(
            {
                "ending_window": end,
                "stable": _is_stable(
                    recent,
                    required_windows=args.stability_windows,
                    tolerance=args.stability_tolerance,
                ),
            }
        )

    metric_names = (
        "throughput_samples_per_s",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "cpu_ms_per_request",
    )
    distributions = {
        name: _metric_distribution(
            [
                float(value)
                for window in windows
                if isinstance(
                    value := window["summary"].get(name),
                    (int, float),
                )
            ]
        )
        for name in metric_names
    }
    request_integrity = _request_integrity(shape_passes, windows)
    capture_complete = (
        run_error is None and len(windows) == args.characterization_windows
    )
    integrity_errors = [
        *(
            []
            if capture_complete
            else ["stability capture did not complete every requested window"]
        ),
        *request_integrity["errors"],
        *system_integrity_errors,
    ]
    result_payload = {
        "run_id": args.run_id,
        "mode": args.mode,
        "artifact_dir": str(artifact_dir),
        "workload_contract": args.workload_contract,
        "capture_complete": capture_complete,
        "run_error": (
            f"{type(run_error).__name__}: {run_error}"
            if run_error is not None
            else None
        ),
        "shape_passes": shape_passes,
        "stability_characterization": {
            "requested_windows": args.characterization_windows,
            "completed_windows": len(windows),
            "stability_windows": args.stability_windows,
            "stability_tolerance": args.stability_tolerance,
            "ever_stable": any(item["stable"] for item in rolling_stability),
            "stable_endings": [
                item["ending_window"] for item in rolling_stability if item["stable"]
            ],
            "rolling_stability": rolling_stability,
            "distributions": distributions,
            "windows": windows,
        },
        # Preserve the campaign aggregator's existing metric input contract.
        "measured": [window["summary"] for window in windows],
        "request_integrity": request_integrity,
        "system_integrity": {
            "valid": not system_integrity_errors,
            "errors": system_integrity_errors,
        },
        "accepted": not integrity_errors,
        "integrity_errors": integrity_errors,
    }
    write_json(artifact_dir / "result.json", result_payload)
    write_json(artifact_dir / "artifact_index.json", _artifact_index(artifact_dir))
    if run_error is not None:
        raise run_error
    return result_payload


def _build_collectors(
    args: argparse.Namespace,
    artifact_dir: Path,
    *,
    server_pid: int | None,
) -> list[Any]:
    requested = _requested_collectors(args)
    if requested and server_pid is None:
        needs_pid = requested & {"perf-stat", "perf-sched", "thread-snapshot"}
        if needs_pid:
            raise ValueError(
                f"--server-pid is required for collectors {sorted(needs_pid)}"
            )
    collectors: list[Any] = []
    if "perf-stat" in requested:
        assert server_pid is not None
        collectors.append(
            perf_stat_collector(
                pid=server_pid,
                output_path=artifact_dir / "perf_stat.csv",
                executable=args.perf_binary,
                events=(
                    tuple(
                        event.strip()
                        for event in args.perf_events.split(",")
                        if event.strip()
                    )
                    if args.perf_events
                    else (
                        "task-clock",
                        "cycles",
                        "ref-cycles",
                        "instructions",
                        "context-switches",
                        "cpu-migrations",
                        "page-faults",
                    )
                ),
            )
        )
    if "perf-sched" in requested:
        assert server_pid is not None
        collectors.append(
            perf_sched_collector(
                pid=server_pid,
                output_path=artifact_dir / "perf_sched.data",
                executable=args.perf_binary,
            )
        )
    if "turbostat" in requested:
        collectors.append(
            turbostat_collector(
                output_path=artifact_dir / "turbostat.txt",
                cpus=args.turbostat_cpus,
            )
        )
    if "thread-snapshot" in requested:
        assert server_pid is not None
        collectors.append(
            ThreadSnapshotCollector(
                pid=server_pid,
                output_path=artifact_dir / "thread_snapshots.jsonl",
                interval_ms=args.thread_sample_interval_ms,
            )
        )
    if "gpu-dmon" in requested:
        if args.gpu_index is None:
            raise ValueError("--gpu-index is required for gpu-dmon")
        collectors.append(
            gpu_dmon_collector(
                gpu_index=args.gpu_index,
                output_path=artifact_dir / "gpu_dmon.txt",
            )
        )
    if "cpu-frequency" in requested:
        collectors.append(
            CpuFrequencyCollector(
                output_path=artifact_dir / "cpu_frequency.jsonl",
                cpus=(
                    parse_cpu_list(args.cpu_frequency_cpus)
                    if args.cpu_frequency_cpus
                    else None
                ),
            )
        )
    unknown = requested - {
        "perf-stat",
        "perf-sched",
        "turbostat",
        "psi",
        "cgroup-psi",
        "thread-snapshot",
        "gpu-dmon",
        "cpu-frequency",
    }
    if unknown:
        raise ValueError(f"unknown collectors: {sorted(unknown)}")
    return collectors


def _profile_target_pid(start_response: dict[str, Any] | None) -> int | None:
    if not start_response:
        return None
    manifest = start_response.get("manifest") or {}
    for stage in manifest.get("stages", []):
        for target in stage.get("targets", []):
            pid = target.get("pid")
            if isinstance(pid, int):
                return pid
    return None


def _profile_target_pids(response: dict[str, Any] | None) -> list[int]:
    if not response:
        return []
    manifest = response.get("manifest") or response
    pids: list[int] = []
    for stage in manifest.get("stages", []):
        for target in stage.get("targets", []):
            pid = target.get("pid")
            if isinstance(pid, int):
                pids.append(pid)
    return sorted(set(pids))


async def _discover_stage_pid(
    args: argparse.Namespace,
    artifact_dir: Path,
) -> int:
    """Resolve the measured stage PID through its acknowledged control plane."""
    base_url = f"http://{args.host}:{args.port}"
    discovery_run_id = f"{args.run_id}-target-discovery"
    timeout = aiohttp.ClientTimeout(total=args.profile_timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        start = await _post_json(
            session,
            f"{base_url}/start_request_profile",
            {
                "run_id": discovery_run_id,
                "stages": [args.stage],
                "event_dir": str(artifact_dir / "target_discovery" / "events"),
                "timeout_s": args.profile_timeout_s,
            },
        )
        write_json(artifact_dir / "target_discovery" / "start.json", start)
        try:
            pids = _profile_target_pids(start)
        finally:
            stop = await _post_json(
                session,
                f"{base_url}/stop_request_profile",
                {
                    "run_id": discovery_run_id,
                    "stages": [args.stage],
                    "timeout_s": args.profile_timeout_s,
                },
            )
            write_json(artifact_dir / "target_discovery" / "stop.json", stop)
    discovery_report = validate_stop_response(
        stop,
        require_cuda=False,
        require_events=True,
        require_schedule_complete=False,
        require_nonempty_events=False,
    )
    write_json(
        artifact_dir / "target_discovery" / "integrity.json",
        discovery_report.to_dict(),
    )
    if not discovery_report.valid:
        raise RuntimeError(
            "stage PID discovery did not finalize cleanly: "
            + "; ".join(discovery_report.errors)
        )
    if len(pids) != 1:
        raise RuntimeError(
            f"single-H100 protocol requires exactly one {args.stage} target; "
            f"acknowledged pids={pids}"
        )
    return pids[0]


def _event_paths_from_stop(stop_response: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    coordinator = stop_response.get("coordinator_event_path")
    if coordinator:
        paths.append(str(coordinator))
    manifest = stop_response.get("manifest") or {}
    for stage in manifest.get("stages", []):
        for target in stage.get("targets", []):
            path = (target.get("events") or {}).get("path")
            if path:
                paths.append(str(path))
    return paths


def _system_integrity_errors(
    args: argparse.Namespace,
    artifact_dir: Path,
    system_result: dict[str, Any],
) -> list[str]:
    requested = _requested_collectors(args)
    errors: list[str] = []
    if "perf-stat" in requested:
        counters = system_result.get("perf_stat", {}).get("counters", {})
        if not counters:
            errors.append("perf-stat produced no parseable counters")
    if "perf-sched" in requested:
        perf_sched = artifact_dir / "perf_sched.data"
        if not perf_sched.is_file() or perf_sched.stat().st_size == 0:
            errors.append("perf sched did not finalize a non-empty data file")
    if "turbostat" in requested:
        bzy = system_result.get("turbostat", {}).get("columns", {}).get("Bzy_MHz", {})
        if not bzy.get("samples"):
            errors.append("turbostat produced no parseable Bzy_MHz samples")
    if "psi" in requested and system_result.get("psi_delta") is None:
        errors.append("PSI snapshots were not completed")
    if "cgroup-psi" in requested and system_result.get("cgroup_psi_delta") is None:
        errors.append("cgroup PSI snapshots were not completed")
    if "thread-snapshot" in requested:
        thread_path = artifact_dir / "thread_snapshots.jsonl"
        if not thread_path.is_file() or thread_path.stat().st_size == 0:
            errors.append("thread snapshot collector produced no finalized samples")
    if "gpu-dmon" in requested:
        gpu_path = artifact_dir / "gpu_dmon.txt"
        if not gpu_path.is_file() or gpu_path.stat().st_size == 0:
            errors.append("nvidia-smi dmon produced no samples")
    if "cpu-frequency" in requested:
        frequency = system_result.get("cpu_frequency") or {}
        if int(frequency.get("samples", 0)) < 2:
            errors.append("CPU frequency collector produced fewer than two samples")
        if frequency.get("busy_weighted_sampled_scaling_frequency_mhz") is None:
            errors.append("CPU frequency collector produced no busy-weighted value")
    return errors


def _cpu_psi_some_fraction(delta: dict[str, Any] | None) -> float | None:
    if not delta:
        return None
    stall_us = ((delta.get("cpu") or {}).get("some") or {}).get("total_us")
    window_ns = delta.get("window_ns")
    if (
        not isinstance(stall_us, (int, float))
        or not isinstance(window_ns, (int, float))
        or window_ns <= 0
    ):
        return None
    return float(stall_us) * 1000.0 / float(window_ns)


def _read_pressure_snapshot(
    args: argparse.Namespace,
    *,
    target_pid: int | None,
) -> dict[str, Any]:
    requested = _requested_collectors(args)
    return {
        "psi": read_psi() if "psi" in requested else None,
        "cgroup_psi": (
            read_process_cgroup_psi(target_pid)
            if target_pid is not None and "cgroup-psi" in requested
            else None
        ),
    }


def _pressure_window(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    global_before = before.get("psi")
    global_after = after.get("psi")
    cgroup_before = before.get("cgroup_psi")
    cgroup_after = after.get("cgroup_psi")
    global_delta = (
        psi_delta(global_before, global_after)
        if global_before is not None and global_after is not None
        else None
    )
    cgroup_delta = (
        psi_delta(cgroup_before, cgroup_after)
        if cgroup_before is not None
        and cgroup_after is not None
        and "error" not in cgroup_before
        and "error" not in cgroup_after
        else None
    )
    return {
        "before": before,
        "after": after,
        "psi_delta": global_delta,
        "cgroup_psi_delta": cgroup_delta,
        "cpu_psi_some_fraction": _cpu_psi_some_fraction(global_delta),
        "cgroup_cpu_psi_some_fraction": _cpu_psi_some_fraction(cgroup_delta),
    }


def _pressure_limit_errors(
    args: argparse.Namespace,
    *,
    label: str,
    window: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    requested = _requested_collectors(args)
    if "psi" in requested and window.get("psi_delta") is None:
        errors.append(f"{label}: global PSI snapshots were not completed")
    if "cgroup-psi" in requested and window.get("cgroup_psi_delta") is None:
        errors.append(f"{label}: cgroup PSI snapshots were not completed")
    for pressure_name, observed, limit in (
        (
            "global CPU PSI some",
            window.get("cpu_psi_some_fraction"),
            args.max_cpu_psi_some_fraction,
        ),
        (
            "stage cgroup CPU PSI some",
            window.get("cgroup_cpu_psi_some_fraction"),
            args.max_cgroup_cpu_psi_some_fraction,
        ),
    ):
        if limit is not None and observed is not None and observed > limit:
            errors.append(
                f"{label}: {pressure_name} fraction {observed:.4f} "
                f"exceeds {limit:.4f}"
            )
    return errors


def _midpoint_summary(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float]:
    """Build the unprofiled reference at the profiled window's time position."""
    reference: dict[str, float] = {}
    for key in (
        "throughput_samples_per_s",
        "latency_mean_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
        "rtf_mean",
        "rtfx",
        "cpu_ms_per_request",
    ):
        before_value = before.get(key)
        after_value = after.get(key)
        if isinstance(before_value, (int, float)) and isinstance(
            after_value, (int, float)
        ):
            reference[key] = (float(before_value) + float(after_value)) / 2.0
    return reference


def _build_profile_perturbation(
    before: dict[str, Any],
    after: dict[str, Any],
    measured: list[dict[str, Any]],
) -> dict[str, Any]:
    reference = _midpoint_summary(before, after)
    before_qps = float(before["throughput_samples_per_s"])
    after_qps = float(after["throughput_samples_per_s"])
    reference_qps = float(reference["throughput_samples_per_s"])
    profiled_qps = statistics.mean(
        float(result["throughput_samples_per_s"]) for result in measured
    )
    return {
        "baseline_before_qps": before_qps,
        "baseline_after_qps": after_qps,
        "baseline_qps": reference_qps,
        "baseline_relative_drift": (
            (after_qps - before_qps) / reference_qps if reference_qps else None
        ),
        "profiled_qps": profiled_qps,
        "relative_qps_change": (
            (profiled_qps - reference_qps) / reference_qps if reference_qps else None
        ),
        "within_5_percent": (
            abs(profiled_qps - reference_qps) / reference_qps <= 0.05
            if reference_qps
            else False
        ),
    }


def _perturbation_integrity_errors(
    args: argparse.Namespace,
    perturbation: dict[str, Any] | None,
) -> list[str]:
    if perturbation is None:
        return []
    drift = perturbation.get("baseline_relative_drift")
    if drift is None or abs(float(drift)) > args.max_adjacent_baseline_drift:
        observed = "unavailable" if drift is None else f"{float(drift):+.2%}"
        return [
            "adjacent unprofiled baseline drift "
            f"{observed} exceeds {args.max_adjacent_baseline_drift:.2%}; "
            "profiler perturbation is inconclusive"
        ]
    relative_change = perturbation.get("relative_qps_change")
    if (
        args.mode == "events"
        and relative_change is not None
        and abs(float(relative_change)) > args.max_event_overhead
        and not args.allow_event_overhead
    ):
        return [
            "event-enabled QPS change "
            f"{float(relative_change):+.2%} exceeds {args.max_event_overhead:.2%}; "
            "the trace may have a material probe effect"
        ]
    return []


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    artifact_dir = Path(args.output_dir).expanduser().resolve() / args.run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    samples = load_seedtts_samples(
        args.meta,
        max_samples=args.max_samples or None,
        split=args.lang,
    )
    if len(samples) < args.concurrency:
        raise ValueError("sample count must be at least target concurrency")
    measurement_samples = _duration_stratified_subset(
        samples,
        args.profile_samples if args.mode != "baseline" else 0,
    )
    if len(measurement_samples) < args.concurrency:
        raise ValueError("measurement sample count must be at least target concurrency")

    if args.mode == "stability" and args.server_pid is None:
        raise ValueError(
            "--server-pid is required in stability mode; stage discovery must "
            "not enable the event recorder"
        )
    if args.server_pid is None:
        args.server_pid = await _discover_stage_pid(args, artifact_dir)

    static_manifest = collect_static_manifest(
        server_pid=args.server_pid,
        repo_root=_REPO_ROOT,
    )
    static_manifest["run"] = {
        key: value for key, value in vars(args).items() if key not in {"func"}
    }
    static_manifest["model"] = {
        "repository_or_path": args.model_path,
        "immutable_revision": args.model_revision,
    }
    static_manifest["dataset"] = _dataset_manifest(
        samples,
        measurement_samples,
        source=args.meta,
        split=args.lang,
    )
    write_json(artifact_dir / "manifest.json", static_manifest)

    if args.mode == "stability":
        return await _run_stability_characterization(
            args,
            samples,
            artifact_dir=artifact_dir,
        )

    warmup = await _warm_to_stability(
        args,
        samples,
        artifact_dir=artifact_dir,
    )
    write_json(artifact_dir / "warmup.json", warmup)

    adjacent_baseline = None
    adjacent_baseline_after = None
    adjacent_baseline_pressure: dict[str, Any] = {}
    if args.mode != "baseline" and not args.skip_adjacent_baseline:
        pressure_before = _read_pressure_snapshot(
            args,
            target_pid=args.server_pid,
        )
        adjacent_baseline = await _run_pass(args, measurement_samples)
        _require_complete(adjacent_baseline, allow_failures=args.allow_failures)
        pressure_after = _read_pressure_snapshot(
            args,
            target_pid=args.server_pid,
        )
        adjacent_baseline_pressure["before"] = _pressure_window(
            pressure_before,
            pressure_after,
        )
        write_json(
            artifact_dir / "adjacent_baseline.json",
            adjacent_baseline,
        )
        write_json(
            artifact_dir / "adjacent_baseline_system.json",
            adjacent_baseline_pressure,
        )

    base_url = f"http://{args.host}:{args.port}"
    start_response = None
    stop_response = None
    profile_started_wall_ns: int | None = None
    profile_endpoint = None
    stop_endpoint = None
    timeout = aiohttp.ClientTimeout(total=args.profile_timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if args.mode in {"torch", "nsys"}:
            profile_endpoint = f"{base_url}/start_profile"
            stop_endpoint = f"{base_url}/stop_profile"
            start_response = await _post_json(
                session,
                profile_endpoint,
                {
                    "run_id": args.run_id,
                    "stages": [args.stage],
                    "event_dir": str(artifact_dir / "events"),
                    "trace_path_template": str(artifact_dir / "{stage}" / "torch"),
                    "enable_torch": args.mode == "torch",
                    "enable_nvtx": args.mode == "nsys",
                    "torch_owner": args.torch_owner,
                    "timeout_s": args.profile_timeout_s,
                    "config": {
                        "wait_steps": args.torch_wait_steps,
                        "warmup_steps": args.torch_warmup_steps,
                        "active_steps": args.torch_active_steps,
                        "repeat": 1,
                        "include_cuda": not args.no_cuda,
                        "record_shapes": args.record_shapes,
                        "profile_memory": args.profile_memory,
                        "with_stack": args.with_stack,
                        "with_flops": args.with_flops,
                        "compress": True,
                    },
                },
            )
        elif args.mode == "events":
            profile_endpoint = f"{base_url}/start_request_profile"
            stop_endpoint = f"{base_url}/stop_request_profile"
            start_response = await _post_json(
                session,
                profile_endpoint,
                {
                    "run_id": args.run_id,
                    "stages": [args.stage],
                    "event_dir": str(artifact_dir / "events"),
                    "timeout_s": args.profile_timeout_s,
                },
            )
        if start_response is not None:
            profile_started_wall_ns = time.time_ns()
            write_json(artifact_dir / "profile_start.json", start_response)

        acknowledged_pid = _profile_target_pid(start_response)
        pid_integrity_error = None
        if (
            args.server_pid is not None
            and acknowledged_pid is not None
            and args.server_pid != acknowledged_pid
        ):
            pid_integrity_error = (
                f"--server-pid {args.server_pid} does not match acknowledged "
                f"{args.stage} stage PID {acknowledged_pid}"
            )
        target_pid = acknowledged_pid or args.server_pid
        collectors = _build_collectors(
            args,
            artifact_dir,
            server_pid=target_pid,
        )
        psi_before = read_psi() if "psi" in _requested_collectors(args) else None
        cgroup_psi_before = (
            read_process_cgroup_psi(target_pid)
            if target_pid is not None and "cgroup-psi" in _requested_collectors(args)
            else None
        )
        started: list[Any] = []
        collector_results: list[dict[str, Any]] = []
        measured: list[dict[str, Any]] = []
        measurement_error: BaseException | None = None
        try:
            for collector in collectors:
                collector.start()
                started.append(collector)
            for repeat in range(1, args.measure_repeats + 1):
                result = await _run_pass(args, measurement_samples)
                _require_complete(result, allow_failures=args.allow_failures)
                measured.append(result)
                write_json(
                    artifact_dir / "measurement" / f"repeat_{repeat}.json",
                    result,
                )
                summary = _summarize_pass(result)
                print(
                    f"measure[{repeat}] qps="
                    f"{summary['throughput_samples_per_s']:.3f} "
                    f"p95={summary['latency_p95_s']:.4f}s "
                    f"evaluated={summary['evaluated']}/{summary['total']}"
                )
        except BaseException as exc:  # noqa: BLE001 - stop profilers on cancellation
            measurement_error = exc
        finally:
            for collector in reversed(started):
                try:
                    collector_results.append(collector.stop())
                except Exception as exc:  # noqa: BLE001 - retain collector failures
                    collector_results.append(
                        {
                            "name": collector.name,
                            "returncode": None,
                            "error": str(exc) or type(exc).__name__,
                        }
                    )
        psi_after = read_psi() if psi_before is not None else None
        cgroup_psi_after = (
            read_process_cgroup_psi(target_pid)
            if cgroup_psi_before is not None and target_pid is not None
            else None
        )
        stop_error: BaseException | None = None
        if stop_endpoint is not None:
            try:
                stop_response = await _post_json(
                    session,
                    stop_endpoint,
                    {
                        "run_id": args.run_id,
                        "stages": [args.stage],
                        "timeout_s": args.profile_timeout_s,
                    },
                )
                write_json(artifact_dir / "profile_stop.json", stop_response)
            except BaseException as exc:  # noqa: BLE001 - re-raise after cleanup
                stop_error = exc

        system_result = {
            "target_pid": target_pid,
            "collectors": collector_results,
            "psi_before": psi_before,
            "psi_after": psi_after,
            "psi_delta": (
                psi_delta(psi_before, psi_after)
                if psi_before is not None and psi_after is not None
                else None
            ),
            "cgroup_psi_before": cgroup_psi_before,
            "cgroup_psi_after": cgroup_psi_after,
            "cgroup_psi_delta": (
                psi_delta(cgroup_psi_before, cgroup_psi_after)
                if cgroup_psi_before is not None
                and cgroup_psi_after is not None
                and "error" not in cgroup_psi_before
                and "error" not in cgroup_psi_after
                else None
            ),
        }
        perf_path = artifact_dir / "perf_stat.csv"
        if perf_path.is_file():
            perf_summary = parse_perf_stat(perf_path)
            completed = sum(int(result["summary"]["evaluated"]) for result in measured)
            perf_summary["completed_requests"] = completed
            for event, counter in perf_summary["counters"].items():
                counter["per_completed_request"] = (
                    counter["value"] / completed if completed else None
                )
                if event == "task-clock" and counter["unit"] == "msec":
                    perf_summary["cpu_ms_per_request"] = counter[
                        "per_completed_request"
                    ]
            counters = perf_summary["counters"]
            cycles = (counters.get("cycles") or {}).get("value")
            ref_cycles = (counters.get("ref-cycles") or {}).get("value")
            instructions = (counters.get("instructions") or {}).get("value")
            perf_summary["derived"] = {
                "instructions_per_cycle": (
                    instructions / cycles if instructions and cycles else None
                ),
                "cycles_per_ref_cycle": (
                    cycles / ref_cycles if cycles and ref_cycles else None
                ),
            }
            system_result["perf_stat"] = perf_summary
        turbostat_path = artifact_dir / "turbostat.txt"
        if turbostat_path.is_file():
            system_result["turbostat"] = parse_turbostat(turbostat_path)
        thread_path = artifact_dir / "thread_snapshots.jsonl"
        if thread_path.is_file():
            thread_summary = parse_thread_snapshots(thread_path)
            completed = sum(int(result["summary"]["evaluated"]) for result in measured)
            for thread in thread_summary.get("threads", []):
                thread["cpu_ms_per_completed_request"] = (
                    thread["cpu_ms"] / completed if completed else None
                )
                thread["runqueue_delay_ms_per_completed_request"] = (
                    thread["runqueue_delay_ms"] / completed if completed else None
                )
            system_result["thread_summary"] = thread_summary
        gpu_path = artifact_dir / "gpu_dmon.txt"
        if gpu_path.is_file():
            system_result["gpu_dmon"] = parse_gpu_dmon(gpu_path)
        cpu_frequency_path = artifact_dir / "cpu_frequency.jsonl"
        if cpu_frequency_path.is_file():
            system_result["cpu_frequency"] = parse_cpu_frequency(cpu_frequency_path)
        system_result["integrity_errors"] = _system_integrity_errors(
            args,
            artifact_dir,
            system_result,
        )
        system_result["cpu_psi_some_fraction"] = _cpu_psi_some_fraction(
            system_result["psi_delta"]
        )
        system_result["cgroup_cpu_psi_some_fraction"] = _cpu_psi_some_fraction(
            system_result["cgroup_psi_delta"]
        )
        for label, observed, limit in (
            (
                "global CPU PSI some",
                system_result["cpu_psi_some_fraction"],
                args.max_cpu_psi_some_fraction,
            ),
            (
                "stage cgroup CPU PSI some",
                system_result["cgroup_cpu_psi_some_fraction"],
                args.max_cgroup_cpu_psi_some_fraction,
            ),
        ):
            if limit is not None and observed is not None and observed > limit:
                system_result["integrity_errors"].append(
                    f"{label} fraction {observed:.4f} exceeds {limit:.4f}"
                )
        if pid_integrity_error is not None:
            system_result["integrity_errors"].append(pid_integrity_error)
        write_json(artifact_dir / "system.json", system_result)

        if measurement_error is not None:
            if stop_error is not None:
                raise measurement_error from stop_error
            raise measurement_error
        if stop_error is not None:
            raise stop_error

    if adjacent_baseline is not None:
        pressure_before = _read_pressure_snapshot(
            args,
            target_pid=target_pid,
        )
        adjacent_baseline_after = await _run_pass(args, measurement_samples)
        _require_complete(
            adjacent_baseline_after,
            allow_failures=args.allow_failures,
        )
        pressure_after = _read_pressure_snapshot(
            args,
            target_pid=target_pid,
        )
        adjacent_baseline_pressure["after"] = _pressure_window(
            pressure_before,
            pressure_after,
        )
        write_json(
            artifact_dir / "adjacent_baseline_after.json",
            adjacent_baseline_after,
        )
        write_json(
            artifact_dir / "adjacent_baseline_system.json",
            adjacent_baseline_pressure,
        )

    adjacent_before_summary = (
        _summarize_pass(adjacent_baseline) if adjacent_baseline is not None else None
    )
    adjacent_after_summary = (
        _summarize_pass(adjacent_baseline_after)
        if adjacent_baseline_after is not None
        else None
    )
    adjacent_reference = (
        _midpoint_summary(adjacent_before_summary, adjacent_after_summary)
        if adjacent_before_summary is not None and adjacent_after_summary is not None
        else None
    )
    control_integrity_errors: list[str] = []
    for label, window in adjacent_baseline_pressure.items():
        control_integrity_errors.extend(
            _pressure_limit_errors(
                args,
                label=f"adjacent baseline {label}",
                window=window,
            )
        )
    system_integrity_errors = [
        *system_result["integrity_errors"],
        *control_integrity_errors,
    ]
    result_payload: dict[str, Any] = {
        "run_id": args.run_id,
        "mode": args.mode,
        "artifact_dir": str(artifact_dir),
        "workload_contract": args.workload_contract,
        "warmup": warmup,
        "adjacent_baseline": adjacent_before_summary,
        "adjacent_baselines": (
            {
                "before": adjacent_before_summary,
                "after": adjacent_after_summary,
                "reference": adjacent_reference,
            }
            if adjacent_before_summary is not None
            else None
        ),
        "measured": [_summarize_pass(result) for result in measured],
        "measurement_artifacts": [
            str(artifact_dir / "measurement" / f"repeat_{repeat}.json")
            for repeat in range(1, len(measured) + 1)
        ],
        "profile_start": start_response,
        "profile_stop": stop_response,
        "system_integrity": {
            "valid": not system_integrity_errors,
            "errors": system_integrity_errors,
        },
    }
    if (
        adjacent_before_summary is not None
        and adjacent_after_summary is not None
        and measured
    ):
        result_payload["profile_perturbation"] = _build_profile_perturbation(
            adjacent_before_summary,
            adjacent_after_summary,
            [_summarize_pass(result) for result in measured],
        )
        result_payload["adjacent_baselines"]["baseline_relative_drift"] = (
            result_payload["profile_perturbation"]["baseline_relative_drift"]
        )

    integrity_errors = list(system_integrity_errors)
    perturbation = result_payload.get("profile_perturbation")
    integrity_errors.extend(_perturbation_integrity_errors(args, perturbation))
    if stop_response is not None:
        integrity = validate_stop_response(
            stop_response,
            require_cuda=args.mode == "torch" and not args.no_cuda,
            require_events=True,
            require_schedule_complete=args.mode == "torch",
            forbid_event_names=(
                None if args.allow_cache_hits else {"pre_lm_cache_hit"}
            ),
        )
        result_payload["integrity"] = integrity.to_dict()
        if not integrity.valid:
            integrity_errors.extend(integrity.errors)
        expected_request_ids = {
            str(row["server_request_id"])
            for result in measured
            for row in result.get("per_sample", [])
            if row.get("server_request_id")
        }
        lifecycle = validate_request_lifecycle(
            _event_paths_from_stop(stop_response),
            expected_request_ids=expected_request_ids,
        )
        result_payload["request_lifecycle_integrity"] = lifecycle.to_dict()
        if not lifecycle.valid:
            integrity_errors.extend(lifecycle.errors)
        write_json(
            artifact_dir / "event_report.json",
            build_report(_event_paths_from_stop(stop_response)),
        )

    if args.mode == "nsys":
        assert args.nsys_report is not None
        report_path = Path(args.nsys_report).expanduser().resolve()
        deadline = time.monotonic() + args.nsys_finalize_timeout_s
        while not report_path.is_file() and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        if not report_path.is_file():
            integrity_errors.append(f"Nsight report was not finalized at {report_path}")
        else:
            stable_observations = 0
            prior_signature: tuple[int, int] | None = None
            while stable_observations < 3 and time.monotonic() < deadline:
                current_stat = report_path.stat()
                signature = (current_stat.st_size, current_stat.st_mtime_ns)
                if signature == prior_signature and current_stat.st_size > 0:
                    stable_observations += 1
                else:
                    stable_observations = 0
                    prior_signature = signature
                await asyncio.sleep(1.0)
            if stable_observations < 3:
                integrity_errors.append(
                    f"Nsight report did not reach a stable finalized size: {report_path}"
                )
            stat = report_path.stat()
            if stat.st_size == 0:
                integrity_errors.append(f"Nsight report is empty: {report_path}")
            if (
                profile_started_wall_ns is not None
                and stat.st_mtime_ns < profile_started_wall_ns
            ):
                integrity_errors.append(
                    f"Nsight report predates this profile run: {report_path}"
                )
            retained_report = artifact_dir / "nsight" / report_path.name
            retained_report.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(report_path, retained_report)
            stats_capture = capture_command(
                [
                    "nsys",
                    "stats",
                    "--report",
                    "nvtx_sum",
                    "--report",
                    "cuda_api_sum",
                    "--report",
                    "cuda_gpu_kern_sum",
                    "--report",
                    "osrt_sum",
                    str(retained_report),
                ],
                timeout_s=args.nsys_stats_timeout_s,
            )
            write_json(
                artifact_dir / "nsight" / "stats.json",
                {
                    "argv": stats_capture.argv,
                    "returncode": stats_capture.returncode,
                    "stdout": stats_capture.stdout,
                    "stderr": stats_capture.stderr,
                    "available": stats_capture.available,
                },
            )
            stats_text = f"{stats_capture.stdout}\n{stats_capture.stderr}"
            if stats_capture.returncode != 0:
                integrity_errors.append(
                    "nsys stats failed: "
                    f"{stats_capture.stderr[-1000:] or stats_capture.stdout[-1000:]}"
                )
            for label, tokens in {
                "NVTX capture window": ("sglang_omni.capture_window",),
                "CUDA API": ("cuda",),
                "CUDA kernel": ("kernel", "gpu"),
                "OS runtime": ("os runtime", "poll", "pthread"),
            }.items():
                lowered = stats_text.lower()
                if not any(token.lower() in lowered for token in tokens):
                    integrity_errors.append(
                        f"Nsight report has no validated {label} coverage"
                    )
            result_payload["nsys_report"] = {
                "source_path": str(report_path),
                "path": str(retained_report),
                "bytes": retained_report.stat().st_size,
                "sha256": _file_sha256(retained_report),
                "mtime_ns": retained_report.stat().st_mtime_ns,
                "stats_path": str(artifact_dir / "nsight" / "stats.json"),
            }

    result_payload["accepted"] = not integrity_errors
    result_payload["integrity_errors"] = list(integrity_errors)
    write_json(artifact_dir / "result.json", result_payload)
    write_json(artifact_dir / "artifact_index.json", _artifact_index(artifact_dir))
    if integrity_errors:
        raise RuntimeError(
            "profile integrity gate failed: " + "; ".join(integrity_errors)
        )
    return result_payload


def _require_complete(
    result: dict[str, Any],
    *,
    allow_failures: bool,
) -> None:
    summary = result["summary"]
    if not allow_failures and summary["evaluated"] != summary["total_samples"]:
        raise RuntimeError(
            f"only {summary['evaluated']}/{summary['total_samples']} "
            "requests were evaluated"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--stage", default="asr")
    parser.add_argument("--model-path", default=FUN_ASR_MODEL_PATH)
    parser.add_argument(
        "--model-revision",
        default=os.environ.get("HF_MODEL_REVISION"),
        help="Immutable model commit/revision; required for an accepted run.",
    )
    parser.add_argument(
        "--allow-unresolved-model-revision",
        action="store_true",
        help="Development-only escape hatch; accepted H100 runs must not use it.",
    )
    parser.add_argument("--meta", default=DATASETS["seedtts"])
    parser.add_argument("--lang", choices=["en", "zh"], default="en")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--profile-samples",
        type=int,
        default=256,
        help=(
            "Duration-stratified requests used by profiled and adjacent passes; "
            "0 uses the full corpus. Baseline mode always uses the full corpus."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Open-loop offered rate; default inf is closed-loop dispatch.",
    )
    parser.add_argument(
        "--request-rate-seed",
        type=int,
        default=20260805,
        help="Seed for deterministic Poisson inter-arrival sampling.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "events", "torch", "nsys", "stability"],
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="artifacts/cpu-saturation")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument(
        "--collectors",
        default="psi",
        help=(
            "Comma-separated: psi,cgroup-psi,thread-snapshot,gpu-dmon,"
            "cpu-frequency,perf-stat,perf-sched,turbostat"
        ),
    )
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--thread-sample-interval-ms", type=int, default=100)
    parser.add_argument(
        "--required-thread-comms",
        default="",
        help=(
            "Comma-separated Linux comm labels that a stability capture must "
            "observe, after the kernel's 15-byte truncation."
        ),
    )
    parser.add_argument(
        "--cpu-frequency-cpus",
        default="",
        help=(
            "Optional CPU list for sampled scaling frequency (for example "
            "0-15,64-79). Omit only for explicitly host-wide evidence."
        ),
    )
    parser.add_argument("--max-cpu-psi-some-fraction", type=float)
    parser.add_argument("--max-cgroup-cpu-psi-some-fraction", type=float)
    parser.add_argument(
        "--turbostat-cpus",
        help="CPU list passed to turbostat -c (for example 0-15,64-79).",
    )
    parser.add_argument(
        "--perf-events",
        help=(
            "Comma-separated perf events. Omit for the low-overhead default; "
            "use a separate pass for detailed PMU events."
        ),
    )
    parser.add_argument(
        "--perf-binary",
        default="perf",
        help="Exact perf executable or name resolved through PATH.",
    )
    parser.add_argument(
        "--workload-contract",
        choices=["direct-steady-miss", "direct-cold", "issue-reproduction"],
        default="direct-steady-miss",
    )
    parser.add_argument(
        "--shape-warmup-samples",
        type=int,
        default=0,
        help="Shape coverage pass size; 0 means the complete corpus.",
    )
    parser.add_argument("--shape-warmup-passes", type=int, default=1)
    parser.add_argument("--warmup-samples", type=int, default=256)
    parser.add_argument("--stability-windows", type=int, default=3)
    parser.add_argument("--stability-tolerance", type=float, default=0.05)
    parser.add_argument("--max-warmup-windows", type=int, default=8)
    parser.add_argument(
        "--max-thread-cpu-accounting-error",
        type=float,
        default=0.05,
        help=(
            "Maximum relative difference between process CPU time and the "
            "sum of persistent native-thread CPU time in stability windows."
        ),
    )
    parser.add_argument(
        "--characterization-windows",
        type=int,
        default=20,
        help=(
            "Fixed unprofiled windows retained in stability mode; instability "
            "is reported rather than treated as a run failure."
        ),
    )
    parser.add_argument("--measure-repeats", type=int, default=1)
    parser.add_argument("--skip-adjacent-baseline", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--max-adjacent-baseline-drift", type=float, default=0.02)
    parser.add_argument("--max-event-overhead", type=float, default=0.02)
    parser.add_argument("--allow-event-overhead", action="store_true")
    parser.add_argument(
        "--allow-cache-hits",
        action="store_true",
        help="Permit pre-LM encoder cache hits in profiled event artifacts.",
    )
    parser.add_argument("--profile-timeout-s", type=float, default=180.0)
    parser.add_argument("--torch-wait-steps", type=int, default=1)
    parser.add_argument("--torch-warmup-steps", type=int, default=1)
    parser.add_argument("--torch-active-steps", type=int, default=20)
    parser.add_argument(
        "--torch-owner",
        choices=["scheduler", "pre_lm_encoder"],
        default="scheduler",
    )
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--with-flops", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--nsys-report")
    parser.add_argument("--nsys-finalize-timeout-s", type=float, default=180.0)
    parser.add_argument("--nsys-stats-timeout-s", type=float, default=180.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--show-progress",
        action="store_false",
        dest="disable_tqdm",
        help="Show the per-request progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    if args.measure_repeats < 1:
        raise ValueError("--measure-repeats must be positive")
    if args.profile_samples < 0:
        raise ValueError("--profile-samples must be non-negative")
    if not (args.request_rate == float("inf") or args.request_rate > 0):
        raise ValueError("--request-rate must be positive or inf")
    if not args.allow_unresolved_model_revision and (
        args.model_revision is None
        or re.fullmatch(r"[0-9a-fA-F]{40,64}", args.model_revision) is None
    ):
        raise ValueError(
            "--model-revision must be an immutable 40-64 character hex commit "
            "for an accepted profiling run"
        )
    if args.mode == "nsys" and not args.nsys_report:
        raise ValueError("--nsys-report is required in nsys mode")
    if not 0 < args.stability_tolerance < 1:
        raise ValueError("--stability-tolerance must be between 0 and 1")
    if not 0 <= args.max_thread_cpu_accounting_error <= 1:
        raise ValueError("--max-thread-cpu-accounting-error must be between 0 and 1")
    if args.cpu_frequency_cpus:
        parse_cpu_list(args.cpu_frequency_cpus)
    for name in (
        "max_cpu_psi_some_fraction",
        "max_cgroup_cpu_psi_some_fraction",
        "max_adjacent_baseline_drift",
        "max_event_overhead",
    ):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    for name in (
        "warmup_samples",
        "shape_warmup_passes",
        "stability_windows",
        "max_warmup_windows",
        "characterization_windows",
        "profile_timeout_s",
        "nsys_finalize_timeout_s",
        "nsys_stats_timeout_s",
        "thread_sample_interval_ms",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    unknown_collectors = _requested_collectors(args) - {
        "perf-stat",
        "perf-sched",
        "turbostat",
        "psi",
        "cgroup-psi",
        "thread-snapshot",
        "gpu-dmon",
        "cpu-frequency",
    }
    if unknown_collectors:
        raise ValueError(f"unknown collectors: {sorted(unknown_collectors)}")
    if args.required_thread_comms and "thread-snapshot" not in _requested_collectors(
        args
    ):
        raise ValueError(
            "--required-thread-comms requires the thread-snapshot collector"
        )
    if (
        args.mode == "baseline"
        and _requested_collectors(args) & {"perf-stat", "perf-sched"}
        and args.server_pid is None
    ):
        raise ValueError("--server-pid is required for baseline perf collectors")
    result = asyncio.run(_run(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
