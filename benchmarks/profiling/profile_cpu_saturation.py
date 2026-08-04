# SPDX-License-Identifier: Apache-2.0
"""Direct-server Fun-ASR CPU-saturation profiling harness.

This command assumes the model server is already running.  One invocation is
one server-lifetime trial; repeat the command across fresh server restarts for
the interleaved multi-restart protocol in the accompanying README.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
import wave
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.eval.benchmark_asr_seedtts import run_asr_seedtts_once
from benchmarks.profiling.system_collectors import (
    ManagedCollector,
    collect_static_manifest,
    parse_perf_stat,
    parse_turbostat,
    perf_sched_collector,
    perf_stat_collector,
    psi_delta,
    read_psi,
    turbostat_collector,
    write_json,
)
from benchmarks.tasks.asr import FUN_ASR_MODEL_PATH
from sglang_omni.profiler.integrity import validate_stop_response

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


async def _run_pass(args: argparse.Namespace, samples: list[Any]) -> dict[str, Any]:
    return await run_asr_seedtts_once(
        samples,
        host=args.host,
        port=args.port,
        concurrency=args.concurrency,
        request_rate=args.request_rate,
        request_rate_seed=args.request_rate_seed,
        model_path=args.model_path,
        lang=args.lang,
        disable_tqdm=args.disable_tqdm,
        stream=args.stream,
    )


async def _warm_to_stability(
    args: argparse.Namespace,
    samples: list[Any],
) -> list[dict[str, Any]]:
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
            return windows
    raise RuntimeError(
        "server did not reach warmup stability after "
        f"{args.max_warmup_windows} windows"
    )


def _build_collectors(
    args: argparse.Namespace,
    artifact_dir: Path,
    *,
    server_pid: int | None,
) -> list[ManagedCollector]:
    requested = _requested_collectors(args)
    if requested and server_pid is None:
        needs_pid = requested & {"perf-stat", "perf-sched"}
        if needs_pid:
            raise ValueError(
                f"--server-pid is required for collectors {sorted(needs_pid)}"
            )
    collectors: list[ManagedCollector] = []
    if "perf-stat" in requested:
        assert server_pid is not None
        collectors.append(
            perf_stat_collector(
                pid=server_pid,
                output_path=artifact_dir / "perf_stat.csv",
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
            )
        )
    if "turbostat" in requested:
        collectors.append(
            turbostat_collector(
                output_path=artifact_dir / "turbostat.txt",
                cpus=args.turbostat_cpus,
            )
        )
    unknown = requested - {"perf-stat", "perf-sched", "turbostat", "psi"}
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
    return errors


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

    static_manifest = collect_static_manifest(
        server_pid=args.server_pid,
        repo_root=_REPO_ROOT,
    )
    static_manifest["run"] = {
        key: value for key, value in vars(args).items() if key not in {"func"}
    }
    static_manifest["dataset"] = {
        "samples": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "audio_paths": [sample.ref_audio for sample in samples],
        "measurement_samples": len(measurement_samples),
        "measurement_sample_ids": [sample.sample_id for sample in measurement_samples],
    }
    write_json(artifact_dir / "manifest.json", static_manifest)

    warmup_windows = await _warm_to_stability(args, measurement_samples)
    write_json(
        artifact_dir / "warmup.json",
        [_summarize_pass(result) for result in warmup_windows],
    )

    adjacent_baseline = None
    if args.mode != "baseline" and not args.skip_adjacent_baseline:
        adjacent_baseline = await _run_pass(args, measurement_samples)
        _require_complete(adjacent_baseline, allow_failures=args.allow_failures)
        write_json(
            artifact_dir / "adjacent_baseline.json",
            adjacent_baseline,
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
                    "torch_owner": "scheduler",
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
        started: list[ManagedCollector] = []
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
                summary = _summarize_pass(result)
                print(
                    f"measure[{repeat}] qps="
                    f"{summary['throughput_samples_per_s']:.3f} "
                    f"p95={summary['latency_p95_s']:.4f}s "
                    f"evaluated={summary['evaluated']}/{summary['total']}"
                )
        except BaseException as exc:
            measurement_error = exc
        finally:
            for collector in reversed(started):
                try:
                    collector_results.append(collector.stop())
                except Exception as exc:
                    collector_results.append(
                        {
                            "name": collector.name,
                            "returncode": None,
                            "error": str(exc) or type(exc).__name__,
                        }
                    )
        psi_after = read_psi() if psi_before is not None else None
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
            except BaseException as exc:
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
            system_result["perf_stat"] = perf_summary
        turbostat_path = artifact_dir / "turbostat.txt"
        if turbostat_path.is_file():
            system_result["turbostat"] = parse_turbostat(turbostat_path)
        system_result["integrity_errors"] = _system_integrity_errors(
            args,
            artifact_dir,
            system_result,
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

    result_payload: dict[str, Any] = {
        "run_id": args.run_id,
        "mode": args.mode,
        "artifact_dir": str(artifact_dir),
        "warmup": [_summarize_pass(result) for result in warmup_windows],
        "adjacent_baseline": (
            _summarize_pass(adjacent_baseline)
            if adjacent_baseline is not None
            else None
        ),
        "measured": [_summarize_pass(result) for result in measured],
        "profile_start": start_response,
        "profile_stop": stop_response,
        "system_integrity": {
            "valid": not system_result["integrity_errors"],
            "errors": system_result["integrity_errors"],
        },
    }
    if adjacent_baseline is not None and measured:
        baseline_qps = float(adjacent_baseline["speed"]["throughput_samples_per_s"])
        measured_qps = statistics.mean(
            float(result["speed"]["throughput_samples_per_s"]) for result in measured
        )
        result_payload["profile_perturbation"] = {
            "baseline_qps": baseline_qps,
            "profiled_qps": measured_qps,
            "relative_qps_change": (
                (measured_qps - baseline_qps) / baseline_qps if baseline_qps else None
            ),
            "within_5_percent": (
                abs(measured_qps - baseline_qps) / baseline_qps <= 0.05
                if baseline_qps
                else False
            ),
        }

    integrity_errors = list(system_result["integrity_errors"])
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

    if args.mode == "nsys":
        assert args.nsys_report is not None
        report_path = Path(args.nsys_report).expanduser().resolve()
        deadline = time.monotonic() + args.nsys_finalize_timeout_s
        while not report_path.is_file() and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        if not report_path.is_file():
            integrity_errors.append(f"Nsight report was not finalized at {report_path}")
        else:
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
            result_payload["nsys_report"] = {
                "path": str(report_path),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }

    write_json(artifact_dir / "result.json", result_payload)
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
        choices=["baseline", "events", "torch", "nsys"],
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default="artifacts/cpu-saturation")
    parser.add_argument("--server-pid", type=int)
    parser.add_argument(
        "--collectors",
        default="psi",
        help="Comma-separated: psi,perf-stat,perf-sched,turbostat",
    )
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
    parser.add_argument("--warmup-samples", type=int, default=128)
    parser.add_argument("--stability-windows", type=int, default=3)
    parser.add_argument("--stability-tolerance", type=float, default=0.05)
    parser.add_argument("--max-warmup-windows", type=int, default=8)
    parser.add_argument("--measure-repeats", type=int, default=1)
    parser.add_argument("--skip-adjacent-baseline", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument(
        "--allow-cache-hits",
        action="store_true",
        help="Permit pre-LM encoder cache hits in profiled event artifacts.",
    )
    parser.add_argument("--profile-timeout-s", type=float, default=180.0)
    parser.add_argument("--torch-wait-steps", type=int, default=1)
    parser.add_argument("--torch-warmup-steps", type=int, default=1)
    parser.add_argument("--torch-active-steps", type=int, default=20)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--with-flops", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--nsys-report")
    parser.add_argument("--nsys-finalize-timeout-s", type=float, default=60.0)
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
    if args.mode == "nsys" and not args.nsys_report:
        raise ValueError("--nsys-report is required in nsys mode")
    if not 0 < args.stability_tolerance < 1:
        raise ValueError("--stability-tolerance must be between 0 and 1")
    for name in (
        "warmup_samples",
        "stability_windows",
        "max_warmup_windows",
        "profile_timeout_s",
        "nsys_finalize_timeout_s",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    unknown_collectors = _requested_collectors(args) - {
        "perf-stat",
        "perf-sched",
        "turbostat",
        "psi",
    }
    if unknown_collectors:
        raise ValueError(f"unknown collectors: {sorted(unknown_collectors)}")
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
