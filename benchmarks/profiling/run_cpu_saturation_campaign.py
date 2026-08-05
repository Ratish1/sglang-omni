# SPDX-License-Identifier: Apache-2.0
"""Own a restart-balanced CPU-saturation profiling campaign.

The configuration is JSON so the exact server, interferer, environment, and
harness argument vectors are durable artifacts. Commands are argv arrays and
are never executed through a shell.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.profiling.system_collectors import (
    psi_delta,
    read_process_cgroup_psi,
    read_psi,
    write_json,
)


@dataclass(frozen=True)
class Trial:
    ordinal: int
    condition: str
    repeat: int


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("campaign config root must be an object")
    if not isinstance(config.get("server", {}).get("argv"), list):
        raise TypeError("server.argv must be an argv array")
    conditions = config.get("conditions")
    if not isinstance(conditions, list) or len(conditions) < 1:
        raise ValueError("conditions must be a non-empty array")
    names = [condition.get("name") for condition in conditions]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every condition requires a non-empty name")
    if len(set(names)) != len(names):
        raise ValueError("condition names must be unique")
    host_preflight = config.get("host_preflight")
    if host_preflight is not None:
        if not isinstance(host_preflight, dict):
            raise TypeError("host_preflight must be an object")
        limits = (
            host_preflight.get("max_cpu_psi_some_fraction"),
            host_preflight.get("max_cgroup_cpu_psi_some_fraction"),
        )
        if all(value is None for value in limits):
            raise ValueError("host_preflight requires at least one CPU PSI limit")
        for key, value in zip(
            (
                "max_cpu_psi_some_fraction",
                "max_cgroup_cpu_psi_some_fraction",
            ),
            limits,
            strict=True,
        ):
            if value is not None and not 0 <= float(value) <= 1:
                raise ValueError(f"host_preflight.{key} must be between 0 and 1")
        for key, default in (
            ("window_s", 5.0),
            ("required_consecutive_windows", 2),
            ("timeout_s", 300.0),
        ):
            if float(host_preflight.get(key, default)) <= 0:
                raise ValueError(f"host_preflight.{key} must be positive")
    return config


def build_trial_plan(
    condition_names: list[str],
    *,
    restarts_per_condition: int,
    seed: int,
) -> list[Trial]:
    """Build randomized rounds with alternating reverse order.

    For two conditions, adjacent rounds form A/B/B/A (or B/A/A/B), while each
    condition still receives exactly ``restarts_per_condition`` fresh servers.
    """
    if restarts_per_condition < 1:
        raise ValueError("restarts_per_condition must be positive")
    rng = random.Random(seed)
    counts = {name: 0 for name in condition_names}
    ordered: list[str] = []
    previous: list[str] | None = None
    for round_index in range(restarts_per_condition):
        if previous is not None and round_index % 2 == 1:
            round_names = list(reversed(previous))
        else:
            round_names = list(condition_names)
            rng.shuffle(round_names)
        ordered.extend(round_names)
        previous = round_names
    trials: list[Trial] = []
    for ordinal, name in enumerate(ordered, 1):
        counts[name] += 1
        trials.append(Trial(ordinal=ordinal, condition=name, repeat=counts[name]))
    return trials


def _merged_env(*layers: dict[str, Any] | None) -> dict[str, str]:
    env = dict(os.environ)
    for layer in layers:
        for key, value in (layer or {}).items():
            env[str(key)] = str(value)
    return env


def _start_process(
    argv: list[Any],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> tuple[subprocess.Popen[str], Any]:
    if not argv or any(not isinstance(item, (str, int, float)) for item in argv):
        raise ValueError("process argv must contain scalar values")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("x", encoding="utf-8")
    process = subprocess.Popen(
        [str(item) for item in argv],
        cwd=cwd,
        env=env,
        text=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle


def _stop_process(
    process: subprocess.Popen[str] | None,
    log_handle: Any,
    *,
    timeout_s: float,
) -> dict[str, Any] | None:
    if process is None:
        return None
    terminated = False
    killed = False
    if process.poll() is None:
        terminated = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            killed = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    if log_handle is not None:
        log_handle.flush()
        log_handle.close()
    return {
        "pid": process.pid,
        "returncode": process.returncode,
        "terminated_by_campaign": terminated,
        "killed_after_timeout": killed,
    }


def _wait_ready(
    url: str,
    process: subprocess.Popen[str],
    *,
    timeout_s: float,
    interval_s: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = "not attempted"
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(f"server exited before readiness (rc={returncode})")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
                last_error = f"HTTP {response.status}"
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(interval_s)
    raise TimeoutError(f"server did not become ready at {url}: {last_error}")


def _condition_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in config["conditions"]}


def _harness_argv(
    config: dict[str, Any],
    condition: dict[str, Any],
    *,
    run_id: str,
    output_dir: Path,
) -> list[str]:
    global_args = [str(value) for value in config.get("harness_args", [])]
    condition_args = [str(value) for value in condition.get("harness_args", [])]
    return [
        sys.executable,
        "-m",
        "benchmarks.profiling.profile_cpu_saturation",
        *global_args,
        *condition_args,
        "--run-id",
        run_id,
        "--output-dir",
        str(output_dir),
    ]


def _argv_value(argv: list[str], name: str, default: str | None = None) -> str | None:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        return default


def _is_descendant(pid: int, root_pid: int) -> bool:
    current = int(pid)
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == root_pid:
            return True
        seen.add(current)
        try:
            for line in (
                Path(f"/proc/{current}/status").read_text(encoding="utf-8").splitlines()
            ):
                if line.startswith("PPid:"):
                    current = int(line.split(":", 1)[1])
                    break
            else:
                return False
        except (OSError, ValueError):
            return False
    return False


def _resolve_stage_pid_from_server_log(
    *,
    server_pid: int,
    server_log_path: Path,
    stage: str,
) -> int:
    """Resolve and ancestry-check one stage process without profiler control."""
    pattern = re.compile(
        rf"StageGroup {re.escape(stage)}: spawned \d+ process\(es\) "
        r"\(pids=\[([^]]+)\]\)"
    )
    content = server_log_path.read_text(encoding="utf-8", errors="replace")
    matches = pattern.findall(content)
    if not matches:
        raise RuntimeError(
            f"server log does not contain a spawned PID for stage {stage!r}"
        )
    pids = [int(value.strip()) for value in matches[-1].split(",") if value.strip()]
    if len(pids) != 1:
        raise RuntimeError(
            f"stability mode requires exactly one {stage!r} stage PID; got {pids}"
        )
    target_pid = pids[0]
    if not _is_descendant(target_pid, server_pid):
        raise RuntimeError(
            f"resolved stage PID {target_pid} is not a descendant of server "
            f"PID {server_pid}"
        )
    return target_pid


def _metric(result: dict[str, Any], key: str) -> float | None:
    if result.get("mode") == "stability":
        distribution = (
            (result.get("stability_characterization") or {})
            .get("distributions", {})
            .get(key, {})
        )
        if isinstance(distribution.get("median"), (int, float)):
            return float(distribution["median"])
    reference = (result.get("adjacent_baselines") or {}).get("reference") or {}
    if isinstance(reference.get(key), (int, float)):
        return float(reference[key])
    measured = result.get("measured") or []
    values = [
        float(item[key]) for item in measured if isinstance(item.get(key), (int, float))
    ]
    return statistics.mean(values) if values else None


def _cpu_psi_fraction_between(
    before: dict[str, Any],
    after: dict[str, Any],
) -> float | None:
    delta = psi_delta(before, after)
    stall_us = ((delta.get("cpu") or {}).get("some") or {}).get("total_us")
    window_ns = delta.get("window_ns")
    if (
        not isinstance(stall_us, (int, float))
        or not isinstance(window_ns, (int, float))
        or window_ns <= 0
    ):
        return None
    return float(stall_us) * 1000.0 / float(window_ns)


def _read_ambient_pressure() -> dict[str, Any]:
    return {
        "global": read_psi(),
        "cgroup": read_process_cgroup_psi(os.getpid()),
    }


def _wait_for_ambient_cpu_psi(
    config: dict[str, Any] | None,
    *,
    output_path: Path,
    snapshot_reader: Any = None,
    sleep_fn: Any = None,
    monotonic_fn: Any = None,
) -> dict[str, Any]:
    """Require quiet execution windows before starting trial processes."""
    if not config:
        report = {"enabled": False, "accepted": True, "observations": []}
        write_json(output_path, report)
        return report

    snapshot_reader = snapshot_reader or _read_ambient_pressure
    sleep_fn = sleep_fn or time.sleep
    monotonic_fn = monotonic_fn or time.monotonic
    global_limit = config.get("max_cpu_psi_some_fraction")
    cgroup_limit = config.get("max_cgroup_cpu_psi_some_fraction")
    limits = {
        "global": float(global_limit) if global_limit is not None else None,
        "cgroup": float(cgroup_limit) if cgroup_limit is not None else None,
    }
    window_s = float(config.get("window_s", 5.0))
    required = int(config.get("required_consecutive_windows", 2))
    timeout_s = float(config.get("timeout_s", 300.0))
    if all(limit is None for limit in limits.values()):
        raise ValueError("host_preflight requires at least one CPU PSI limit")
    if any(limit is not None and not 0 <= limit <= 1 for limit in limits.values()):
        raise ValueError("host preflight CPU PSI limits must be between 0 and 1")
    if window_s <= 0 or required <= 0 or timeout_s <= 0:
        raise ValueError("host preflight durations and window count must be positive")

    started = monotonic_fn()
    consecutive = 0
    observations: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "enabled": True,
        "accepted": False,
        "max_cpu_psi_some_fraction": limits["global"],
        "max_cgroup_cpu_psi_some_fraction": limits["cgroup"],
        "window_s": window_s,
        "required_consecutive_windows": required,
        "timeout_s": timeout_s,
        "observations": observations,
    }
    while monotonic_fn() - started + window_s <= timeout_s:
        before = snapshot_reader()
        sleep_fn(window_s)
        after = snapshot_reader()
        fractions = {
            scope: (
                _cpu_psi_fraction_between(before[scope], after[scope])
                if isinstance(before.get(scope), dict)
                and isinstance(after.get(scope), dict)
                and "error" not in before[scope]
                and "error" not in after[scope]
                else None
            )
            for scope in ("global", "cgroup")
        }
        accepted_window = all(
            limit is None
            or (fractions[scope] is not None and fractions[scope] <= limit)
            for scope, limit in limits.items()
        )
        consecutive = consecutive + 1 if accepted_window else 0
        observations.append(
            {
                "window": len(observations) + 1,
                "cpu_psi_some_fraction": fractions["global"],
                "cgroup_cpu_psi_some_fraction": fractions["cgroup"],
                "within_limit": accepted_window,
                "consecutive_within_limit": consecutive,
                "before": before,
                "after": after,
            }
        )
        report["elapsed_s"] = monotonic_fn() - started
        report["last_cpu_psi_some_fraction"] = fractions["global"]
        report["last_cgroup_cpu_psi_some_fraction"] = fractions["cgroup"]
        report["accepted"] = consecutive >= required
        write_json(output_path, report)
        if report["accepted"]:
            return report

    report["elapsed_s"] = monotonic_fn() - started
    write_json(output_path, report)
    last_global = report.get("last_cpu_psi_some_fraction")
    last_cgroup = report.get("last_cgroup_cpu_psi_some_fraction")
    global_text = "unavailable" if last_global is None else f"{float(last_global):.4f}"
    cgroup_text = "unavailable" if last_cgroup is None else f"{float(last_cgroup):.4f}"
    raise RuntimeError(
        "ambient CPU PSI preflight timed out: "
        f"last global={global_text}, last cgroup={cgroup_text}, "
        f"required consecutive windows={required}"
    )


def _bootstrap_median_ci(
    values: list[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    boot = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(samples)
    )
    return (
        boot[math.floor(0.025 * (samples - 1))],
        boot[math.ceil(0.975 * (samples - 1))],
    )


def _distribution(values: list[float], *, seed: int) -> dict[str, Any]:
    ci = _bootstrap_median_ci(values, seed=seed)
    center = statistics.median(values) if values else None
    return {
        "values": values,
        "median": center,
        "mean": statistics.mean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mad": (
            statistics.median(abs(value - center) for value in values)
            if values and center is not None
            else None
        ),
        "bootstrap_median_ci95": list(ci) if ci else None,
    }


def _causal_run_metrics(result: dict[str, Any]) -> dict[str, float]:
    artifact_dir = Path(result["artifact_dir"])
    report_path = artifact_dir / "event_report.json"
    metrics: dict[str, float] = {}
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        for row in report.get("stage_breakdown", []):
            stage = str(row.get("stage", "unknown"))
            interval = str(row.get("interval", "unknown"))
            for statistic in ("avg_ms", "p95_ms"):
                value = row.get(statistic)
                if isinstance(value, (int, float)):
                    metrics[f"interval|{stage}|{interval}|{statistic}"] = float(value)
        causal_state = report.get("causal_state") or {}
        queue_wait = causal_state.get("pre_lm_queue_wait_ms") or {}
        for statistic in ("p50", "p95", "max"):
            value = queue_wait.get(statistic)
            if isinstance(value, (int, float)):
                metrics[f"pre_lm_queue_wait_ms|{statistic}"] = float(value)
        for name, value in (causal_state.get("maxima") or {}).items():
            if isinstance(value, (int, float)):
                metrics[f"max_observed|{name}"] = float(value)
        for name, value in (report.get("event_counts") or {}).items():
            if isinstance(value, (int, float)) and name in {
                "scheduler_request_build_hol_start",
                "request_build_admission_rejected",
                "pre_lm_batch_start",
            }:
                metrics[f"event_count|{name}"] = float(value)

    system_path = artifact_dir / "system.json"
    if system_path.is_file():
        with system_path.open("r", encoding="utf-8") as handle:
            system = json.load(handle)
        for name, value in (
            (system.get("perf_stat") or {}).get("derived") or {}
        ).items():
            if isinstance(value, (int, float)):
                metrics[f"perf|{name}"] = float(value)
        gpu_columns = (system.get("gpu_dmon") or {}).get("columns") or {}
        for column in ("sm", "pwr", "mclk", "pclk"):
            for statistic in ("mean", "zero_fraction"):
                value = (gpu_columns.get(column) or {}).get(statistic)
                if isinstance(value, (int, float)):
                    metrics[f"gpu|{column}|{statistic}"] = float(value)
        cpu_frequency = system.get("cpu_frequency") or {}
        for name, value in (
            (
                "frequency_mhz_mean",
                (cpu_frequency.get("frequency_mhz") or {}).get("mean"),
            ),
            (
                "busy_weighted_frequency_mhz",
                cpu_frequency.get("busy_weighted_frequency_mhz"),
            ),
        ):
            if isinstance(value, (int, float)):
                metrics[f"cpu|{name}"] = float(value)
        if result.get("mode") != "stability":
            by_thread_name: dict[str, dict[str, float]] = {}
            for thread in (system.get("thread_summary") or {}).get("threads", []):
                name = str(thread.get("comm") or "unknown")
                row = by_thread_name.setdefault(
                    name,
                    {"cpu_ms_per_request": 0.0, "runqueue_ms_per_request": 0.0},
                )
                cpu = thread.get("cpu_ms_per_completed_request")
                delay = thread.get("runqueue_delay_ms_per_completed_request")
                if isinstance(cpu, (int, float)):
                    row["cpu_ms_per_request"] += float(cpu)
                if isinstance(delay, (int, float)):
                    row["runqueue_ms_per_request"] += float(delay)
            for name, row in by_thread_name.items():
                for statistic, value in row.items():
                    metrics[f"threads|{name}|{statistic}"] = value
        cpu_psi = ((system.get("psi_delta") or {}).get("cpu") or {}).get("some") or {}
        stall_us = cpu_psi.get("total_us")
        window_ns = (system.get("psi_delta") or {}).get("window_ns")
        if isinstance(stall_us, (int, float)):
            metrics["psi|cpu_some_stall_ms"] = float(stall_us) / 1000.0
        if (
            isinstance(stall_us, (int, float))
            and isinstance(window_ns, (int, float))
            and window_ns > 0
        ):
            metrics["psi|cpu_some_stall_fraction"] = (
                float(stall_us) * 1000.0 / float(window_ns)
            )
    if result.get("mode") == "stability":
        window_metrics: dict[str, list[float]] = {}
        for window in result.get("stability_characterization", {}).get("windows") or []:
            system_artifact = Path(window["system_artifact"])
            with system_artifact.open("r", encoding="utf-8") as handle:
                window_system = json.load(handle)
            summary = window.get("summary") or {}
            completed = int(
                (summary.get("request_accounting") or {}).get("completed", 0)
            )
            pressure = window_system.get("pressure") or {}
            for name in (
                "cpu_psi_some_fraction",
                "cgroup_cpu_psi_some_fraction",
            ):
                value = pressure.get(name)
                if isinstance(value, (int, float)):
                    window_metrics.setdefault(f"window|{name}", []).append(float(value))
            thread_delta = window_system.get("thread_delta") or {}
            for name in ("cpu_ms", "runqueue_delay_ms", "migrations"):
                value = thread_delta.get(name)
                if isinstance(value, (int, float)) and completed:
                    window_metrics.setdefault(
                        f"window|threads|{name}_per_request",
                        [],
                    ).append(float(value) / completed)
            by_name: dict[str, dict[str, float]] = {}
            for thread in thread_delta.get("threads", []):
                name = str(thread.get("comm") or "unknown")
                row = by_name.setdefault(
                    name, {"cpu_ms": 0.0, "runqueue_delay_ms": 0.0}
                )
                for metric_name in row:
                    value = thread.get(metric_name)
                    if isinstance(value, (int, float)):
                        row[metric_name] += float(value)
            for name, row in by_name.items():
                if completed:
                    for metric_name, value in row.items():
                        window_metrics.setdefault(
                            f"window|thread|{name}|{metric_name}_per_request",
                            [],
                        ).append(value / completed)
        for name, values in window_metrics.items():
            if values:
                metrics[name] = statistics.median(values)
    return metrics


def _aggregate(
    campaign_dir: Path,
    trials: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        if trial.get("status") != "completed":
            continue
        result_path = Path(trial["result_path"])
        with result_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        grouped.setdefault(trial["condition"], []).append(result)

    metrics = (
        "throughput_samples_per_s",
        "latency_p50_s",
        "latency_p95_s",
        "latency_p99_s",
    )
    conditions: dict[str, Any] = {}
    for condition, results in grouped.items():
        condition_result: dict[str, Any] = {
            "runs": len(results),
            "metrics": {},
            "causal_metrics": {},
            "window_metrics": {},
        }
        for metric in metrics:
            values = [
                value
                for result in results
                if (value := _metric(result, metric)) is not None
            ]
            condition_result["metrics"][metric] = _distribution(values, seed=seed)
            window_values = [
                float(window["summary"][metric])
                for result in results
                if result.get("mode") == "stability"
                for window in (
                    result.get("stability_characterization", {}).get("windows") or []
                )
                if isinstance(window.get("summary", {}).get(metric), (int, float))
            ]
            if window_values:
                condition_result["window_metrics"][metric] = _distribution(
                    window_values,
                    seed=seed,
                )
        causal_runs = [_causal_run_metrics(result) for result in results]
        causal_keys = sorted({key for run in causal_runs for key in run})
        for key in causal_keys:
            condition_result["causal_metrics"][key] = _distribution(
                [run[key] for run in causal_runs if key in run],
                seed=seed,
            )
        conditions[condition] = condition_result

    reference_name = "quiet" if "quiet" in conditions else next(iter(conditions), None)
    comparisons: dict[str, Any] = {}
    if reference_name is not None:
        reference = conditions[reference_name]
        for condition_name, condition in conditions.items():
            if condition_name == reference_name:
                continue
            deltas: dict[str, Any] = {}
            for group_name in ("metrics", "causal_metrics"):
                reference_group = reference[group_name]
                condition_group = condition[group_name]
                for metric_name in sorted(set(reference_group) & set(condition_group)):
                    baseline = reference_group[metric_name]["median"]
                    observed = condition_group[metric_name]["median"]
                    if baseline is None or observed is None:
                        continue
                    deltas[f"{group_name}|{metric_name}"] = {
                        "reference_median": baseline,
                        "condition_median": observed,
                        "absolute_delta": observed - baseline,
                        "relative_delta": (
                            (observed - baseline) / baseline if baseline else None
                        ),
                    }
            comparisons[f"{condition_name}_vs_{reference_name}"] = deltas
    summary = {
        "campaign_dir": str(campaign_dir),
        "performance_source": (
            "unprofiled_stability_windows"
            if any(
                result.get("mode") == "stability"
                for results in grouped.values()
                for result in results
            )
            else "bracketed_unprofiled_reference"
        ),
        "reference_condition": reference_name,
        "conditions": conditions,
        "comparisons": comparisons,
        "completed_trials": sum(
            1 for trial in trials if trial.get("status") == "completed"
        ),
        "failed_trials": sum(1 for trial in trials if trial.get("status") == "failed"),
    }
    write_json(campaign_dir / "summary.json", summary)
    return summary


def run_campaign(config: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "campaign_config.json", config)
    protocol = config.get("protocol") or {}
    restarts = int(protocol.get("restarts_per_condition", 5))
    seed = int(protocol.get("random_seed", 20260805))
    conditions = _condition_map(config)
    plan = build_trial_plan(
        list(conditions),
        restarts_per_condition=restarts,
        seed=seed,
    )
    write_json(
        output_dir / "trial_plan.json",
        [trial.__dict__ for trial in plan],
    )

    server_cfg = config["server"]
    repo_root = Path(config.get("repo_root") or Path.cwd()).resolve()
    ready_url = str(server_cfg.get("ready_url", "http://127.0.0.1:8000/health"))
    ready_timeout_s = float(server_cfg.get("ready_timeout_s", 900))
    stop_timeout_s = float(server_cfg.get("stop_timeout_s", 60))
    continue_on_failure = bool(protocol.get("continue_on_failure", False))
    trial_results: list[dict[str, Any]] = []

    for trial in plan:
        condition = conditions[trial.condition]
        run_id = (
            f"{output_dir.name}-{trial.ordinal:03d}-{trial.condition}-r{trial.repeat}"
        )
        trial_dir = output_dir / "trials" / run_id
        trial_dir.mkdir(parents=True, exist_ok=False)
        server: subprocess.Popen[str] | None = None
        server_log = None
        interferer: subprocess.Popen[str] | None = None
        interferer_log = None
        record: dict[str, Any] = {
            **trial.__dict__,
            "run_id": run_id,
            "status": "starting",
            "started_wall_ns": time.time_ns(),
        }
        write_json(trial_dir / "condition.json", condition)
        try:
            record["host_preflight"] = _wait_for_ambient_cpu_psi(
                config.get("host_preflight"),
                output_path=trial_dir / "host_preflight.json",
            )
            server_argv = [
                *[str(value) for value in condition.get("server_prefix_argv", [])],
                *[str(value) for value in server_cfg["argv"]],
                *[str(value) for value in condition.get("server_argv_append", [])],
            ]
            server_env = _merged_env(
                server_cfg.get("env"),
                condition.get("server_env"),
            )
            server, server_log = _start_process(
                server_argv,
                cwd=repo_root,
                env=server_env,
                log_path=trial_dir / "server.log",
            )
            _wait_ready(
                ready_url,
                server,
                timeout_s=ready_timeout_s,
            )

            interferer_argv = condition.get("interferer_argv")
            if interferer_argv:
                if not isinstance(interferer_argv, list):
                    raise ValueError("condition.interferer_argv must be an array")
                interferer, interferer_log = _start_process(
                    interferer_argv,
                    cwd=repo_root,
                    env=_merged_env(condition.get("interferer_env")),
                    log_path=trial_dir / "interferer.log",
                )
                time.sleep(float(condition.get("interferer_settle_s", 2.0)))
                if interferer.poll() is not None:
                    raise RuntimeError(
                        f"interferer exited during settle (rc={interferer.returncode})"
                    )

            harness_argv = _harness_argv(
                config,
                condition,
                run_id=run_id,
                output_dir=output_dir / "artifacts",
            )
            target_stage_pid = None
            if (
                _argv_value(harness_argv, "--mode") == "stability"
                and "--server-pid" not in harness_argv
            ):
                target_stage_pid = _resolve_stage_pid_from_server_log(
                    server_pid=server.pid,
                    server_log_path=trial_dir / "server.log",
                    stage=str(_argv_value(harness_argv, "--stage", "asr")),
                )
                harness_argv.extend(["--server-pid", str(target_stage_pid)])
            write_json(
                trial_dir / "launch.json",
                {
                    "server_argv": server_argv,
                    "server_pid": server.pid,
                    "interferer_argv": interferer_argv,
                    "interferer_pid": interferer.pid if interferer else None,
                    "target_stage_pid": target_stage_pid,
                    "harness_argv": harness_argv,
                },
            )
            with (trial_dir / "harness.log").open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    harness_argv,
                    cwd=repo_root,
                    env=_merged_env(config.get("harness_env")),
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"profiling harness failed rc={completed.returncode}"
                )
            result_path = output_dir / "artifacts" / run_id / "result.json"
            if not result_path.is_file():
                raise RuntimeError(f"harness did not finalize {result_path}")
            record.update(
                {
                    "status": "completed",
                    "result_path": str(result_path),
                }
            )
        except Exception as exc:  # noqa: BLE001 - persist and isolate trial failure
            record.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            record["interferer_stop"] = _stop_process(
                interferer,
                interferer_log,
                timeout_s=stop_timeout_s,
            )
            record["server_stop"] = _stop_process(
                server,
                server_log,
                timeout_s=stop_timeout_s,
            )
            record["stopped_wall_ns"] = time.time_ns()
            write_json(trial_dir / "trial_result.json", record)
            trial_results.append(record)
        if record["status"] == "failed" and not continue_on_failure:
            break

    write_json(output_dir / "trials.json", trial_results)
    return _aggregate(output_dir, trial_results, seed=seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary = run_campaign(_load_config(config_path), output_dir=output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed_trials"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
