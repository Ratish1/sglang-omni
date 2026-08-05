# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.profiling.cpu_interferer import parse_cpu_list
from benchmarks.profiling.run_cpu_saturation_campaign import (
    _bootstrap_median_ci,
    _causal_run_metrics,
    _cpu_psi_fraction_between,
    _finalized_result_errors,
    _harness_argv,
    _metric,
    _process_placement_snapshot,
    _resolve_stage_pid_from_server_log,
    _wait_for_ambient_cpu_psi,
    build_trial_plan,
)


def _arg_value(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_two_condition_plan_forms_abba_and_balances_restarts() -> None:
    plan = build_trial_plan(
        ["quiet", "stress"],
        restarts_per_condition=5,
        seed=7,
    )
    names = [trial.condition for trial in plan]
    assert len(names) == 10
    assert names.count("quiet") == 5
    assert names.count("stress") == 5
    assert names[:4] in (
        ["quiet", "stress", "stress", "quiet"],
        ["stress", "quiet", "quiet", "stress"],
    )
    assert [trial.ordinal for trial in plan] == list(range(1, 11))
    assert [trial.repeat for trial in plan if trial.condition == "quiet"] == [
        1,
        2,
        3,
        4,
        5,
    ]


def test_bootstrap_median_interval_is_seeded_and_contains_center() -> None:
    values = [50.0, 51.0, 52.0, 53.0, 54.0]
    first = _bootstrap_median_ci(values, seed=9, samples=1000)
    second = _bootstrap_median_ci(values, seed=9, samples=1000)
    assert first == second
    assert first is not None
    assert first[0] <= 52.0 <= first[1]


def test_campaign_does_not_misidentify_coordinator_as_stage_pid(tmp_path) -> None:
    argv = _harness_argv(
        {"harness_args": ["--mode", "events"]},
        {"harness_args": ["--concurrency", 64]},
        run_id="trial-1",
        output_dir=tmp_path,
    )
    assert "--server-pid" not in argv
    assert argv[-4:] == ["--run-id", "trial-1", "--output-dir", str(tmp_path)]


def test_stability_campaign_resolves_stage_pid_without_profiler_control(
    tmp_path: Path,
) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "StageGroup asr: spawned 1 process(es) " f"(pids=[{os.getpid()}])\n",
        encoding="utf-8",
    )
    assert (
        _resolve_stage_pid_from_server_log(
            server_pid=os.getpid(),
            server_log_path=server_log,
            stage="asr",
        )
        == os.getpid()
    )


def test_h100_campaign_separates_request_concurrency_from_cpu_stress() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (
            repo_root / "benchmarks/profiling/campaign.events.h100.example.json"
        ).read_text(encoding="utf-8")
    )
    conditions = {item["name"]: item for item in config["conditions"]}

    assert _arg_value(config["harness_args"], "--concurrency") == "32"
    assert "--max-queued-requests" not in config["server"]["argv"]
    assert (
        _arg_value(conditions["unbound-cpu64"]["interferer_argv"], "--workers") == "64"
    )
    assert "harness_args" not in conditions["quiet"]
    assert config["host_preflight"] == {
        "max_cgroup_cpu_psi_some_fraction": 0.02,
        "window_s": 5,
        "required_consecutive_windows": 2,
        "timeout_s": 300,
    }


def test_stability_campaign_is_unprofiled_and_retains_twenty_windows() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = json.loads(
        (
            repo_root / "benchmarks/profiling/campaign.stability.h100.example.json"
        ).read_text(encoding="utf-8")
    )
    assert _arg_value(config["harness_args"], "--mode") == "stability"
    assert _arg_value(config["harness_args"], "--characterization-windows") == "20"
    assert (
        _arg_value(
            config["harness_args"],
            "--max-thread-cpu-accounting-error",
        )
        == "0.05"
    )
    assert "events" not in config["harness_args"]
    assert config["protocol"]["continue_on_failure"] is True


def test_campaign_performance_uses_bracketed_unprofiled_reference() -> None:
    result = {
        "adjacent_baselines": {"reference": {"throughput_samples_per_s": 30.0}},
        "measured": [{"throughput_samples_per_s": 34.0}],
    }
    assert _metric(result, "throughput_samples_per_s") == 30.0


def test_stability_campaign_uses_window_median() -> None:
    result = {
        "mode": "stability",
        "stability_characterization": {
            "distributions": {
                "throughput_samples_per_s": {
                    "mean": 35.0,
                    "median": 30.0,
                }
            }
        },
        "measured": [{"throughput_samples_per_s": 100.0}],
    }
    assert _metric(result, "throughput_samples_per_s") == 30.0


def test_rejected_stability_result_is_not_eligible_for_aggregation() -> None:
    assert _finalized_result_errors(
        {
            "mode": "stability",
            "accepted": False,
            "integrity_errors": ["window 3 had a rejected request"],
        }
    ) == ["window 3 had a rejected request"]
    assert _finalized_result_errors({"mode": "stability", "accepted": True}) == []


def test_process_placement_snapshot_records_affinity_and_cgroup() -> None:
    if not Path("/proc/self/status").is_file():
        pytest.skip("requires Linux procfs")
    snapshot = _process_placement_snapshot(os.getpid())
    assert snapshot is not None
    root = next(row for row in snapshot["processes"] if row["pid"] == os.getpid())
    assert root["cpus_allowed_list"]
    assert root["task_affinities"]
    assert root["cgroup"] is not None


def _psi_snapshot(monotonic_ns: int, total_us: int) -> dict:
    return {
        "captured_monotonic_ns": monotonic_ns,
        "cpu": {"some": {"total": total_us, "avg10": 0.0}},
        "memory": {},
        "io": {},
    }


def test_cpu_psi_fraction_uses_window_delta() -> None:
    assert _cpu_psi_fraction_between(
        _psi_snapshot(0, 100),
        _psi_snapshot(1_000_000_000, 20_100),
    ) == pytest.approx(0.02)


def test_host_preflight_requires_consecutive_quiet_windows(tmp_path) -> None:
    # One quiet window is reset by a noisy window before two quiet windows pass.
    snapshots = iter(
        [
            _psi_snapshot(0, 0),
            _psi_snapshot(1_000_000_000, 10_000),
            _psi_snapshot(1_000_000_000, 10_000),
            _psi_snapshot(2_000_000_000, 110_000),
            _psi_snapshot(2_000_000_000, 110_000),
            _psi_snapshot(3_000_000_000, 120_000),
            _psi_snapshot(3_000_000_000, 120_000),
            _psi_snapshot(4_000_000_000, 130_000),
        ]
    )
    report = _wait_for_ambient_cpu_psi(
        {
            "max_cgroup_cpu_psi_some_fraction": 0.02,
            "window_s": 1,
            "required_consecutive_windows": 2,
            "timeout_s": 10,
        },
        output_path=tmp_path / "host_preflight.json",
        snapshot_reader=lambda: {
            "global": _psi_snapshot(0, 0),
            "cgroup": next(snapshots),
        },
        sleep_fn=lambda _seconds: None,
        monotonic_fn=lambda: 0.0,
    )
    assert report["accepted"] is True
    assert [row["within_limit"] for row in report["observations"]] == [
        True,
        False,
        True,
        True,
    ]
    assert json.loads((tmp_path / "host_preflight.json").read_text())["accepted"]


def test_interferer_cpu_list_parser_rejects_ambiguous_placement() -> None:
    assert parse_cpu_list("0-2,8") == [0, 1, 2, 8]
    with pytest.raises(ValueError, match="duplicates"):
        parse_cpu_list("0-2,2")


def test_campaign_extracts_first_diverging_causal_boundaries(tmp_path) -> None:
    report = {
        "stage_breakdown": [
            {
                "stage": "asr",
                "interval": "pre_lm_enqueue->pre_lm_dequeue",
                "avg_ms": 12.5,
                "p95_ms": 20.0,
            }
        ],
        "event_counts": {
            "scheduler_request_build_hol_start": 3,
            "unrelated": 99,
        },
        "causal_state": {
            "maxima": {"pending_builds": 16},
            "pre_lm_queue_wait_ms": {"p50": 10.0, "p95": 20.0, "max": 24.0},
        },
    }
    (tmp_path / "event_report.json").write_text(json.dumps(report), encoding="utf-8")
    metrics = _causal_run_metrics({"artifact_dir": str(tmp_path)})
    assert metrics["interval|asr|pre_lm_enqueue->pre_lm_dequeue|avg_ms"] == 12.5
    assert metrics["event_count|scheduler_request_build_hol_start"] == 3
    assert metrics["max_observed|pending_builds"] == 16
    assert "event_count|unrelated" not in metrics
