# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchmarks.profiling.cpu_interferer import parse_cpu_list
from benchmarks.profiling.run_cpu_saturation_campaign import (
    _aggregate,
    _bootstrap_median_ci,
    _causal_run_metrics,
    _cpu_psi_fraction_between,
    _finalized_result_errors,
    _harness_argv,
    _load_config,
    _metric,
    _process_placement_snapshot,
    _resolve_stage_pid_from_server_log,
    _server_log_contract,
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
    assert (
        _arg_value(config["harness_args"], "--required-thread-comms")
        == "sched-asr,omni-request-bu,fun-asr-audio-e"
    )
    assert "events" not in config["harness_args"]
    assert config["protocol"]["continue_on_failure"] is True


def test_encoder_cuda_graph_campaign_is_a_four_arm_unprofiled_ab() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = _load_config(
        repo_root
        / "benchmarks/profiling/campaign.encoder_cuda_graph_ab.h100.example.json"
    )
    conditions = {item["name"]: item for item in config["conditions"]}

    assert _arg_value(config["harness_args"], "--mode") == "stability"
    assert config["reference_condition"] == "eager-quiet"
    assert set(conditions) == {
        "eager-quiet",
        "graph-quiet",
        "eager-cpu64",
        "graph-cpu64",
    }
    for name in ("eager-quiet", "eager-cpu64"):
        assert conditions[name]["server_argv_append"] == [
            "--stages.asr.factory_args.enable_encoder_cuda_graph=false"
        ]
    for name in ("graph-quiet", "graph-cpu64"):
        assert conditions[name]["server_argv_append"] == [
            "--stages.asr.factory_args.enable_encoder_cuda_graph=true"
        ]
        assert (
            "Captured Fun-ASR encoder CUDA graph"
            in conditions[name]["required_server_log_substrings"]
        )
        assert conditions[name]["forbidden_server_log_substrings"]
    assert {pair["name"] for pair in config["comparison_pairs"]} == {
        "graph_vs_eager_quiet",
        "cpu64_vs_quiet_eager",
        "graph_vs_eager_cpu64",
        "cpu64_vs_quiet_graph",
    }


def test_server_log_contract_requires_graph_capture_and_rejects_fallback(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "Fun-ASR encoder CUDA graphs enabled\n"
        "Captured Fun-ASR encoder CUDA graph batch=8 t=512\n",
        encoding="utf-8",
    )
    condition = {
        "required_server_log_substrings": [
            "Fun-ASR encoder CUDA graphs enabled",
            "Captured Fun-ASR encoder CUDA graph",
        ],
        "forbidden_server_log_substrings": ["capture failed"],
    }
    assert _server_log_contract(log_path, condition)["valid"] is True

    log_path.write_text(
        log_path.read_text(encoding="utf-8") + "capture failed\n",
        encoding="utf-8",
    )
    report = _server_log_contract(log_path, condition)
    assert report["valid"] is False
    assert report["present_forbidden"] == ["capture failed"]


def test_campaign_emits_explicit_graph_and_stress_comparisons(tmp_path: Path) -> None:
    trials = []
    for condition, qps in (
        ("eager-quiet", 50.0),
        ("graph-quiet", 100.0),
        ("eager-cpu64", 30.0),
        ("graph-cpu64", 80.0),
    ):
        result_path = tmp_path / f"{condition}.json"
        result_path.write_text(
            json.dumps(
                {
                    "artifact_dir": str(tmp_path),
                    "adjacent_baselines": {
                        "reference": {
                            "throughput_samples_per_s": qps,
                            "corpus_wer": 0.0172,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        trials.append(
            {
                "status": "completed",
                "condition": condition,
                "result_path": str(result_path),
            }
        )

    summary = _aggregate(
        tmp_path,
        trials,
        seed=1,
        reference_condition="eager-quiet",
        comparison_pairs=[
            {
                "name": "graph_vs_eager_cpu64",
                "reference": "eager-cpu64",
                "condition": "graph-cpu64",
            },
            {
                "name": "cpu64_vs_quiet_graph",
                "reference": "graph-quiet",
                "condition": "graph-cpu64",
            },
        ],
    )
    qps_metric = "metrics|throughput_samples_per_s"
    assert summary["conditions"]["graph-cpu64"]["metrics"]["corpus_wer"][
        "median"
    ] == pytest.approx(0.0172)
    assert summary["comparisons"]["graph_vs_eager_cpu64"][qps_metric][
        "relative_delta"
    ] == pytest.approx(5 / 3)
    assert summary["comparisons"]["cpu64_vs_quiet_graph"][qps_metric][
        "relative_delta"
    ] == pytest.approx(-0.2)


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
