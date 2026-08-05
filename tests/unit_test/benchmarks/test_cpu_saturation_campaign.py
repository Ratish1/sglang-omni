# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.profiling.cpu_interferer import parse_cpu_list
from benchmarks.profiling.run_cpu_saturation_campaign import (
    _bootstrap_median_ci,
    _causal_run_metrics,
    _harness_argv,
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
