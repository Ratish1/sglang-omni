# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.same_gpu_dp.summarize import (
    classify_kv_capacity,
    extract_kv_tokens,
    parse_cpu_set,
    summarize_matrix,
    summarize_results,
    summarize_router_snapshot,
    validate_all_requests_succeeded,
    validate_layout,
)


def test_benchmark_start_barrier_waits_for_release(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    runner = BenchmarkRunner(
        RunConfig(
            barrier_ready_file=str(ready),
            barrier_release_file=str(release),
            barrier_timeout_s=1,
        )
    )

    async def exercise() -> None:
        waiting = asyncio.create_task(runner._wait_for_barrier())
        while not ready.exists():
            await asyncio.sleep(0.01)
        assert not waiting.done()
        release.touch()
        await waiting

    asyncio.run(exercise())


def test_condition_launches_barrier_clients_as_direct_children(tmp_path: Path) -> None:
    script = Path("benchmarks/same_gpu_dp/run_condition.sh").resolve()
    env = {
        **os.environ,
        "OUT_ROOT": str(tmp_path),
        "LABEL": "client-pid-dry-run",
        "DP": "2",
        "SERVER_CORE_SETS": "0;1",
        "CLIENT_CORE_SETS": "2;3",
        "MEM_FRACTIONS": "auto,auto",
    }

    subprocess.run(["bash", str(script), "--dry-run"], env=env, check=True)

    commands = (tmp_path / "client-pid-dry-run" / "commands.sh").read_text(
        encoding="utf-8"
    )
    clients = [
        line for line in commands.splitlines() if "benchmark_tts_seedtts" in line
    ]
    assert len(clients) == 2
    assert all(line.startswith("numactl ") for line in clients)


def test_classify_kv_capacity_requires_the_configured_cap() -> None:
    token_counts = {"worker_0": 78015, "worker_1": 78015}

    assert classify_kv_capacity(token_counts, 78015) == "exact"
    assert classify_kv_capacity(token_counts, 78021) == "configured_mismatch"


def _run_fake_capacity_search(
    tmp_path: Path,
    runner_source: str,
    *,
    mem_fractions: str = "auto,auto",
    check: bool = True,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    runner = tmp_path / "fake_condition.sh"
    runner.write_text(runner_source, encoding="utf-8")
    runner.chmod(0o755)
    root = tmp_path / "capacity"
    script = Path("benchmarks/same_gpu_dp/calibrate_capacity.sh").resolve()
    env = {
        **os.environ,
        "CONDITION_RUNNER": str(runner),
        "CALIBRATION_DPS": "2",
        "CALIBRATION_MPS_MODES": "1",
        "CALIBRATION_CONFIRMATIONS": "2",
        "CALIBRATION_TOKEN_TOLERANCE": "1",
        "CALIBRATION_ROOT": str(root),
        "DP2_SERVER_CORE_SETS": "0;1",
        "DP2_CLIENT_CORE_SETS": "2;3",
        "DP2_MEM_FRACTIONS": mem_fractions,
        "DP2_INITIAL_CAP_TOKENS": "20",
    }
    result = subprocess.run(
        ["bash", str(script)],
        env=env,
        check=check,
        text=True,
    )
    return result, root


def test_capacity_search_finds_highest_equal_cap(tmp_path: Path) -> None:
    _, root = _run_fake_capacity_search(
        tmp_path,
        """#!/usr/bin/env bash
set -eu
out="$OUT_ROOT/$LABEL"
mkdir -p "$out"
if ((MAX_TOTAL_TOKENS <= 100)); then
  resolved=$MAX_TOTAL_TOKENS
  status=0
else
  resolved=100
  status=1
fi
printf '{"worker_0": %s, "worker_1": %s}\n' "$resolved" "$resolved" \
  > "$out/kv_capacity.json"
exit "$status"
""",
        mem_fractions="0.90,0.90",
    )

    selection = (root / "capacity_selection.tsv").read_text(encoding="utf-8")
    assert "2\t0.90,0.90\t100\t101\t1\t0\t100" in selection
    assert "DP2_MAX_TOTAL_TOKENS=100" in (root / "capacity.env").read_text(
        encoding="utf-8"
    )
    assert "capacity-limit" in (root / "capacity_trials.tsv").read_text(
        encoding="utf-8"
    )


def test_capacity_search_uses_startup_oom_as_failing_bound(tmp_path: Path) -> None:
    _, root = _run_fake_capacity_search(
        tmp_path,
        """#!/usr/bin/env bash
set -eu
out="$OUT_ROOT/$LABEL"
mkdir -p "$out/workers"
if ((MAX_TOTAL_TOKENS <= 100)); then
  printf '{"worker_0": %s, "worker_1": %s}\n' \
    "$MAX_TOTAL_TOKENS" "$MAX_TOTAL_TOKENS" > "$out/kv_capacity.json"
  exit 0
fi
printf 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.00 MiB.\n' \
  > "$out/workers/worker_1.server.log"
exit 1
""",
    )

    selection = (root / "capacity_selection.tsv").read_text(encoding="utf-8")
    assert "2\tauto,auto\t100\t101\t1\t0\t100" in selection
    trials = (root / "capacity_trials.tsv").read_text(encoding="utf-8")
    assert "capacity-limit\tstartup-oom" in trials


def test_capacity_search_keeps_non_oom_startup_failure_distinct(
    tmp_path: Path,
) -> None:
    result, root = _run_fake_capacity_search(
        tmp_path,
        """#!/usr/bin/env bash
set -eu
out="$OUT_ROOT/$LABEL"
mkdir -p "$out/workers"
printf 'worker exited before readiness\n' > "$out/workers/worker_1.server.log"
exit 1
""",
        check=False,
    )

    assert result.returncode != 0
    trials = (root / "capacity_trials.tsv").read_text(encoding="utf-8")
    assert "infrastructure-failure\tmissing" in trials


def test_parse_cpu_set_expands_linux_syntax() -> None:
    assert parse_cpu_set("0-2,5,7-8") == {0, 1, 2, 5, 7, 8}


@pytest.mark.parametrize("value", ["", "3-1", "1,,2", "cpu3"])
def test_parse_cpu_set_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_cpu_set(value)


def test_validate_layout_rejects_cross_role_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        validate_layout(2, ["0-3", "4-7"], ["8-9", "7-10"])


def test_validate_layout_counts_fixed_budgets_and_router() -> None:
    result = validate_layout(
        2,
        ["0-3", "4-7"],
        ["8-9", "10-11"],
        online_cpus=set(range(16)),
        extra_core_sets=(("router", "12-13"),),
    )
    assert result["server_cpu_count"] == 8
    assert result["client_cpu_count"] == 4


def test_validate_layout_accepts_dp10() -> None:
    result = validate_layout(
        10,
        [str(cpu) for cpu in range(10)],
        [str(cpu) for cpu in range(10, 20)],
    )

    assert result["server_cpu_count"] == 10
    assert result["client_cpu_count"] == 10


def test_validate_layout_rejects_cpu_outside_numa_node() -> None:
    with pytest.raises(ValueError, match="outside the selected NUMA node"):
        validate_layout(
            1,
            ["0-3"],
            ["8-9"],
            online_cpus=set(range(16)),
            allowed_cpus=set(range(8)),
        )


def _write_result(path: Path, qps: float, latency: float, tokens: int) -> Path:
    path.mkdir()
    payload = {
        "summary": {
            "total_requests": 1,
            "completed_requests": 1,
            "failed_requests": 0,
            "throughput_qps": qps,
            "audio_throughput_s_per_s": qps * 2,
            "output_throughput": qps * tokens,
            "output_tokens_total": tokens,
        },
        "config": {"concurrency": 1},
        "per_request": [
            {
                "is_success": True,
                "latency_s": latency,
                "rtf": latency / 2,
                "audio_duration_s": 2.0,
                "completion_tokens": tokens,
            }
        ],
    }
    result = path / "speed_results.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


def test_summarize_results_combines_concurrent_worker_artifacts(tmp_path: Path) -> None:
    first = _write_result(tmp_path / "worker_0", 2.0, 1.0, 10)
    second = _write_result(tmp_path / "worker_1", 3.0, 3.0, 20)
    result = summarize_results([first, second])["aggregate"]
    assert result["throughput_qps"] == 5.0
    assert result["latency_p50_s"] == 2.0
    assert result["audio_throughput_s_per_s"] == 10.0
    assert result["output_throughput_tok_s"] == 80.0
    assert result["output_tokens_total"] == 30
    assert result["output_tokens_mean"] == 15.0
    assert result["audio_duration_total_s"] == 4.0
    assert result["worker_qps_cv"] == 0.2


def test_summarize_results_uses_shared_wall_clock(tmp_path: Path) -> None:
    first = _write_result(tmp_path / "worker_0", 2.0, 1.0, 10)
    second = _write_result(tmp_path / "worker_1", 3.0, 3.0, 20)

    result = summarize_results([first, second], wall_clock_s=2.0)["aggregate"]

    assert result["throughput_qps"] == 1.0
    assert result["worker_qps_sum"] == 5.0
    assert result["measurement_wall_clock_s"] == 2.0
    assert result["audio_throughput_s_per_s"] == 2.0
    assert result["output_throughput_tok_s"] == 15.0


def test_request_success_gate_rejects_failed_requests() -> None:
    result = {
        "aggregate": {
            "total_requests": 10,
            "completed_requests": 9,
            "failed_requests": 1,
        }
    }

    with pytest.raises(ValueError, match="completed 9/10 requests; failed requests: 1"):
        validate_all_requests_succeeded(result)


def test_summarize_cli_retains_summary_when_request_gate_fails(
    tmp_path: Path,
) -> None:
    result_path = _write_result(tmp_path / "worker_0", 1.0, 1.0, 10)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["summary"].update(
        total_requests=2,
        completed_requests=1,
        failed_requests=1,
    )
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    summary_path = tmp_path / "summary.json"

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/same_gpu_dp/summarize.py",
            "summarize",
            "--require-all-successful",
            "--output",
            str(summary_path),
            str(result_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "completed 1/2 requests; failed requests: 1" in completed.stderr
    assert (
        json.loads(summary_path.read_text(encoding="utf-8"))["aggregate"][
            "failed_requests"
        ]
        == 1
    )


def test_summarize_router_snapshot_uses_actual_worker_counters(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "workers.json"
    snapshot.write_text(
        json.dumps(
            {
                "workers": [
                    {
                        "worker_id": "worker-0",
                        "routed_requests": 60,
                        "successful_requests": 60,
                        "failed_requests": 0,
                    },
                    {
                        "worker_id": "worker-1",
                        "routed_requests": 40,
                        "successful_requests": 39,
                        "failed_requests": 1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = summarize_router_snapshot(snapshot)

    assert result["routed_requests_total"] == 100
    assert result["successful_requests_total"] == 99
    assert result["failed_requests_total"] == 1
    assert result["routed_requests_cv"] == 0.2


def test_extract_kv_tokens_uses_last_startup_value() -> None:
    text = "max_total_num_tokens=1024\nKV Cache allocated 2048 tokens\n"
    assert extract_kv_tokens(text) == 2048


def test_extract_kv_tokens_supports_sglang_allocated_log_format() -> None:
    text = "KV Cache is allocated. #tokens: 84,328"
    assert extract_kv_tokens(text) == 84328


def test_extract_kv_tokens_uses_textually_last_supported_format() -> None:
    text = "KV Cache is allocated. #tokens: 84,328\nmax_total_num_tokens=84000"
    assert extract_kv_tokens(text) == 84000


def test_summarize_matrix_reports_repetitions_ci_and_failures(tmp_path: Path) -> None:
    rows = []
    for repetition, qps in ((1, 10.0), (2, 14.0)):
        output = tmp_path / f"rep{repetition}"
        output.mkdir()
        (output / "summary.json").write_text(
            json.dumps({"aggregate": {"throughput_qps": qps}}), encoding="utf-8"
        )
        rows.append(f"{repetition}\t1\t2\t1\t64\tpass\t{output}\n")
    rows.append(f"3\t1\t2\t1\t64\tfail\t{tmp_path / 'rep3'}\n")
    matrix = tmp_path / "matrix_results.tsv"
    matrix.write_text(
        "repetition\torder_index\tdp\tmps\tconcurrency\tstatus\toutput_dir\n"
        + "".join(rows),
        encoding="utf-8",
    )

    result = summarize_matrix(matrix)
    condition = result["conditions"][0]
    assert condition["throughput_qps_mean"] == 12.0
    assert condition["repetitions"] == 2
    assert condition["throughput_qps_ci95_low"] is not None
    assert len(result["failed_or_incomplete_runs"]) == 1
