# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.same_gpu_dp.summarize import (
    extract_kv_tokens,
    parse_cpu_set,
    summarize_matrix,
    summarize_results,
    validate_layout,
)


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
