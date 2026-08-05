# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.profiling.system_collectors import (
    CpuFrequencyCollector,
    parse_cpu_frequency,
    parse_gpu_dmon,
    parse_perf_stat,
    parse_thread_snapshots,
    parse_turbostat,
    psi_delta,
    read_psi,
    summarize_thread_snapshot_delta,
)


def _write_psi(root: Path, *, cpu_total: int, memory_total: int) -> None:
    root.mkdir(exist_ok=True)
    (root / "cpu").write_text(
        f"some avg10=0.10 avg60=0.20 avg300=0.30 total={cpu_total}\n"
        "full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
        encoding="utf-8",
    )
    for resource in ("memory", "io"):
        (root / resource).write_text(
            f"some avg10=1.00 avg60=2.00 avg300=3.00 total={memory_total}\n"
            f"full avg10=0.50 avg60=1.00 avg300=1.50 total={memory_total // 2}\n",
            encoding="utf-8",
        )


def test_psi_parser_and_exact_total_delta(tmp_path: Path) -> None:
    root = tmp_path / "pressure"
    _write_psi(root, cpu_total=100, memory_total=200)
    before = read_psi(root)
    _write_psi(root, cpu_total=175, memory_total=260)
    after = read_psi(root)

    delta = psi_delta(before, after)
    assert delta["cpu"]["some"]["total_us"] == 75
    assert delta["memory"]["some"]["total_us"] == 60
    assert delta["memory"]["full"]["total_us"] == 30
    assert delta["window_ns"] >= 0


def test_perf_interval_parser_sums_counters(tmp_path: Path) -> None:
    output = tmp_path / "perf.csv"
    output.write_text(
        "1.000,1000.0,msec,task-clock,100.0,\n"
        "1.000,2000000,,instructions,100.0,\n"
        "2.000,900.0,msec,task-clock,100.0,\n"
        "2.000,<not counted>,,instructions,0.0,\n",
        encoding="utf-8",
    )
    parsed = parse_perf_stat(output)
    assert parsed["counters"]["task-clock"]["value"] == 1900.0
    assert parsed["counters"]["task-clock"]["intervals"] == 2
    assert parsed["counters"]["instructions"]["value"] == 2_000_000
    assert len(parsed["ignored"]) == 1


def test_turbostat_parser_uses_measured_busy_frequency(tmp_path: Path) -> None:
    output = tmp_path / "turbostat.txt"
    output.write_text(
        "Core CPU Busy% Bzy_MHz PkgWatt CoreTmp\n"
        "- - 80.0 2500 200.0 70\n"
        "0 0 90.0 2400 201.0 71\n",
        encoding="utf-8",
    )
    parsed = parse_turbostat(output)
    assert parsed["columns"]["Bzy_MHz"]["mean"] == 2450.0
    assert parsed["columns"]["Busy%"]["min"] == 80.0


def test_thread_snapshot_parser_attributes_cpu_and_runnable_delay(
    tmp_path: Path,
) -> None:
    output = tmp_path / "threads.jsonl"
    samples = [
        {
            "monotonic_ns": 1_000_000_000,
            "pid": 1,
            "threads": [
                {
                    "tid": 10,
                    "comm": "omni-request-build",
                    "utime_ticks": 100,
                    "stime_ticks": 20,
                    "runtime_ns": 800_000_000,
                    "runqueue_delay_ns": 100_000_000,
                    "timeslices": 50,
                    "processor": 2,
                }
            ],
        },
        {
            "monotonic_ns": 2_000_000_000,
            "pid": 1,
            "threads": [
                {
                    "tid": 10,
                    "comm": "omni-request-build",
                    "utime_ticks": 150,
                    "stime_ticks": 30,
                    "runtime_ns": 1_300_000_000,
                    "runqueue_delay_ns": 300_000_000,
                    "timeslices": 80,
                    "processor": 4,
                }
            ],
        },
    ]
    output.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    parsed = parse_thread_snapshots(output)
    thread = parsed["threads"][0]
    assert thread["tid"] == 10
    assert thread["runtime_ms"] == 500.0
    assert thread["runqueue_delay_ms"] == 200.0
    assert thread["timeslices"] == 30
    assert thread["runtime_fraction"] == 0.5


def test_thread_snapshot_delta_includes_migrations_and_thread_lifecycle() -> None:
    before = {
        "monotonic_ns": 1_000_000_000,
        "threads": [
            {
                "tid": 10,
                "comm": "scheduler-asr",
                "starttime_ticks": 1,
                "utime_ticks": 100,
                "stime_ticks": 20,
                "runtime_ns": 800_000_000,
                "runqueue_delay_ns": 100_000_000,
                "timeslices": 50,
                "migrations": 3,
                "processor": 2,
            },
            {
                "tid": 11,
                "comm": "exiting-worker",
                "starttime_ticks": 2,
                "utime_ticks": 1,
                "stime_ticks": 1,
                "runtime_ns": 1,
                "runqueue_delay_ns": 1,
                "timeslices": 1,
                "migrations": 0,
                "processor": 1,
            },
        ],
    }
    after = {
        "monotonic_ns": 2_000_000_000,
        "threads": [
            {
                "tid": 10,
                "comm": "scheduler-asr",
                "starttime_ticks": 1,
                "utime_ticks": 150,
                "stime_ticks": 30,
                "runtime_ns": 1_300_000_000,
                "runqueue_delay_ns": 300_000_000,
                "timeslices": 80,
                "migrations": 8,
                "processor": 4,
            },
            {
                "tid": 12,
                "comm": "new-worker",
                "starttime_ticks": 3,
                "utime_ticks": 5,
                "stime_ticks": 1,
                "runtime_ns": 40_000_000,
                "runqueue_delay_ns": 10_000_000,
                "timeslices": 4,
                "migrations": 2,
                "processor": 6,
            },
        ],
    }
    parsed = summarize_thread_snapshot_delta(before, after)
    assert parsed["threads_started"] == [12]
    assert parsed["threads_exited"] == [11]
    assert parsed["threads_persistent"] == 1
    assert parsed["migrations"] == 7
    assert parsed["threads"][0]["runqueue_delay_ms"] == 200.0
    assert parsed["threads"][0]["first_processor"] == 2
    assert parsed["threads"][0]["last_processor"] == 4
    started = next(row for row in parsed["threads"] if row["tid"] == 12)
    assert started["lifecycle"] == "started"
    assert started["cpu_ms"] == 60.0
    assert started["runtime_ms"] == 40.0
    assert started["runqueue_delay_ms"] == 10.0
    assert started["first_processor"] is None


def test_gpu_dmon_parser_reports_idle_fraction(tmp_path: Path) -> None:
    output = tmp_path / "gpu_dmon.txt"
    output.write_text(
        "# gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk\n"
        "0 250 60 45 0 10 0 0 0 0 1200 1500\n"
        "0 300 62 46 80 20 0 0 0 0 1200 1500\n",
        encoding="utf-8",
    )
    parsed = parse_gpu_dmon(output)
    assert parsed["samples"] == 2
    assert parsed["columns"]["sm"]["mean"] == 40.0
    assert parsed["columns"]["sm"]["zero_fraction"] == 0.5


def test_cpu_frequency_parser_reports_busy_weighted_frequency(tmp_path: Path) -> None:
    output = tmp_path / "cpu_frequency.jsonl"
    samples = [
        {
            "monotonic_ns": 1,
            "requested_cpus": [0],
            "cpus": [
                {
                    "cpu": 0,
                    "frequency_khz": 2_000_000,
                    "total_ticks": 100,
                    "idle_ticks": 50,
                }
            ],
        },
        {
            "monotonic_ns": 2,
            "requested_cpus": [0],
            "cpus": [
                {
                    "cpu": 0,
                    "frequency_khz": 3_000_000,
                    "total_ticks": 200,
                    "idle_ticks": 75,
                }
            ],
        },
    ]
    output.write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples),
        encoding="utf-8",
    )
    parsed = parse_cpu_frequency(output)
    assert parsed["samples"] == 2
    assert parsed["interior_samples"] == 2
    assert parsed["scope"] == {
        "kind": "selected_cpus",
        "requested_cpus": [0],
        "observed_cpus": [0],
    }
    assert parsed["sampled_scaling_frequency_mhz"]["mean"] == 2500.0
    assert parsed["busy_weighted_sampled_scaling_frequency_mhz"] == 3000.0
    assert parsed["busy_ticks"] == 75
    assert parsed["coverage"] == {
        "first_sample_monotonic_ns": 1,
        "last_sample_monotonic_ns": 2,
        "brackets_start": True,
        "brackets_stop": True,
    }


def test_cpu_frequency_collector_finalizes_with_trailing_sample(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cpu_frequency.jsonl"
    collector = CpuFrequencyCollector(output_path=output, interval_ms=10_000)
    collector.start()
    result = collector.stop()
    samples = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert result["returncode"] == 0
    assert result["samples"] >= 2
    assert len(samples) == result["samples"]
    assert samples[0]["monotonic_ns"] <= samples[-1]["monotonic_ns"]


def test_cpu_frequency_parser_filters_to_monotonic_window(tmp_path: Path) -> None:
    output = tmp_path / "cpu_frequency.jsonl"
    output.write_text(
        "\n".join(
            json.dumps(
                {
                    "monotonic_ns": timestamp,
                    "requested_cpus": [0],
                    "cpus": [
                        {
                            "cpu": 0,
                            "frequency_khz": frequency,
                            "total_ticks": ticks,
                            "idle_ticks": 0,
                        }
                    ],
                }
            )
            for timestamp, frequency, ticks in (
                (100, 1_000_000, 10),
                (200, 2_000_000, 20),
                (300, 3_000_000, 30),
                (400, 4_000_000, 40),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_cpu_frequency(
        output,
        start_monotonic_ns=200,
        stop_monotonic_ns=300,
    )
    assert parsed["samples"] == 2
    assert parsed["interior_samples"] == 2
    assert parsed["total_file_samples"] == 4
    assert parsed["sampled_scaling_frequency_mhz"]["mean"] == 2500.0
    assert parsed["busy_weighted_sampled_scaling_frequency_mhz"] == 3000.0


def test_cpu_frequency_parser_includes_bracketing_samples(
    tmp_path: Path,
) -> None:
    output = tmp_path / "cpu_frequency.jsonl"
    output.write_text(
        "\n".join(
            json.dumps(
                {
                    "monotonic_ns": timestamp,
                    "requested_cpus": [0],
                    "cpus": [
                        {
                            "cpu": 0,
                            "frequency_khz": timestamp * 10_000,
                            "total_ticks": timestamp,
                            "idle_ticks": 0,
                        }
                    ],
                }
            )
            for timestamp in (100, 200, 300, 400)
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_cpu_frequency(
        output,
        start_monotonic_ns=150,
        stop_monotonic_ns=350,
    )
    assert parsed["samples"] == 4
    assert parsed["interior_samples"] == 2
    assert parsed["coverage"] == {
        "first_sample_monotonic_ns": 100,
        "last_sample_monotonic_ns": 400,
        "brackets_start": True,
        "brackets_stop": True,
    }
