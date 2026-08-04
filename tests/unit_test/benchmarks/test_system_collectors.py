# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from benchmarks.profiling.system_collectors import (
    parse_perf_stat,
    parse_turbostat,
    psi_delta,
    read_psi,
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
