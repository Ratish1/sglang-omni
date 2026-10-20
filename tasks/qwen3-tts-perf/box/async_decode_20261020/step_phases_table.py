"""Scheduler loop phases over a pair's bench window, from each arm's step_phases.jsonl (pair.py --step-phases).

usage: python3 step_phases_table.py <label> [<label> ...]
The window is the difference between the last snapshot before the bench started and the first one after it ended.
Per arm, one row per iteration kind and batch size band: iterations, window seconds, then per iteration the wall and
thread CPU milliseconds (wall minus CPU is time off the CPU: lock waits, sleeps, blocking calls), each phase's wall/CPU,
and the device waits (count per iteration, mean ms per wait). The lookahead's step wait splits into steps already done
at resolve (step_ready) and steps the host waited for (step_waited).
"""

import json
import sys
from pathlib import Path

ROOT = Path("/data/async_decode_20261020/pairs")
BANDS = ((1, 1), (2, 8), (9, 16), (17, 32), (33, 48), (49, 64))


def snapshot(path: Path, start: float, end: float) -> tuple[dict, dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    before = next((record["totals"] for record in reversed(records) if record["time"] <= start), {})
    after = next((record["totals"] for record in records if record["time"] >= end), records[-1]["totals"])
    return before, after


def difference(after: dict, before: dict) -> dict:
    window = {}
    for key, entry in after.items():
        base = before.get(key, {"iterations": 0, "phases": {}, "waits": {}})
        iterations = entry["iterations"] - base["iterations"]
        if iterations <= 0:
            continue
        else:
            pass
        phases = {
            name: [value - base_value for value, base_value in zip(totals, base["phases"].get(name, [0, 0.0, 0.0]))]
            for name, totals in entry["phases"].items()
        }
        waits = {
            name: [value - base_value for value, base_value in zip(totals, base["waits"].get(name, [0, 0.0]))]
            for name, totals in entry["waits"].items()
        }
        window[key] = {"iterations": iterations, "phases": phases, "waits": waits}
    return window


def band_of(batch_size: int) -> str:
    for low, high in BANDS:
        if low <= batch_size <= high:
            return f"{low}-{high}" if low != high else str(low)
        else:
            pass
    return str(batch_size)


def rows(window: dict) -> list[str]:
    grouped: dict[tuple[str, str], dict] = {}
    for key, entry in window.items():
        kind, batch_size = key.split("|")
        group = grouped.setdefault((kind, band_of(int(batch_size))), {"iterations": 0, "phases": {}, "waits": {}})
        group["iterations"] += entry["iterations"]
        for name, totals in entry["phases"].items():
            merged = group["phases"].setdefault(name, [0, 0.0, 0.0])
            for index, value in enumerate(totals):
                merged[index] += value
        for name, totals in entry["waits"].items():
            merged = group["waits"].setdefault(name, [0, 0.0])
            for index, value in enumerate(totals):
                merged[index] += value
    total_wall = sum(sum(totals[1] for totals in group["phases"].values()) for group in grouped.values())
    lines = [f"window {total_wall:.2f} s of loop wall"]
    for (kind, band), group in sorted(grouped.items(), key=lambda item: (item[0][0], int(item[0][1].split("-")[0]))):
        iterations = group["iterations"]
        wall = sum(totals[1] for totals in group["phases"].values())
        cpu = sum(totals[2] for totals in group["phases"].values())
        phases = " ".join(
            f"{name} {totals[1] / iterations * 1e3:.2f}/{totals[2] / iterations * 1e3:.2f}"
            for name, totals in group["phases"].items()
        )
        waits = " ".join(
            f"{name} {count / iterations:.2f}x{(seconds / count * 1e3) if count else 0:.2f}"
            for name, (count, seconds) in group["waits"].items()
        )
        lines.append(
            f"{kind} {band} | {iterations} | {wall:.2f} s ({wall / total_wall * 100:.1f} %) | "
            f"{wall / iterations * 1e3:.2f} / {cpu / iterations * 1e3:.2f} | {phases} | {waits}"
        )
    return lines


def main() -> None:
    for label in sys.argv[1:]:
        for arm in ("a", "b"):
            directory = ROOT / label / arm
            start, end = (float(value) for value in (directory / "bench_window.txt").read_text().split())
            before, after = snapshot(directory / "step_phases.jsonl", start, end)
            tree = (directory / "tree.txt").read_text().splitlines()[1]
            print(f"{label} {arm} {tree}")
            print("kind band | iterations | window | wall / cpu ms per iteration | phase wall/cpu ms | waits")
            for line in rows(difference(after, before)):
                print(line)


if __name__ == "__main__":
    main()
