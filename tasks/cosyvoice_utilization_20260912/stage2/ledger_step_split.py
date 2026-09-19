#!/usr/bin/env python3
"""Where the streaming vocoder's time goes, from a call ledger.

Reads the ledger_<pid>.jsonl files of a profiling boot and splits the vocoder
thread's span into idle, Flow hop calls (plain and cached), Flow final calls,
HiFT calls and the rest of a step, by host wall and by device time. A step
that makes both hop calls is a mixed step.

  python ledger_step_split.py <run dir>/ledger
"""

from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict


def p(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def line(name: str, values: list[float]) -> str:
    if not values:
        return f"  {name:34s} n=   0"
    return (
        f"  {name:34s} n={len(values):4d} sum={sum(values) / 1e3:7.1f}s "
        f"p50={statistics.median(values):7.1f} p95={p(values, 0.95):7.1f} "
        f"max={max(values):7.1f} ms"
    )


def size(rows: int) -> str:
    return "1-4" if rows <= 4 else "5-9" if rows <= 9 else "10-16"


def main() -> None:
    entries = []
    for path in glob.glob(f"{sys.argv[1]}/ledger_*.jsonl"):
        with open(path) as ledger:
            entries += [json.loads(row) for row in ledger]
    steps = sorted(
        (entry for entry in entries if entry["kind"] == "step"),
        key=lambda entry: entry["t_wall"],
    )
    calls = defaultdict(list)
    for entry in entries:
        if entry["kind"] != "step" and entry.get("step") is not None:
            calls[entry["step"]].append(entry)

    span = steps[-1]["t_wall"] + steps[-1]["host_ms"] / 1e3 - steps[0]["t_wall"]
    busy = sum(step["host_ms"] for step in steps) / 1e3
    print(
        f"steps={len(steps)} span={span:.1f}s in steps={busy:.1f}s idle={span - busy:.1f}s"
    )

    host = defaultdict(float)
    device = defaultdict(float)
    for step in steps:
        inside = 0.0
        for call in calls[step["id"]]:
            host[call["kind"]] += call["host_ms"]
            device[call["kind"]] += call.get("gpu_ms", 0.0)
            inside += call["host_ms"]
        host[f"rest of {step['plan']} step"] += step["host_ms"] - inside
    print("host wall and device time inside steps, seconds:")
    for kind in sorted(host):
        print(
            f"  {kind:34s} host={host[kind] / 1e3:7.1f} device={device[kind] / 1e3:7.1f}"
        )

    groups: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        kinds = {call["kind"] for call in calls[step["id"]]}
        rows = len(step["participants"])
        if step["plan"] == "leftover":
            shape = "final"
        elif {"hop", "hop_cached"} <= kinds:
            shape = "hop mixed"
        elif "hop_cached" in kinds:
            shape = "hop all cached"
        else:
            shape = "hop all plain"
        first = sum(1 for row in step["participants"] if row["token_offset"] == 0)
        if shape != "final":
            shape += " first" if first == rows else " later" if first == 0 else " both"
        groups[f"step {shape}"].append(step["host_ms"])
        groups[f"step {shape} rows {size(rows)}"].append(step["host_ms"])
    print("step host wall:")
    for name in sorted(groups):
        print(line(name, groups[name]))

    print("Flow calls:")
    for kind in ("hop", "hop_cached", "final"):
        chosen = [e for e in entries if e["kind"] == kind and e.get("step") is not None]
        for bucket in ("1-4", "5-9", "10-16"):
            part = [call for call in chosen if size(call["rows"]) == bucket]
            if not part:
                continue
            frames = statistics.median(call["total_frames"] for call in part)
            new = statistics.median(
                sum(call.get("new_frames", call["row_frames"])) for call in part
            )
            print(
                f" {kind} rows {bucket}: window frames p50={frames:.0f} computed frames p50={new:.0f}"
            )
            print(line("host", [call["host_ms"] for call in part]))
            print(line("device", [call.get("gpu_ms", 0.0) for call in part]))

    print("HiFT calls:")
    hift = [e for e in entries if e["kind"] == "hift" and e.get("step") is not None]
    for final in (False, True):
        part = [call for call in hift if call["finalize"] is final]
        print(line(f"finalize={final} host", [call["host_ms"] for call in part]))
        print(
            line(f"finalize={final} device", [call.get("gpu_ms", 0.0) for call in part])
        )
    # The first HiFT call of a step copies to the host, so it also waits for
    # the Flow work the hop call only launched.
    first, later = [], []
    for step in steps:
        ordered = sorted(
            (call for call in calls[step["id"]] if call["kind"] == "hift"),
            key=lambda call: call["t_wall"],
        )
        first += [call["host_ms"] for call in ordered[:1]]
        later += [call["host_ms"] for call in ordered[1:]]
    print(line("first HiFT of a step host", first))
    print(line("later HiFT of a step host", later))

    waits = [
        row["wait_ms"]
        for step in steps
        for row in step["participants"]
        if row["wait_ms"] is not None
        and row["token_offset"] == 0
        and step["plan"] != "leftover"
    ]
    print(line("first hop wait since ready", waits))


if __name__ == "__main__":
    main()
