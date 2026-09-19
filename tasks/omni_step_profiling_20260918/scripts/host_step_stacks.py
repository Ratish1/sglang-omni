"""Where the scheduler thread's host time goes per step class, from a with-stack trace.

Steps are the omni.step spans of the scheduler thread (start to the next start). Classes:
prefill, decode steps whose device-empty time is at least --stall-ms (stalled), and the
other decode steps. For each class: steps, mean wall, and the python functions of the
scheduler thread ranked by self time per step (a function's duration minus its direct
children on the same thread), with inclusive time next to it. With-stack tracing
inflates host time several times over, so the ranking names the code; it never times it.

usage: python host_step_stacks.py STACK_TRACE.json.gz [--stall-ms 3] [--top 25]
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
from collections import Counter, defaultdict

SPAN = re.compile(r"^omni\.step (\S+) (\S+) fwd=(\d+) bs=(\d+)")
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def short(name: str) -> str:
    for marker in ("sglang_omni/", "site-packages/"):
        if marker in name:
            return name.split(marker, 1)[1]
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--stall-ms", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as handle:
        events = [e for e in json.load(handle)["traceEvents"] if e.get("ph") == "X"]

    spans = [
        e for e in events if e.get("cat") == "user_annotation" and SPAN.match(e["name"])
    ]
    sched_tid = Counter(e.get("tid") for e in spans).most_common(1)[0][0]
    spans = sorted(
        (e for e in spans if e.get("tid") == sched_tid), key=lambda e: e["ts"]
    )
    busy = sorted(
        (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        for e in events
        if e.get("cat") in GPU_CATS
    )
    merged: list[list[float]] = []
    for start, end in busy:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    merged_starts = [m[0] for m in merged]

    def empty_us(lo: float, hi: float) -> float:
        covered = 0.0
        index = max(bisect.bisect_right(merged_starts, lo) - 1, 0)
        while index < len(merged) and merged[index][0] < hi:
            covered += max(0.0, min(merged[index][1], hi) - max(merged[index][0], lo))
            index += 1
        return (hi - lo) - covered

    steps = []
    for current, following in zip(spans, spans[1:]):
        kind = SPAN.match(current["name"]).group(1)
        start, end = float(current["ts"]), float(following["ts"])
        label = kind
        if kind == "decode":
            label = (
                "decode stalled"
                if empty_us(start, end) >= args.stall_ms * 1e3
                else "decode"
            )
        steps.append((start, end, label))
    step_starts = [s[0] for s in steps]

    calls = sorted(
        (
            e
            for e in events
            if e.get("cat") == "python_function" and e.get("tid") == sched_tid
        ),
        key=lambda e: (float(e["ts"]), -float(e["dur"])),
    )
    self_us: dict[str, Counter] = defaultdict(Counter)
    total_us: dict[str, Counter] = defaultdict(Counter)
    stack: list[list] = []
    for event in calls:
        start = float(event["ts"])
        end = start + float(event["dur"])
        while stack and stack[-1][1] <= start:
            stack.pop()
        if stack:
            stack[-1][2] -= end - start
        frame = [start, end, end - start, short(event["name"])]
        stack.append(frame)
        index = bisect.bisect_right(step_starts, start) - 1
        if 0 <= index < len(steps) and start < steps[index][1]:
            frame.append(steps[index][2])
            total_us[steps[index][2]][frame[3]] += end - start
        else:
            frame.append(None)
        event["_frame"] = frame
    for event in calls:
        frame = event["_frame"]
        if frame[4] is not None:
            self_us[frame[4]][frame[3]] += max(frame[2], 0.0)

    counts = Counter(label for _, _, label in steps)
    for label in ("prefill", "decode stalled", "decode"):
        if not counts[label]:
            continue
        walls = [end - start for start, end, lab in steps if lab == label]
        print(
            f"\n{label}: {counts[label]} steps, wall mean {sum(walls) / len(walls) / 1e3:.2f} ms "
            f"(with stack; inflated). scheduler thread functions, ms per step:"
        )
        print(f"  {'self':>8s}{'incl':>9s}  function")
        for name, value in self_us[label].most_common(args.top):
            print(
                f"  {value / counts[label] / 1e3:>8.3f}{total_us[label][name] / counts[label] / 1e3:>9.3f}  {name[:150]}"
            )


if __name__ == "__main__":
    main()
