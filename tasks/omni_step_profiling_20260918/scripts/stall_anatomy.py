"""What the host is doing while the device is empty, from a formal (no stack) live trace.

Takes every device-empty interval of at least --gap-ms inside the window of omni.step
spans, and attributes the interval to the scheduler thread's own host events that overlap
it (cpu ops, CUDA runtime calls, user annotations; innermost first, so a time slice counts
for the deepest event covering it) and to "no scheduler event" when the scheduler thread
shows nothing. Next to it: how many host events every other thread ran inside the same
intervals. A scheduler thread with nothing on it while other threads run ops is a thread
waiting (the interpreter lock or a queue), not one working.

usage: python stall_anatomy.py TRACE.json.gz [--gap-ms 1.0] [--top 20]
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
from collections import Counter

SPAN = re.compile(r"^omni\.step (\S+) (\S+) fwd=(\d+) bs=(\d+)")
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
HOST_CATS = ("cpu_op", "cuda_runtime", "cuda_driver", "user_annotation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--gap-ms", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as handle:
        events = [e for e in json.load(handle)["traceEvents"] if e.get("ph") == "X"]

    spans = sorted(
        (
            e
            for e in events
            if e.get("cat") == "user_annotation" and SPAN.match(e["name"])
        ),
        key=lambda e: e["ts"],
    )
    sched_tid = Counter(e.get("tid") for e in spans).most_common(1)[0][0]
    spans = [e for e in spans if e.get("tid") == sched_tid]
    t0, t1 = float(spans[0]["ts"]), float(spans[-1]["ts"])

    merged: list[list[float]] = []
    for start, end in sorted(
        (float(e["ts"]), float(e["ts"]) + float(e["dur"]))
        for e in events
        if e.get("cat") in GPU_CATS
    ):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    gaps = [
        (max(lo[1], t0), min(hi[0], t1))
        for lo, hi in zip(merged, merged[1:])
        if min(hi[0], t1) - max(lo[1], t0) >= args.gap_ms * 1e3
    ]
    total_gap = sum(hi - lo for lo, hi in gaps)
    print(f"trace {args.trace}")
    print(
        f"window {(t1 - t0) / 1e3:.1f} ms; {len(gaps)} device-empty intervals of at least {args.gap_ms} ms, "
        f"{total_gap / 1e3:.1f} ms = {100 * total_gap / (t1 - t0):.1f}% of the window"
    )

    span_starts = [float(span["ts"]) for span in spans]

    def step_of(ts: float) -> str:
        index = bisect.bisect_right(span_starts, ts) - 1
        return SPAN.match(spans[index]["name"]).group(1) if index >= 0 else "outside"

    by_kind = Counter()
    for lo, hi in gaps:
        by_kind[step_of(lo)] += hi - lo
    print(
        "by step kind at the start of the interval: "
        + ", ".join(f"{k} {v / 1e3:.1f} ms" for k, v in by_kind.most_common())
    )

    host = [
        e
        for e in events
        if e.get("cat") in HOST_CATS and not e["name"].startswith("omni.step ")
    ]
    sched = sorted(
        (
            (float(e["ts"]), float(e["ts"]) + float(e["dur"]), e["name"])
            for e in host
            if e.get("tid") == sched_tid
        ),
        key=lambda item: (item[0], -item[1]),
    )
    sched_starts = [ev[0] for ev in sched]
    longest = max(ev[1] - ev[0] for ev in sched)
    other_starts: dict = {}
    for event in host:
        if event.get("tid") != sched_tid:
            other_starts.setdefault(event.get("tid"), []).append(float(event["ts"]))
    for starts in other_starts.values():
        starts.sort()
    attributed = Counter()
    others = Counter()
    for lo, hi in gaps:
        first = bisect.bisect_left(sched_starts, lo - longest)
        last = bisect.bisect_left(sched_starts, hi)
        inside = [ev for ev in sched[first:last] if ev[1] > lo]
        cuts = sorted({lo, hi, *(min(max(t, lo), hi) for ev in inside for t in ev[:2])})
        for left, right in zip(cuts, cuts[1:]):
            covering = [ev for ev in inside if ev[0] <= left and ev[1] >= right]
            name = (
                min(covering, key=lambda ev: ev[1] - ev[0])[2]
                if covering
                else "no scheduler event"
            )
            attributed[name] += right - left
        for tid, starts in other_starts.items():
            others[tid] += bisect.bisect_left(starts, hi) - bisect.bisect_left(
                starts, lo
            )
    print(
        f"\nscheduler thread during the device-empty intervals (ms, share of the {total_gap / 1e3:.1f} ms)"
    )
    for name, value in attributed.most_common(args.top):
        print(f"  {value / 1e3:>9.1f}{100 * value / total_gap:>7.1f}%  {name[:120]}")
    print("\nhost events of other threads inside the same intervals (count per thread)")
    for tid, count in others.most_common(8):
        print(f"  tid {tid}: {count}")


if __name__ == "__main__":
    main()
