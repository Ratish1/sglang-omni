"""Qwen3-Omni talker admissions against its decode steps, from an events cell and the hop probe's talker batches
(OMNI_HOP_PROBE_TALKER_BATCHES=1), boots of one arm pooled.

A talker prefill step is the requests whose talker_ar scheduler_prefill_end share one timestamp. A decode interval is
the time between two consecutive "tb" records (each talker decode step), inside the events pass. Printed per arm:
prefill steps and requests a step; decode intervals holding two or more prefill steps (admissions back to back), the
slowest 1 % of intervals with their prefill steps, intervals above 150 ms; each request's first gap (its first to
second code2wav chunk) with the longest decode interval inside it, over all requests and the slowest 5 %.

usage: python3 admission_trains.py --cell c64_events --arm base=run/a1,run/a2 --arm cap=run/b1,run/b2
"""

import argparse
import bisect
import glob
import json
import statistics
from collections import defaultdict

LONG_INTERVAL_MS = 150.0


def boot_records(boot: str, cell: str):
    events = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(f"{boot}/{cell}/events/*.jsonl"):
        with open(path, encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                events[record["request_id"]][(record["stage"], record["event_name"])].append(record["timestamp_ns"])
    steps = []
    for path in glob.glob(f"{boot}/probe/p.*.jsonl"):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if '"tb"' in line:
                    steps.append(json.loads(line)["ns"])
                else:
                    pass
    return events, sorted(steps)


def boot_rows(boot: str, cell: str):
    events, steps = boot_records(boot, cell)
    starts, ends = {}, defaultdict(list)
    for rid, by_name in events.items():
        start = by_name.get(("talker_ar", "scheduler_prefill_start"))
        end = by_name.get(("talker_ar", "scheduler_prefill_end"))
        if start and end:
            starts[rid] = min(start)
            ends[min(end)].append(rid)
        else:
            pass
    prefill_steps = sorted((min(starts[rid] for rid in rids), end, len(rids)) for end, rids in ends.items())
    window_start = min(step[0] for step in prefill_steps)
    window_end = max(step[1] for step in prefill_steps)
    steps = [ns for ns in steps if window_start <= ns <= window_end]
    step_starts = [step[0] for step in prefill_steps]
    intervals = []
    for left, right in zip(steps, steps[1:]):
        inside = bisect.bisect_left(step_starts, right) - bisect.bisect_left(step_starts, left)
        intervals.append(((right - left) / 1e6, inside))
    gaps = []
    for by_name in events.values():
        chunks = sorted(by_name.get(("code2wav", "stage_stream_chunk_sent"), []))
        if len(chunks) < 2:
            continue
        else:
            pass
        first, second = chunks[0], chunks[1]
        lo, hi = bisect.bisect_left(steps, first), bisect.bisect_left(steps, second)
        longest = max(((steps[k + 1] - steps[k]) / 1e6 for k in range(lo, min(hi, len(steps) - 1))), default=0.0)
        gaps.append(((second - first) / 1e6, longest))
    return prefill_steps, intervals, gaps


def report(label: str, boots: list[str], cell: str) -> None:
    prefill_steps, intervals, gaps = [], [], []
    for boot in boots:
        boot_steps, boot_intervals, boot_gaps = boot_rows(boot, cell)
        prefill_steps += boot_steps
        intervals += boot_intervals
        gaps += boot_gaps
    sizes = [step[2] for step in prefill_steps]
    durations = [(step[1] - step[0]) / 1e6 for step in prefill_steps]
    print(
        f"{label}: {len(prefill_steps)} talker prefill steps, requests a step mean {statistics.mean(sizes):.2f} "
        f"max {max(sizes)}, step ms p50 {statistics.median(durations):.1f}"
    )
    intervals.sort()
    slowest = intervals[-max(len(intervals) // 100, 1):]
    long_intervals = [interval for interval in intervals if interval[0] > LONG_INTERVAL_MS]
    print(
        f"  decode intervals {len(intervals)}: with two or more prefill steps {sum(i[1] >= 2 for i in intervals)}, "
        f"most prefill steps in one {max(i[1] for i in intervals)}; slowest 1 % {statistics.mean(i[0] for i in slowest):.0f} ms "
        f"with {statistics.mean(i[1] for i in slowest):.1f} prefill steps; above {LONG_INTERVAL_MS:.0f} ms {len(long_intervals)}"
    )
    gaps.sort()
    top = gaps[-max(len(gaps) // 20, 1):]
    print(
        f"  first gap: {len(gaps)} requests, mean {statistics.mean(g[0] for g in gaps):.0f} ms, longest decode interval "
        f"inside {statistics.mean(g[1] for g in gaps):.0f}; slowest 5 % {statistics.mean(g[0] for g in top):.0f} ms, "
        f"longest interval {statistics.mean(g[1] for g in top):.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--arm", action="append", required=True, help="label=boot,boot")
    args = parser.parse_args()
    for arm in args.arm:
        label, boots = arm.split("=", 1)
        report(label, boots.split(","), args.cell)


if __name__ == "__main__":
    main()
