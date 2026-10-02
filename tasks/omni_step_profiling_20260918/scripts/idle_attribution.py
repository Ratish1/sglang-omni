"""What the host threads are doing while the device is empty, from a pipeline_nvtx trace.

Loads the trace with pipeline_census.Report (same bench window, ranges and device
intervals), takes the complement of the merged device intervals inside the window, and
samples every empty interval every --step-us. At each sample it records, for the engine
scheduler thread, its innermost probe range and its host state (inside a CUDA launch,
sync or other runtime call, waiting for the GIL, in a blocking OS call, or running
Python), and for the vocoder threads the innermost range of each one that is inside a
range. Empty time is then summed by those keys. Graph replays count as device busy for
their whole span, as in section D.

usage: python idle_attribution.py REPORT.sqlite --bench-log bench.log [--step-us 10]
       [--top 25]
"""

from __future__ import annotations

import argparse
import bisect
import collections

from pipeline_census import Report, merged, ms

ENGINE = "scheduler-tts_engine"
VOCODER_PREFIXES = ("qwen3-tts-vocoder", "scheduler-vocoder")


def state_at(intervals, starts, t):
    """The interval of a start-sorted list that covers t, else None."""
    index = bisect.bisect_right(starts, t) - 1
    if index < 0:
        return None
    item = intervals[index]
    return item if item[0] <= t < item[1] else None


def host_state(r: Report, tid: int, t: int) -> str:
    api = r.api_by_tid.get(tid)
    if api:
        hit = state_at(api, r.api_starts[tid], t)
        if hit is not None:
            return hit[2]
    gil = r.gil_wait.get(tid)
    if gil and state_at(gil, r.gil_starts[tid], t) is not None:
        return "gil"
    os_calls = r.os_wait.get(tid)
    if os_calls:
        hit = state_at(os_calls, r.os_starts[tid], t)
        if hit is not None:
            return "os " + hit[2]
    return "py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--bench-log", required=True)
    parser.add_argument("--step-us", type=float, default=10.0)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    r = Report(args.report, args.bench_log)
    busy = merged(
        [[max(s, r.t0), min(e, r.t1)] for s, e, *_ in r.device if e > r.t0 and s < r.t1]
    )
    empty = []
    last = r.t0
    for s, e in busy:
        if s > last:
            empty.append((last, s))
        last = max(last, e)
    if last < r.t1:
        empty.append((last, r.t1))
    names = {tid: r.tname(tid) for tid in set(r.ranges) | set(r.api_by_tid)}
    engine = [tid for tid, name in names.items() if name == ENGINE]
    vocoder = [tid for tid, name in names.items() if name.startswith(VOCODER_PREFIXES)]
    step = int(args.step_us * 1000)
    window = r.t1 - r.t0
    total_empty = sum(e - s for s, e in empty)
    lengths = sorted(e - s for s, e in empty)
    print(
        f"window {ms(window):.0f} ms, device empty {ms(total_empty):.0f} ms "
        f"({100 * total_empty / window:.1f}%) in {len(empty)} intervals; "
        f"engine threads {len(engine)}, vocoder threads {len(vocoder)}"
    )
    for bound_us in (10, 50, 100, 500, 1000):
        share = sum(x for x in lengths if x >= bound_us * 1000)
        print(f"  empty time in intervals of {bound_us} us or more: {ms(share):.0f} ms")
    by_engine = collections.Counter()
    by_engine_state = collections.Counter()
    by_vocoder = collections.Counter()
    for s, e in empty:
        t = s + step // 2
        while t < e:
            kinds = []
            for tid in engine:
                item = r.innermost(r.ranges, r.starts, tid, t)
                kind = item.kind if item is not None else "(no range)"
                state = host_state(r, tid, t)
                kinds.append(kind)
                by_engine_state[(kind, state)] += step
            by_engine[" + ".join(kinds) or "(no engine thread)"] += step
            active = sorted(
                {
                    item.kind
                    for tid in vocoder
                    if (item := r.innermost(r.ranges, r.starts, tid, t)) is not None
                }
            )
            by_vocoder[" + ".join(active) or "(none)"] += step
            t += step
    sampled = sum(by_engine.values())
    print(
        f"\nengine thread's innermost range during empty time (sampled {ms(sampled):.0f} ms)"
    )
    for key, value in by_engine.most_common(args.top):
        print(f"  {ms(value):8.1f} ms  {100 * value / sampled:5.1f}%  {key}")
    print("\nengine range and host state during empty time")
    for (kind, state), value in by_engine_state.most_common(args.top):
        print(
            f"  {ms(value):8.1f} ms  {100 * value / sampled:5.1f}%  {kind:28s} {state}"
        )
    print("\nvocoder threads' ranges during empty time")
    for key, value in by_vocoder.most_common(args.top):
        print(f"  {ms(value):8.1f} ms  {100 * value / sampled:5.1f}%  {key}")


if __name__ == "__main__":
    main()
