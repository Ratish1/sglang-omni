"""Whole-window budget of a live sglang-omni torch trace (mixed prefill and decode steps).

step_ledger.py reads the modal step class of a degenerate capture. A window taken under
real load mixes step classes, so this reads the whole window instead: where the wall time
of the scheduler loop goes by step class, and where the device time goes by owner thread.
Ownership is step_ledger's rule (cpu op thread through External id, else the launch
thread). Thread roles come from python frames; a formal trace has none, so --roles takes
a with-stack trace of the same boot (thread ids persist for the life of the process).

usage: python window_budget.py TRACE.json.gz [--roles STACK_TRACE.json.gz] [--top 12]
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict

SPAN = re.compile(r"^omni\.step (\S+) (\S+) fwd=(\d+) bs=(\d+)(?: toks=(\d+))?$")
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
Intervals = list[tuple[float, float]]


def merge(intervals: Intervals) -> Intervals:
    merged: Intervals = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def clipped_us(merged: Intervals, starts: list[float], lo: float, hi: float) -> float:
    """Time of a merged interval list inside [lo, hi)."""
    total = 0.0
    index = max(bisect.bisect_right(starts, lo) - 1, 0)
    while index < len(merged) and merged[index][0] < hi:
        total += max(0.0, min(merged[index][1], hi) - max(merged[index][0], lo))
        index += 1
    return total


def intersect(left: Intervals, right: Intervals) -> Intervals:
    out: Intervals = []
    i = j = 0
    while i < len(left) and j < len(right):
        lo, hi = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if hi > lo:
            out.append((lo, hi))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return out


def load(path: str) -> list[dict]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)["traceEvents"]


def thread_frames(events: list[dict]) -> dict[str, Counter]:
    frames: dict[str, Counter] = defaultdict(Counter)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") == "python_function":
            name = event["name"]
            if "sglang_omni/" in name:
                frames[str(event.get("tid"))][
                    name.split("sglang_omni/")[1].split("(")[0]
                ] += 1
    return frames


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--roles")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    events = load(args.trace)
    frames = thread_frames(load(args.roles) if args.roles else events)

    ext_tid: dict[int, str] = {}
    launch_tid: dict[int, str] = {}
    gpu: list[dict] = []
    spans: list[tuple[str, dict]] = []
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = event.get("cat")
        args_ = event.get("args") or {}
        if cat in ("cpu_op", "user_annotation"):
            if args_.get("External id") is not None:
                ext_tid.setdefault(args_["External id"], str(event.get("tid")))
            if cat == "user_annotation" and event["name"].startswith("omni.step "):
                spans.append((str(event.get("tid")), event))
        elif cat in ("cuda_runtime", "cuda_driver"):
            if args_.get("correlation") is not None:
                launch_tid[args_["correlation"]] = str(event.get("tid"))
        elif cat in GPU_CATS:
            gpu.append(event)
    live_tids = set(ext_tid.values())
    sched_tid = Counter(tid for tid, _ in spans).most_common(1)[0][0]

    steps = []
    for tid, event in sorted(spans, key=lambda item: item[1]["ts"]):
        match = SPAN.match(event["name"])
        if tid == sched_tid and match is not None:
            kind, _, fwd, bs, toks = match.groups()
            start = float(event["ts"])
            steps.append(
                (
                    kind,
                    int(bs),
                    int(toks) if toks else None,
                    start,
                    start + float(event["dur"]),
                    int(fwd),
                )
            )
    fwds = [s[5] for s in steps]
    t0, t1 = steps[0][3], steps[-1][3]
    window = t1 - t0
    print(f"trace {args.trace}")
    print(
        f"scheduler tid {sched_tid}; {len(steps) - 1} whole steps, fwd {fwds[0]}..{fwds[-1]} "
        f"contiguous {fwds == list(range(fwds[0], fwds[-1] + 1))}; window {window / 1e3:.1f} ms"
    )

    by_owner: dict[str, Intervals] = defaultdict(list)
    names: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    unknown = 0
    for event in gpu:
        start = float(event["ts"])
        end = start + float(event["dur"])
        if end <= t0 or start >= t1:
            continue
        args_ = event.get("args") or {}
        ext = args_.get("External id")
        launcher = launch_tid.get(args_.get("correlation"))
        if ext is not None and ext in ext_tid:
            owner = ext_tid[ext]
        elif launcher in live_tids:
            owner = launcher
        else:
            owner = "unknown"
            unknown += 1
        by_owner[owner].append((max(start, t0), min(end, t1)))
        names[owner][event["name"]].append(float(event["dur"]))
    merged = {owner: merge(intervals) for owner, intervals in by_owner.items()}
    everything = merge([iv for intervals in merged.values() for iv in intervals])
    others = merge(
        [iv for owner, ivs in merged.items() if owner != sched_tid for iv in ivs]
    )
    sched = merged.get(sched_tid, [])
    busy_all = sum(hi - lo for lo, hi in everything)
    print(f"gpu activities with unknown owner: {unknown}")

    def role(tid: str) -> str:
        if tid == sched_tid:
            return "scheduler"
        top = frames[tid].most_common(1)
        return f"tid {tid} ({top[0][0]})" if top else f"tid {tid}"

    print(
        f"\ndevice time by owner thread (union per thread, share of the {window / 1e3:.1f} ms window)"
    )
    print(f"{'owner':64s}{'busy ms':>11s}{'share':>8s}{'kernels':>10s}")
    for owner, intervals in sorted(
        merged.items(), key=lambda item: -sum(hi - lo for lo, hi in item[1])
    ):
        busy = sum(hi - lo for lo, hi in intervals)
        count = sum(len(v) for v in names[owner].values())
        print(
            f"{role(owner)[:63]:64s}{busy / 1e3:>11.2f}{100 * busy / window:>7.1f}%{count:>10d}"
        )
    overlap = sum(hi - lo for lo, hi in intersect(sched, others))
    print(
        f"{'any thread (union)':64s}{busy_all / 1e3:>11.2f}{100 * busy_all / window:>7.1f}%"
    )
    print(
        f"{'device empty':64s}{(window - busy_all) / 1e3:>11.2f}{100 * (window - busy_all) / window:>7.1f}%"
    )
    print(
        f"{'scheduler and another thread at once':64s}{overlap / 1e3:>11.2f}{100 * overlap / window:>7.1f}%"
    )

    starts = {
        key: [iv[0] for iv in ivs]
        for key, ivs in (("sched", sched), ("other", others), ("all", everything))
    }
    classes: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (kind, bs, toks, start, span_end, _), following in zip(steps, steps[1:]):
        end = following[3]
        label = f"bs {bs}" if kind != "prefill" else f"seqs {bs}"
        classes[(kind, label)].append(
            {
                "wall": end - start,
                "span": span_end - start,
                "toks": toks or 0,
                "sched": clipped_us(sched, starts["sched"], start, end),
                "other": clipped_us(others, starts["other"], start, end),
                "idle": (end - start)
                - clipped_us(everything, starts["all"], start, end),
            }
        )
    print(
        "\nscheduler loop wall by step class (ms; sched / other = device time of the scheduler / of other threads inside the step)"
    )
    print(
        f"{'class':22s}{'n':>6s}{'wall sum':>11s}{'share':>8s}{'wall p50':>10s}{'wall mean':>11s}"
        f"{'span mean':>11s}{'sched mean':>12s}{'other mean':>12s}{'idle mean':>11s}{'toks mean':>11s}"
    )
    for (kind, label), rows in sorted(
        classes.items(), key=lambda item: -sum(r["wall"] for r in item[1])
    ):
        walls = [r["wall"] for r in rows]
        cells = [
            statistics.fmean(r[key] for r in rows) / 1e3
            for key in ("span", "sched", "other", "idle")
        ]
        print(
            f"{kind + ' ' + label:22s}{len(rows):>6d}{sum(walls) / 1e3:>11.1f}{100 * sum(walls) / window:>7.1f}%"
            f"{quantile(walls, .5) / 1e3:>10.2f}{statistics.fmean(walls) / 1e3:>11.2f}"
            + "".join(f"{c:>{w}.2f}" for c, w in zip(cells, (11, 12, 12, 11)))
            + f"{statistics.fmean(r['toks'] for r in rows):>11.1f}"
        )
    by_kind: dict[str, float] = defaultdict(float)
    for (kind, _), rows in classes.items():
        by_kind[kind] += sum(r["wall"] for r in rows)
    print(
        "wall share by kind: "
        + ", ".join(f"{k} {100 * v / window:.1f}%" for k, v in sorted(by_kind.items()))
    )

    for owner, _ in sorted(
        merged.items(), key=lambda item: -sum(hi - lo for lo, hi in item[1])
    )[:4]:
        total = sum(sum(v) for v in names[owner].values())
        print(f"\ntop kernels of {role(owner)} (sum of durations {total / 1e3:.1f} ms)")
        for name, durs in sorted(names[owner].items(), key=lambda item: -sum(item[1]))[
            : args.top
        ]:
            print(
                f"  {name[:84]:85s}{len(durs):>8d}{sum(durs) / 1e3:>10.2f} ms{100 * sum(durs) / total:>7.1f}%{statistics.fmean(durs):>9.1f} us"
            )


if __name__ == "__main__":
    main()
