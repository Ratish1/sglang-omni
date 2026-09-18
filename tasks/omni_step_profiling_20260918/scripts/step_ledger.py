"""Per-iteration ledger of an sglang-omni torch trace with omni.step spans.

A step runs from the start of one omni.step span on the scheduler thread to the start
of the next (the scheduler loop is serial). GPU activity (kernels, memcpy, memset) is
owned by a thread: the thread of its cpu op through External id when it has one,
else the thread of its launch call (graph replays), trusted only for threads that own
cpu ops in the same trace. Counts come from every event; nothing is cut by share.

usage: python step_ledger.py TRACE.json.gz [--top 25]
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

SPAN = re.compile(r"^omni\.step (\S+) (\S+) fwd=(\d+) bs=(\d+)(?: toks=(\d+))?$")
LAUNCH_NAMES = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx")
SYNC_NAMES = (
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaDeviceSynchronize",
    "cudaMemcpy",
    "cudaStreamWaitEvent",
    "cudaEventQuery",
)
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


@dataclass
class Step:
    kind: str
    phase: str
    fwd: int
    bs: int
    toks: int | None
    start: float
    span_end: float
    end: float = 0.0
    counts: Counter = field(default_factory=Counter)
    sync_us: float = 0.0
    graph_kernels: list[int] = field(default_factory=list)
    graph_us: list[float] = field(default_factory=list)
    device_us: dict[str, float] = field(default_factory=dict)


def union_us(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    cur_start = cur_end = None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as handle:
        events = json.load(handle)["traceEvents"]

    ext_tid: dict[int, str] = {}
    launches: dict[int, dict] = {}
    runtime: list[dict] = []
    gpu: list[dict] = []
    spans: list[tuple[str, dict]] = []
    frames: dict[str, Counter] = defaultdict(Counter)
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = event.get("cat")
        args_ = event.get("args") or {}
        tid = str(event.get("tid"))
        if cat in ("cpu_op", "user_annotation"):
            ext = args_.get("External id")
            if ext is not None:
                ext_tid.setdefault(ext, tid)
            if cat == "user_annotation" and event["name"].startswith("omni.step "):
                spans.append((tid, event))
        elif cat in ("cuda_runtime", "cuda_driver"):
            runtime.append(event)
            if args_.get("correlation") is not None:
                launches[args_["correlation"]] = event
        elif cat in GPU_CATS:
            gpu.append(event)
        elif cat == "python_function" and "sglang_omni/" in event["name"]:
            frames[tid][event["name"].split("sglang_omni/")[1].split("(")[0]] += 1

    live_tids = set(ext_tid.values())
    owner_counts: Counter = Counter()
    mismatches = 0
    for event in gpu:
        args_ = event.get("args") or {}
        launch = launches.get(args_.get("correlation"))
        ext = args_.get("External id")
        if ext is None and launch is not None:
            ext = (launch.get("args") or {}).get("External id")
        launch_tid = str(launch["tid"]) if launch is not None else None
        if ext is not None and ext in ext_tid:
            event["owner"] = ext_tid[ext]
            owner_counts["by cpu op"] += 1
            if launch_tid is not None and launch_tid != event["owner"]:
                mismatches += 1
        elif launch_tid in live_tids:
            event["owner"] = launch_tid
            owner_counts["by launch thread"] += 1
        else:
            event["owner"] = "unknown"
            owner_counts["unknown"] += 1
        event["launch_ts"] = launch["ts"] if launch is not None else event["ts"]

    sched_tids = Counter(tid for tid, _ in spans)
    if not sched_tids:
        raise SystemExit("no omni.step spans in the trace")
    sched_tid = sched_tids.most_common(1)[0][0]
    steps: list[Step] = []
    for tid, event in sorted(spans, key=lambda item: item[1]["ts"]):
        match = SPAN.match(event["name"])
        if tid != sched_tid or match is None:
            continue
        kind, phase, fwd, bs, toks = match.groups()
        steps.append(
            Step(
                kind=kind,
                phase=phase,
                fwd=int(fwd),
                bs=int(bs),
                toks=int(toks) if toks else None,
                start=float(event["ts"]),
                span_end=float(event["ts"]) + float(event["dur"]),
            )
        )
    for current, following in zip(steps, steps[1:]):
        current.end = following.start
    steps[-1].end = steps[-1].span_end

    def role(tid: str) -> str:
        if tid == sched_tid:
            return "scheduler"
        top = frames[tid].most_common(1)
        return f"tid {tid} ({top[0][0]})" if top else f"tid {tid}"

    runtime.sort(key=lambda event: event["ts"])
    gpu.sort(key=lambda event: event["launch_ts"])
    graph_launch_corr = {
        (e.get("args") or {}).get("correlation")
        for e in runtime
        if e["name"] == "cudaGraphLaunch" and str(e["tid"]) == sched_tid
    }
    graph_names: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    graph_steps: Counter = Counter()
    for step in steps:
        per_graph: Counter = Counter()
        graph_us: Counter = Counter()
        graph_events: dict[int, list[dict]] = defaultdict(list)
        busy_all: list[tuple[float, float]] = []
        busy_by_role: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for event in gpu:
            start = float(event["ts"])
            end = start + float(event["dur"])
            if end > step.start and start < step.end:
                clipped = (max(start, step.start), min(end, step.end))
                busy_all.append(clipped)
                busy_by_role[role(event["owner"])].append(clipped)
            if event["owner"] != sched_tid or not (step.start <= event["launch_ts"] < step.end):
                continue
            step.counts[event["cat"]] += 1
            corr = (event.get("args") or {}).get("correlation")
            if corr in graph_launch_corr:
                per_graph[corr] += 1
                graph_us[corr] += float(event["dur"])
                graph_events[corr].append(event)
            if event["cat"] == "gpu_memcpy":
                step.counts[event["name"].split(" (")[0]] += 1
        step.graph_kernels = [per_graph[c] for c in sorted(per_graph)]
        step.graph_us = [graph_us[c] for c in sorted(per_graph)]
        for index, corr in enumerate(sorted(per_graph)):
            graph_steps[index] += 1
            for event in graph_events[corr]:
                graph_names[index][event["name"]].append(float(event["dur"]))
        step.device_us = {"all": union_us(busy_all)}
        for name, intervals in busy_by_role.items():
            step.device_us[name] = union_us(intervals)
        for event in runtime:
            if str(event["tid"]) != sched_tid or not (step.start <= event["ts"] < step.end):
                continue
            name = event["name"]
            if name in LAUNCH_NAMES or name == "cudaGraphLaunch":
                step.counts[name] += 1
            if name in SYNC_NAMES:
                step.counts[name] += 1
                step.sync_us += float(event["dur"])

    fwds = [step.fwd for step in steps]
    print(f"trace {args.trace}")
    print(f"scheduler tid {sched_tid}; omni.step spans {len(steps)}")
    print(f"span kinds {dict(Counter((s.kind, s.phase, s.bs) for s in steps))}")
    print(f"fwd range {min(fwds)}..{max(fwds)} contiguous {fwds == list(range(min(fwds), max(fwds) + 1))}")
    print(f"gpu activity owner resolution {dict(owner_counts)}; cpu-op vs launch-thread mismatches {mismatches}")
    print(f"threads owning gpu work: {dict(Counter(role(e['owner']) for e in gpu if e['owner'] != 'unknown'))}")

    modal = Counter((s.kind, s.bs) for s in steps).most_common(1)[0][0]
    steady = [s for s in steps[:-1] if (s.kind, s.bs) == modal]
    print(f"\nsteady steps: kind {modal[0]} bs {modal[1]}, n={len(steady)} (last step excluded: no following start)")
    rows = {
        "wall ms": [(s.end - s.start) / 1e3 for s in steady],
        "span ms": [(s.span_end - s.start) / 1e3 for s in steady],
        "device busy ms (all threads)": [s.device_us["all"] / 1e3 for s in steady],
        "device idle ms": [(s.end - s.start - s.device_us["all"]) / 1e3 for s in steady],
        "scheduler device busy ms": [s.device_us.get("scheduler", 0.0) / 1e3 for s in steady],
        "sync calls blocked ms": [s.sync_us / 1e3 for s in steady],
    }
    for key in sorted({k for s in steady for k in s.counts}):
        rows[f"count {key}"] = [float(s.counts.get(key, 0)) for s in steady]
    for name in sorted({k for s in steady for k in s.device_us} - {"all", "scheduler"}):
        rows[f"device busy ms {name}"] = [s.device_us.get(name, 0.0) / 1e3 for s in steady]
    print(f"{'read':52s}{'p50':>10s}{'p95':>10s}{'mean':>10s}{'min':>10s}{'max':>10s}")
    for key, values in rows.items():
        print(
            f"{key[:51]:52s}{quantile(values, .5):>10.3f}{quantile(values, .95):>10.3f}"
            f"{statistics.fmean(values):>10.3f}{min(values):>10.3f}{max(values):>10.3f}"
        )
    graph_shapes = Counter(tuple(s.graph_kernels) for s in steady)
    print(f"kernels per scheduler graph replay, per step: {dict(graph_shapes.most_common(5))}")
    for index in sorted(graph_steps):
        kernel_ms = [s.graph_us[index] / 1e3 for s in steady if len(s.graph_us) > index]
        print(
            f"\nscheduler graph replay #{index} (launch order in the step), all steps: "
            f"summed kernel ms per replay p50 {quantile(kernel_ms, .5):.3f} "
            f"mean {statistics.fmean(kernel_ms):.3f}"
        )
        names = graph_names[index]
        replays = graph_steps[index]
        for name, durs in sorted(names.items(), key=lambda item: sum(item[1]), reverse=True)[:8]:
            print(
                f"  {name[:66]:67s}{len(durs) / replays:>8.1f}/replay"
                f"{sum(durs) / replays / 1e3:>9.3f} ms/replay{statistics.fmean(durs):>9.2f} us"
            )

    window_start, window_end = steady[0].start, steady[-1].end
    by_name: dict[str, list[float]] = defaultdict(list)
    for event in gpu:
        if window_start <= float(event["ts"]) < window_end:
            by_name[event["name"]].append(float(event["dur"]))
    total = sum(sum(v) for v in by_name.values())
    print(f"\ngpu activity in the steady window: {total / 1e3:.3f} ms over {sum(len(v) for v in by_name.values())} events, {len(by_name)} names")
    print(f"{'name':70s}{'n':>8s}{'ms':>10s}{'share':>8s}{'mean us':>10s}")
    ranked = sorted(by_name.items(), key=lambda item: sum(item[1]), reverse=True)
    for name, durs in ranked[: args.top]:
        print(f"{name[:69]:70s}{len(durs):>8d}{sum(durs) / 1e3:>10.3f}{100 * sum(durs) / total:>7.1f}%{statistics.fmean(durs):>10.2f}")
    rest = ranked[args.top :]
    if rest:
        rest_us = sum(sum(v) for _, v in rest)
        print(f"{'rest: ' + str(len(rest)) + ' names':70s}{sum(len(v) for _, v in rest):>8d}{rest_us / 1e3:>10.3f}{100 * rest_us / total:>7.1f}%")


if __name__ == "__main__":
    main()
