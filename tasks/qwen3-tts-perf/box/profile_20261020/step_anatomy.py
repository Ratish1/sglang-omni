"""Per talker step anatomy from a formal trace of the profiling branch (one omni.step span per scheduler forward).

usage: python3 step_anatomy.py <trace.json.gz>
For each omni.step span on the scheduler thread: its wall; the CUDA runtime calls made inside it by kind (graph
launches, eager kernel launches, copies, synchronizations) with their counts; the device work those calls launched
(through CUPTI correlation): first start to last end, busy union and its kernel count; and the gap from the span's end
to the next span's start (the scheduler loop between forwards). Medians and means per step kind and batch size.
"""

import bisect
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict

STEP = re.compile(r"omni\.step (\w+) (\w+) fwd=(\d+) bs=(\d+)")
GRAPH = ("cudaGraphLaunch",)
EAGER = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx")
COPY = ("cudaMemcpyAsync", "cudaMemcpy", "cudaMemsetAsync")
SYNC = ("cudaEventSynchronize", "cudaStreamSynchronize", "cudaDeviceSynchronize")
DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


def union_us(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if start > end:
            total, end = total + stop - start, stop
        elif stop > end:
            total, end = total + stop - end, stop
        else:
            pass
    return total


def main() -> None:
    with gzip.open(sys.argv[1], "rt", encoding="utf-8") as handle:
        events = json.load(handle)["traceEvents"]
    steps = sorted(
        (e for e in events if e.get("ph") == "X" and e.get("cat") == "user_annotation" and STEP.match(str(e.get("name")))),
        key=lambda e: e["ts"],
    )
    thread = steps[0]["tid"]
    runtime = sorted(
        (e for e in events if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver") and e["tid"] == thread),
        key=lambda e: e["ts"],
    )
    runtime_starts = [e["ts"] for e in runtime]
    traced = sorted(
        (e["ts"], e["ts"] + e["dur"])
        for e in events
        if e.get("ph") == "X" and e["tid"] == thread and e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver")
    )
    traced_starts = [start for start, _ in traced]

    def traced_ms(start: float, end: float) -> float:
        """Host time on the scheduler thread inside an aten op or a CUDA call, clipped to [start, end]."""
        low = max(0, bisect.bisect_left(traced_starts, start) - 64)
        clipped = [
            (max(a, start), min(b, end)) for a, b in traced[low:bisect.bisect_right(traced_starts, end)] if b > start
        ]
        return union_us([span for span in clipped if span[1] > span[0]]) / 1e3
    device_by_correlation: dict[int, list[dict]] = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in DEVICE_CATEGORIES:
            device_by_correlation[e.get("args", {}).get("correlation")].append(e)
        else:
            pass
    rows: dict[tuple[str, int], list[dict[str, float]]] = defaultdict(list)
    for index, step in enumerate(steps):
        kind, _, _, batch = STEP.match(step["name"]).groups()
        start, end = step["ts"], step["ts"] + step["dur"]
        calls = runtime[bisect.bisect_left(runtime_starts, start):bisect.bisect_right(runtime_starts, end)]
        row: dict[str, float] = defaultdict(float)
        spans: list[tuple[float, float]] = []
        graph_index = 0
        for call in calls:
            name = call["name"]
            group = (
                "graph" if name in GRAPH else "eager" if name in EAGER else "copy" if name in COPY
                else "sync" if name in SYNC else "other"
            )
            row[f"{group} ms"] += call["dur"] / 1e3
            row[f"{group} n"] += 1
            launched = [
                (device["ts"], device["ts"] + device["dur"])
                for device in device_by_correlation.get(call.get("args", {}).get("correlation"), ())
            ]
            spans.extend(launched)
            if group == "graph" and launched:
                graph_index += 1
                row[f"graph{graph_index} busy ms"] = union_us(launched) / 1e3
                row[f"graph{graph_index} kernels n"] = len(launched)
            else:
                pass
        row["wall ms"] = step["dur"] / 1e3
        row["untraced in step ms"] = row["wall ms"] - traced_ms(start, end)
        row["kernels n"] = len(spans)
        if spans:
            row["device span ms"] = (max(s for _, s in spans) - min(s for s, _ in spans)) / 1e3
            row["device busy ms"] = union_us(spans) / 1e3
            row["start to first kernel ms"] = (min(s for s, _ in spans) - start) / 1e3
        else:
            pass
        if index + 1 < len(steps):
            row["gap to next ms"] = (steps[index + 1]["ts"] - end) / 1e3
            row["untraced in gap ms"] = row["gap to next ms"] - traced_ms(end, steps[index + 1]["ts"])
        else:
            pass
        rows[(kind, int(batch))].append(row)
    columns = (
        "wall ms", "untraced in step ms", "gap to next ms", "untraced in gap ms", "device busy ms", "device span ms",
        "start to first kernel ms", "kernels n",
        "graph1 busy ms", "graph1 kernels n", "graph2 busy ms", "graph2 kernels n",
        "graph n", "graph ms", "eager n", "eager ms", "copy n", "copy ms", "sync n", "sync ms", "other ms",
    )
    print(f"scheduler thread {thread}, {len(steps)} steps")
    print("| kind | bs | steps | " + " | ".join(f"{c} (median / mean)" for c in columns) + " |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in columns) + " |")
    for (kind, batch), items in sorted(rows.items()):
        cells = []
        for column in columns:
            values = [item.get(column, 0.0) for item in items]
            cells.append(f"{statistics.median(values):.3f} / {statistics.fmean(values):.3f}")
        print(f"| {kind} | {batch} | {len(items)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
