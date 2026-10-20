"""Gates, streams and launching threads of one torch profiler trace (PLAN.md 4 and 5.2).

usage: python3 trace_streams.py <trace.json.gz> [--top 6]
Prints: the gate counts (graph capture, compile, lazy loading inside the window, graph launches); the window and the
device busy union over every stream; per stream its kernel count, kernel sum, busy union, the threads that launched
onto it and its top kernels; per launching thread its CUDA runtime time by call (launches, syncs, copies); and per
stream how much of its busy time ran while another stream was busy. Sum, union and wall stay apart.
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

# note (ratish): "Torch-Compiled Region" and "TorchDynamo Cache Lookup" mark a compiled function running, not
# compiling; only Dynamo's timed compile regions mean a compile inside the window.
GATE_PATTERNS = {
    "graph capture": ("cudaStreamBeginCapture", "cudaGraphInstantiate", "cudaStreamEndCapture"),
    "compile": ("dynamo_timed", "compile_fx", "OutputGraph.call_user_compiler"),
    "lazy load": ("Lazy Function Loading", "cuModuleLoad", "cuLibraryLoad"),
}
DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


def union_us(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
        else:
            pass
    return total


def merged(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    spans: list[list[float]] = []
    for start, stop in sorted(intervals):
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], stop)
        else:
            spans.append([start, stop])
    return [(start, stop) for start, stop in spans]


def overlap_us(spans: list[tuple[float, float]], others: list[tuple[float, float]]) -> float:
    total = 0.0
    j = 0
    for start, stop in spans:
        while j < len(others) and others[j][1] <= start:
            j += 1
        k = j
        while k < len(others) and others[k][0] < stop:
            total += max(0.0, min(stop, others[k][1]) - max(start, others[k][0]))
            k += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt", encoding="utf-8") as handle:
        events = json.load(handle)["traceEvents"]
    launches: dict[int, tuple[int, str]] = {}
    runtime_by_tid: dict[int, Counter[str]] = defaultdict(Counter)
    runtime_count_by_tid: dict[int, Counter[str]] = defaultdict(Counter)
    gates: Counter[str] = Counter()
    device: list[dict] = []
    for event in events:
        if event.get("ph") != "X":
            continue
        else:
            pass
        name = str(event.get("name", ""))
        category = event.get("cat", "")
        for gate, patterns in GATE_PATTERNS.items():
            if any(pattern in name for pattern in patterns):
                gates[gate] += 1
            else:
                pass
        if category == "cuda_runtime" or category == "cuda_driver":
            runtime_by_tid[event["tid"]][name] += event.get("dur", 0)
            runtime_count_by_tid[event["tid"]][name] += 1
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation] = (event["tid"], name)
            else:
                pass
        elif category in DEVICE_CATEGORIES:
            device.append(event)
        else:
            pass
    graph_launches = sum(counter["cudaGraphLaunch"] for counter in runtime_count_by_tid.values())
    print("gates:", dict(gates), "| cudaGraphLaunch", graph_launches)
    if not device:
        print("no device events")
        return
    else:
        pass
    window = (min(e["ts"] for e in device), max(e["ts"] + e["dur"] for e in device))
    all_spans = [(e["ts"], e["ts"] + e["dur"]) for e in device]
    print(
        f"window {(window[1] - window[0]) / 1e3:.1f} ms | device busy union {union_us(all_spans) / 1e3:.1f} ms "
        f"| device event sum {sum(e['dur'] for e in device) / 1e3:.1f} ms | events {len(device)}"
    )
    by_stream: dict[int, list[dict]] = defaultdict(list)
    for event in device:
        by_stream[event.get("args", {}).get("stream", -1)].append(event)
    stream_spans = {stream: merged([(e["ts"], e["ts"] + e["dur"]) for e in items]) for stream, items in by_stream.items()}
    print("\n| stream | events | sum ms | union ms | shared with another stream ms | launch threads | top kernels (ms) |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for stream, items in sorted(by_stream.items(), key=lambda pair: -sum(e["dur"] for e in pair[1])):
        others = merged([span for other, spans in stream_spans.items() if other != stream for span in spans])
        threads = Counter(launches.get(e.get("args", {}).get("correlation"), (None, None))[0] for e in items)
        kernels: Counter[str] = Counter()
        for e in items:
            kernels[e["name"][:60]] += e["dur"]
        top = "; ".join(f"{name} {dur / 1e3:.2f}" for name, dur in kernels.most_common(args.top))
        print(
            f"| {stream} | {len(items)} | {sum(e['dur'] for e in items) / 1e3:.2f} | "
            f"{union_us([(e['ts'], e['ts'] + e['dur']) for e in items]) / 1e3:.2f} | "
            f"{overlap_us(stream_spans[stream], others) / 1e3:.2f} | "
            f"{', '.join(f'{tid}:{count}' for tid, count in threads.most_common(4))} | {top} |"
        )
    print("\n| thread | runtime ms by call (top) |")
    print("| --- | --- |")
    for tid, counter in sorted(runtime_by_tid.items(), key=lambda pair: -sum(pair[1].values())):
        calls = "; ".join(
            f"{name} {dur / 1e3:.2f} ({runtime_count_by_tid[tid][name]})" for name, dur in counter.most_common(args.top)
        )
        print(f"| {tid} | {calls} |")


if __name__ == "__main__":
    main()
