"""Two torch profiler traces of the same process side by side (run on the box): per kernel
name the launches and mean duration in each, and per stream the kernel time, the kernel
union and the gaps between consecutive kernels, so a slower process can be split into
slower kernels against slower launches.

usage: python trace_kernel_compare.py A.trace.json.gz B.trace.json.gz [--top 25]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import statistics


def load(path: str) -> list[tuple[str, int, float, float]]:
    with gzip.open(path) as handle:
        events = json.load(handle)["traceEvents"]
    return [
        (
            event["name"],
            int(event["args"].get("stream", -1)),
            float(event["ts"]),
            float(event["dur"]),
        )
        for event in events
        if event.get("cat") == "kernel" and "dur" in event
    ]


def summary(kernels: list[tuple[str, int, float, float]]) -> tuple[dict, dict]:
    by_name: dict[str, list[float]] = collections.defaultdict(list)
    by_stream: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)
    for name, stream, start, duration in kernels:
        by_name[name].append(duration)
        by_stream[stream].append((start, start + duration))
    return by_name, by_stream


def stream_line(spans: list[tuple[float, float]]) -> str:
    spans.sort()
    busy = sum(end - start for start, end in spans)
    gaps = [
        spans[index + 1][0] - spans[index][1]
        for index in range(len(spans) - 1)
        if spans[index + 1][0] > spans[index][1]
    ]
    window = spans[-1][1] - spans[0][0]
    median_gap = statistics.median(gaps) if gaps else 0.0
    return (
        f"kernels {len(spans):8d} kernel ms {busy / 1e3:9.1f} window ms {window / 1e3:9.1f} "
        f"gaps ms {sum(gaps) / 1e3:9.1f} median gap us {median_gap:6.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    names_a, streams_a = summary(load(args.a))
    names_b, streams_b = summary(load(args.b))
    print("per stream (A then B)")
    for stream in sorted(set(streams_a) | set(streams_b)):
        if stream in streams_a:
            print(f"  A stream {stream:4d} {stream_line(streams_a[stream])}")
        else:
            pass
        if stream in streams_b:
            print(f"  B stream {stream:4d} {stream_line(streams_b[stream])}")
        else:
            pass
    print(f"top {args.top} kernels by A total: launches A / B, mean us A / B, change")
    ranked = sorted(names_a, key=lambda name: -sum(names_a[name]))[: args.top]
    for name in ranked:
        a, b = names_a[name], names_b.get(name, [])
        mean_a = statistics.fmean(a)
        mean_b = statistics.fmean(b) if b else float("nan")
        print(
            f"  {len(a):7d} / {len(b):7d}  {mean_a:8.2f} / {mean_b:8.2f}  {(mean_b / mean_a - 1) * 100:+6.1f} %  {name[:70]}"
        )


if __name__ == "__main__":
    main()
