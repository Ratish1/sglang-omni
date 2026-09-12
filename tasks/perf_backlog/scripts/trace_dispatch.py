"""Dispatch delay of kernels launched onto an idle stream, per stream.

Usage: python trace_dispatch.py <trace.json.gz> <out.json>

A kernel whose launch call started after every earlier kernel on its stream had
finished had nothing of its own queued ahead of it, so its start delay is what
the device scheduler imposed from other streams.
"""

import json
import sys
from collections import defaultdict

from trace_streams import parse, pct, raw_events

LAUNCH_PREFIXES = (
    "cudaLaunchKernel",
    "cudaGraphLaunch",
    "cudaMemcpyAsync",
    "cudaMemsetAsync",
)


def main():
    path, out = sys.argv[1], sys.argv[2]
    launch_ts = {}
    kernels = defaultdict(list)
    for text in raw_events(path):
        if '"cuda_runtime"' in text:
            ev = parse(text)
            if ev is None:
                continue
            if ev.get("name", "").startswith(LAUNCH_PREFIXES):
                corr = (ev.get("args") or {}).get("correlation")
                if corr is not None:
                    launch_ts[corr] = ev["ts"]
        elif '"cat": "kernel"' in text or '"cat": "gpu_memcpy"' in text:
            ev = parse(text)
            if ev is None:
                continue
            args = ev.get("args") or {}
            kernels[args.get("stream")].append(
                (
                    ev["ts"],
                    ev.get("dur", 0.0),
                    args.get("correlation"),
                    bool(args.get("graph id")),
                )
            )
    result = {}
    for stream, items in kernels.items():
        items.sort()
        idle_lat = []
        busy_lat = []
        prev_end = None
        for ts, dur, corr, in_graph in items:
            lts = launch_ts.get(corr)
            if lts is not None and not in_graph:
                lat = ts - lts
                if prev_end is None or lts >= prev_end:
                    idle_lat.append(lat)
                else:
                    busy_lat.append(lat)
            prev_end = ts + dur if prev_end is None else max(prev_end, ts + dur)
        result[str(stream)] = {
            "kernels": len(items),
            "idle_stream_dispatch_us": {
                "n": len(idle_lat),
                "p50": pct(idle_lat, 0.5),
                "p90": pct(idle_lat, 0.9),
                "p99": pct(idle_lat, 0.99),
                "mean": (sum(idle_lat) / len(idle_lat)) if idle_lat else None,
            },
            "queued_behind_own_us": {
                "n": len(busy_lat),
                "p50": pct(busy_lat, 0.5),
                "p90": pct(busy_lat, 0.9),
                "p99": pct(busy_lat, 0.99),
            },
        }
    with open(out, "w") as handle:
        json.dump(result, handle, indent=1)
    print("done", path)


if __name__ == "__main__":
    main()
