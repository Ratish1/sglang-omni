"""Stream a Kineto trace gz twice and aggregate per stream and per thread timing.

Usage: python trace_streams.py <trace.json.gz> <out.json>

Pass 1 records the host timestamp of every launch call by correlation id.
Pass 2 aggregates kernels per stream (busy time, launch to start latency, per
kernel durations) and launch calls per host thread (call durations, gaps).
"""

import gzip
import json
import sys
from collections import defaultdict

LAUNCH_PREFIXES = (
    "cudaLaunchKernel",
    "cudaGraphLaunch",
    "cudaMemcpyAsync",
    "cudaMemsetAsync",
)


def raw_events(path):
    buf = []
    depth = 0
    started = False
    with gzip.open(path, "rt") as handle:
        for line in handle:
            stripped = line.strip()
            if not started:
                started = '"traceEvents"' in stripped
                continue
            if depth == 0:
                if stripped == "{":
                    buf = [line]
                    depth = 1
                continue
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                depth = 0
                yield "".join(buf).rstrip().rstrip(",")
                buf = []


def parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[int(q * (len(values) - 1))]


def stats(values):
    return {
        "n": len(values),
        "sum_ms": sum(values) / 1000.0,
        "p50_us": pct(values, 0.5),
        "p90_us": pct(values, 0.9),
        "p99_us": pct(values, 0.99),
    }


def main():
    path, out = sys.argv[1], sys.argv[2]
    launch_ts = {}
    names = {}
    threads = defaultdict(
        lambda: {
            "launch_dur": [],
            "graph_dur": [],
            "gaps": [],
            "prev_end": None,
            "first": None,
            "last": None,
            "sync_dur": [],
            "sync_names": defaultdict(int),
        }
    )
    for text in raw_events(path):
        if '"cuda_runtime"' in text:
            ev = parse(text)
            if ev is None:
                continue
            name = ev.get("name", "")
            key = (ev.get("pid"), ev.get("tid"))
            th = threads[key]
            ts, dur = ev["ts"], ev.get("dur", 0.0)
            if name.startswith(LAUNCH_PREFIXES):
                corr = (ev.get("args") or {}).get("correlation")
                if corr is not None:
                    launch_ts[corr] = (ts, key[1])
                if name.startswith("cudaGraphLaunch"):
                    th["graph_dur"].append(dur)
                else:
                    th["launch_dur"].append(dur)
                if th["first"] is None:
                    th["first"] = ts
                th["last"] = ts + dur
                if th["prev_end"] is not None:
                    gap = ts - th["prev_end"]
                    if 0 < gap < 50000:
                        th["gaps"].append(gap)
                th["prev_end"] = ts + dur
            elif "Synchronize" in name or "EventQuery" in name or "StreamWait" in name:
                th["sync_dur"].append(dur)
                th["sync_names"][name] += 1
        elif '"thread_name"' in text or '"process_name"' in text:
            ev = parse(text)
            if ev is not None and ev.get("ph") == "M":
                names[(ev.get("pid"), ev.get("tid"))] = ev["args"]["name"]
    print("pass1 done, launches", len(launch_ts), file=sys.stderr)

    streams = defaultdict(
        lambda: {
            "count": 0,
            "busy_us": 0.0,
            "lat": [],
            "first": None,
            "last": None,
            "kernels": defaultdict(lambda: [0, 0.0]),
            "channels": defaultdict(int),
        }
    )
    thread_streams = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    thread_kernels = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    t_min, t_max = None, None
    n = 0
    for text in raw_events(path):
        if '"ph": "X"' not in text:
            continue
        if (
            '"cuda_runtime"' in text
            or '"cpu_op"' in text
            or '"user_annotation"' in text
            or '"python_function"' in text
        ):
            continue
        ev = parse(text)
        if ev is None:
            continue
        cat = ev.get("cat")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        n += 1
        ts, dur = ev["ts"], ev.get("dur", 0.0)
        if t_min is None or ts < t_min:
            t_min = ts
        if t_max is None or ts + dur > t_max:
            t_max = ts + dur
        args = ev.get("args") or {}
        key = (ev.get("pid"), ev.get("tid"), args.get("stream"))
        st = streams[key]
        st["count"] += 1
        st["busy_us"] += dur
        if st["first"] is None:
            st["first"] = ts
        st["last"] = max(st["last"] or 0, ts + dur)
        st["channels"][args.get("channel")] += 1
        kn = ev.get("name", "")[:100]
        pk = st["kernels"][kn]
        pk[0] += 1
        pk[1] += dur
        corr = args.get("correlation")
        if corr is not None and corr in launch_ts:
            lts, tid = launch_ts[corr]
            if not args.get("graph id"):
                st["lat"].append(ts - lts)
            tsr = thread_streams[tid][args.get("stream")]
            tsr[0] += 1
            tsr[1] += dur
            tk = thread_kernels[tid][kn]
            tk[0] += 1
            tk[1] += dur
    print("pass2 done, kernels", n, file=sys.stderr)

    summary = {
        "kernel_events": n,
        "window_ms": (t_max - t_min) / 1000.0 if t_min is not None else None,
        "streams": {},
        "threads": {},
    }
    for key, st in streams.items():
        top = sorted(st["kernels"].items(), key=lambda kv: -kv[1][1])[:30]
        summary["streams"][f"{key[0]}/{key[1]}/stream{key[2]}"] = {
            "kernels": st["count"],
            "busy_ms": st["busy_us"] / 1000.0,
            "span_ms": (
                (st["last"] - st["first"]) / 1000.0 if st["first"] is not None else None
            ),
            "eager_launch_latency": stats(st["lat"]),
            "channels": dict(st["channels"]),
            "top_kernels": [
                {
                    "name": k,
                    "count": v[0],
                    "total_ms": v[1] / 1000.0,
                    "mean_us": v[1] / v[0],
                }
                for k, v in top
            ],
        }
    for key, th in threads.items():
        summary["threads"][f"{key[0]}/{key[1]}"] = {
            "thread_name": names.get(key),
            "span_ms": (
                (th["last"] - th["first"]) / 1000.0 if th["first"] is not None else None
            ),
            "launch_call": stats(th["launch_dur"]),
            "graph_launch_call": stats(th["graph_dur"]),
            "gap_between_launches": stats(th["gaps"]),
            "gap_over_1ms": stats([g for g in th["gaps"] if g > 1000]),
            "sync_calls": stats(th["sync_dur"]),
            "sync_names": dict(th["sync_names"]),
            "kernels_by_stream": {
                str(s): {"count": v[0], "busy_ms": v[1] / 1000.0}
                for s, v in thread_streams[key[1]].items()
            },
            "top_kernels": [
                {"name": k, "count": v[0], "total_ms": v[1] / 1000.0}
                for k, v in sorted(
                    thread_kernels[key[1]].items(), key=lambda kv: -kv[1][1]
                )[:8]
            ],
        }
    summary["names"] = {f"{k[0]}/{k[1]}": v for k, v in names.items()}
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=1)
    print("done", path, "kernels", n, "window_ms", summary["window_ms"])


if __name__ == "__main__":
    main()
