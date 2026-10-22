"""Read one baseline.py run directory and print what each pass measured.

usage: python3 report.py /data/runtime-perf/b1_tts [--cells c16_events,c64_pyspy] [--top 12]

Per cell:
  bench    the benchmark's own summary (throughput, latency, first audio, RTF), whatever the model's benchmark wrote
  threads  CPU seconds per server thread over the cell, as a share of the active load window (the event span, else
           the benchmark's wall clock) in units of one core, grouped by process and named from the cell's py-spy
           dumps; threads under 1 % are folded
  events   request event recorder: per request, the gap between consecutive request-level events (chunk events
           excluded), median and p90 over requests; per chunk, sent to received latency per edge and transport
  pyspy    per sampled thread, its busy share (running samples over rate x duration) or, for a --gil pass, its share
           of lock-holder samples, and the top functions of its samples
"""

import argparse
import collections
import json
import re
import statistics
from pathlib import Path

CHUNK_EVENTS = {"stage_stream_chunk_sent", "stage_stream_chunk_received", "coordinator_stream_received"}


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float("nan")


def flatten(prefix, value, out):
    if isinstance(value, dict):
        if set(value) >= {"mean", "n"} and not isinstance(value["mean"], dict):
            out[prefix] = value["mean"]
        else:
            for key, item in value.items():
                flatten(f"{prefix}.{key}" if prefix else key, item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value
    else:
        pass


def bench_summary(cell: Path) -> None:
    keys = ("throughput", "latency", "audio", "rtf", "wall", "ttf", "first", "request", "completed", "failed", "wer")
    for path in sorted((cell / "bench").rglob("*.json")):
        if path.name.startswith("conc") or "raw" in path.parts:
            continue
        else:
            pass
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        else:
            pass
        flat = {}
        for section in ("results", "summary", "speed", "metrics"):
            if section in data:
                flatten(section, data[section], flat)
            else:
                pass
        if not flat:
            flatten("", data, flat)
        else:
            pass
        shown = {k: v for k, v in flat.items() if any(word in k.lower() for word in keys) and "per_repeat" not in k}
        if shown:
            print(f"  bench {path.relative_to(cell)}")
            for key, value in list(shown.items())[:40]:
                print(f"    {key} = {value:.4g}" if isinstance(value, float) else f"    {key} = {value}")
        else:
            pass


def thread_names(cell: Path) -> dict[str, str]:
    """tid to Python thread name from the cell's py-spy dumps (Python threads keep the OS name python3)."""
    names = {}
    for path in cell.glob("dump_*.txt"):
        for line in path.open(errors="replace"):
            match = re.match(r'Thread (\d+) \((?:active|idle)[^)]*\): "(.*)"', line.strip())
            if match:
                names[match.group(1)] = match.group(2)
            else:
                pass
    return names


def active_seconds(cell: Path, window: dict) -> tuple[float, str]:
    """The load window: event span when events exist, else the benchmark's wall clock, else the cell window."""
    stamps = []
    for path in (cell / "events").glob("*.jsonl") if (cell / "events").exists() else ():
        for line in path.open():
            stamps.append(json.loads(line)["timestamp_ns"])
    if stamps:
        return (max(stamps) - min(stamps)) / 1e9, "event span"
    else:
        pass
    for path in (cell / "bench").rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        else:
            pass
        summary = data.get("summary")
        if isinstance(summary, dict) and summary.get("throughput_qps") and summary.get("completed_requests"):
            return summary["completed_requests"] / summary["throughput_qps"], "bench completed / qps"
        else:
            pass
        results = data.get("results")
        if isinstance(results, list) and results and isinstance(results[0], dict) and "wall_clock_s" in results[0]:
            return sum(r["wall_clock_s"]["mean"] * r.get("repeats", 1) for r in results), "bench wall clock x repeats"
        else:
            pass
        flat = {}
        flatten("", data.get("results", data), flat)
        for key, value in flat.items():
            if key.endswith("wall_clock_s"):
                return float(value), f"bench {key}"
            else:
                pass
    return window["end"] - window["start"], "cell window"


def thread_shares(cell: Path) -> None:
    cpu = json.loads((cell / "thread_cpu.json").read_text())
    window = json.loads((cell / "window.json").read_text())
    span, source = active_seconds(cell, window)
    names = thread_names(cell)
    by_process = collections.defaultdict(list)
    for key, end in cpu["end"].items():
        start = cpu["start"].get(key, {"cpu_s": 0.0})["cpu_s"]
        pid, tid = key.split("/")
        by_process[pid].append((end["cpu_s"] - start, names.get(tid, end["name"])))
    print(f"  threads, active {span:.2f} s ({source}), share of one core")
    for pid, rows in sorted(by_process.items(), key=lambda item: -sum(r[0] for r in item[1])):
        total = sum(r[0] for r in rows)
        if total / span < 0.01:
            continue
        else:
            pass
        named = collections.defaultdict(float)
        for seconds, name in rows:
            named[name] += seconds
        top = sorted(named.items(), key=lambda item: -item[1])
        shown = ", ".join(f"{name} {100 * sec / span:.0f}%" for name, sec in top if sec / span >= 0.01)
        print(f"    pid {pid} total {100 * total / span:.0f}%: {shown}")


def events_summary(cell: Path, top: int) -> None:
    files = list((cell / "events").glob("*.jsonl"))
    if not files:
        return
    else:
        pass
    per_request = collections.defaultdict(list)
    sent, received, coordinator = {}, {}, {}
    for path in files:
        for line in path.open():
            event = json.loads(line)
            rid, name, stage, ts = event["request_id"], event["event_name"], event["stage"], event["timestamp_ns"]
            meta = event.get("metadata") or {}
            if name == "stage_stream_chunk_sent":
                sent[(rid, stage, meta.get("to_stage"), meta.get("chunk_id"))] = (ts, meta.get("transport"))
            elif name == "stage_stream_chunk_received":
                received[(rid, meta.get("from_stage"), stage, meta.get("chunk_id"))] = ts
            elif name == "coordinator_stream_received":
                coordinator.setdefault(rid, []).append(ts)
            else:
                pass
            if name not in CHUNK_EVENTS or name.startswith("stage_first"):
                per_request[rid].append((ts, f"{stage}.{name}"))
            else:
                pass
    gaps = collections.defaultdict(list)
    totals = []
    for rid, events in per_request.items():
        events.sort()
        for (t0, a), (t1, b) in zip(events, events[1:]):
            gaps[f"{a} -> {b}"].append((t1 - t0) / 1e6)
        totals.append((events[-1][0] - events[0][0]) / 1e6)
    print(f"  events: {len(per_request)} requests, first to last event p50 {pct(totals, 0.5):.2f} ms "
          f"p90 {pct(totals, 0.9):.2f} ms")
    common = sorted(gaps.items(), key=lambda item: -len(item[1]))[:top * 2]
    for label, values in sorted(common, key=lambda item: -statistics.median(item[1]))[:top * 2]:
        print(f"    {statistics.median(values):8.2f} ms p50 {pct(values, 0.9):8.2f} p90  n={len(values):6d}  {label}")
    hops = collections.defaultdict(list)
    for key, (ts, transport) in sent.items():
        if key in received:
            hops[f"{key[1]}->{key[2]} {transport}"].append((received[key] - ts) / 1e6)
        else:
            pass
    for label, values in sorted(hops.items()):
        print(f"    chunk hop {label}: p50 {statistics.median(values):.3f} ms p90 {pct(values, 0.9):.3f} "
              f"p99 {pct(values, 0.99):.3f} n={len(values)}")
    gaps_to_coordinator = []
    for rid, stamps in coordinator.items():
        stamps.sort()
        gaps_to_coordinator += [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
    if gaps_to_coordinator:
        print(f"    coordinator inter-chunk gap: p50 {statistics.median(gaps_to_coordinator):.2f} ms "
              f"p90 {pct(gaps_to_coordinator, 0.9):.2f} p99 {pct(gaps_to_coordinator, 0.99):.2f} "
              f"n={len(gaps_to_coordinator)}")
    else:
        pass


def pyspy_summary(cell: Path, top: int) -> None:
    """py-spy without --idle keeps only running threads' samples: samples over rate x duration is a thread's busy
    share. With --gil each tick names the lock holder: a thread's samples over all samples is its ownership share.
    Both checked on a known-answer script (one busy, one sleeping thread) on the box image."""
    meta_path = cell / "pyspy.json"
    if not meta_path.exists():
        return
    else:
        pass
    meta = json.loads(meta_path.read_text())
    ticks = meta["rate"] * meta["duration"]
    for path in sorted(cell.glob("pyspy_*.txt")):
        threads = collections.defaultdict(lambda: [0, collections.Counter()])
        for line in path.open():
            stack, _, count = line.rstrip("\n").rpartition(" ")
            if not count.isdigit():
                continue
            else:
                pass
            frames = stack.split(";")
            thread, frames = frames[0], frames[1:]
            stats = threads[thread]
            stats[0] += int(count)
            seen = set()
            for frame in frames:
                function = frame.split(" (")[0] + " (" + frame.split("/")[-1] if "/" in frame else frame
                if function not in seen:
                    stats[1][function] += int(count)
                    seen.add(function)
        total = sum(samples for samples, _ in threads.values())
        print(f"  {meta['mode']} {path.name} ({total} samples, {ticks} ticks)")
        for thread, (samples, functions) in sorted(threads.items(), key=lambda item: -item[1][0])[:8]:
            share = samples / ticks if meta["mode"] == "pyspy" else samples / max(total, 1)
            label = "busy" if meta["mode"] == "pyspy" else "holds the lock in"
            print(f"    {thread[:70]}: {label} {100 * share:.0f}%")
            for function, count in functions.most_common(top):
                print(f"        {100 * count / samples:5.1f}%  {function[:110]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--cells", default="")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    run = Path(args.run)
    print((run / "identity.txt").read_text())
    print("startup", (run / "startup.txt").read_text().strip() if (run / "startup.txt").exists() else "none")
    wanted = set(args.cells.split(",")) if args.cells else None
    for cell in sorted(path for path in run.iterdir() if path.is_dir() and (path / "window.json").exists()):
        if wanted and cell.name not in wanted:
            continue
        else:
            pass
        print(f"== {cell.name}")
        bench_summary(cell)
        thread_shares(cell)
        events_summary(cell, args.top)
        pyspy_summary(cell, args.top)


if __name__ == "__main__":
    main()
