"""Block 8 reader: the talker's steps beside a reference encode against the steps without one.

nsys records no CUDA call from the reference batcher thread (`qwen3-tts-ref-code`), so the census cannot own the
encode's kernels; the probe's ranges on that thread still bound each encode: from a batch's first `ref.encode` start
to its `ref.sync` end the encode is queued or running on the device. Each `mr.execute` range (one talker and predictor
step) owns the device intervals of the runtime calls inside it on its thread; a step is beside an encode when its
device span overlaps an encode window. Printed per arm: steps, mean device time (union of its intervals) and span, both
groups, over the benchmark's window.

python3 encode_overlap.py RUN_DIR [RUN_DIR ...]   (each with serve.sqlite and bench.log)
"""

import argparse
import bisect
import sqlite3
import statistics
import sys
from pathlib import Path

PUSH_POP = 59
SCRIPTS = Path(__file__).resolve().parents[3] / "omni_step_profiling_20260918" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from nsys_metrics import bench_window  # noqa: E402


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total, current_start, current_end = 0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            else:
                pass
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    else:
        pass
    return total


def read_run(run_dir: Path) -> None:
    db = sqlite3.connect(str(run_dir / "serve.sqlite"))
    strings = dict(db.execute("select id, value from StringIds"))
    window_start, window_end = bench_window(db, str(run_dir / "bench.log"))
    ranges = [
        (start, end, tid, text or strings.get(text_id, ""))
        for start, end, tid, text, text_id in db.execute(
            "select start, end, globalTid, text, textId from NVTX_EVENTS where eventType = ? and end > ? and start < ?",
            (PUSH_POP, window_start, window_end),
        )
    ]
    encode_starts: dict[int, list[int]] = {}
    windows = []
    for start, end, tid, label in sorted(ranges):
        if label.startswith("ref.encode"):
            encode_starts.setdefault(tid, []).append(start)
        elif label.startswith("ref.sync") and encode_starts.get(tid):
            windows.append((encode_starts[tid][0], end))
            encode_starts[tid] = []
        else:
            pass
    windows.sort()
    window_begins = [begin for begin, _ in windows]
    steps = [(start, end, tid) for start, end, tid, label in ranges if label.startswith("mr.execute")]
    calls_by_tid: dict[int, list[tuple[int, int]]] = {}
    for start, tid, correlation in db.execute(
        "select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where end > ? and start < ?",
        (window_start, window_end),
    ):
        calls_by_tid.setdefault(tid, []).append((start, correlation))
    for calls in calls_by_tid.values():
        calls.sort()
    device: dict[int, list[tuple[int, int]]] = {}
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        for start, end, correlation in db.execute(
            f"select start, end, correlationId from {table} where end > ? and start < ?", (window_start, window_end)
        ):
            device.setdefault(correlation, []).append((start, end))
    groups: dict[str, list[tuple[float, float]]] = {"beside an encode": [], "no encode": []}
    for start, end, tid in steps:
        calls = calls_by_tid.get(tid, [])
        low = bisect.bisect_left(calls, (start, -1))
        high = bisect.bisect_right(calls, (end, 1 << 62))
        intervals = [interval for _, correlation in calls[low:high] for interval in device.get(correlation, [])]
        if not intervals:
            continue
        else:
            pass
        span_start = min(begin for begin, _ in intervals)
        span_end = max(finish for _, finish in intervals)
        index = bisect.bisect_right(window_begins, span_end)
        is_beside = any(windows[candidate][1] > span_start for candidate in range(max(index - 8, 0), index))
        groups["beside an encode" if is_beside else "no encode"].append(
            (union_ns(intervals) / 1e6, (span_end - span_start) / 1e6)
        )
    encode_ms = [(end - begin) / 1e6 for begin, end in windows]
    print(
        f"== {run_dir.name}: window {(window_end - window_start) / 1e9:.1f} s, encode windows {len(windows)} "
        f"(p50 {statistics.median(encode_ms):.2f} ms, union {union_ns(windows) / 1e6:.0f} ms)"
    )
    for name, values in groups.items():
        if values:
            print(
                f"   {name:17s} steps {len(values):5d}  device ms mean {statistics.mean(v[0] for v in values):.3f} "
                f"p50 {statistics.median(v[0] for v in values):.3f}  span ms mean {statistics.mean(v[1] for v in values):.3f}"
            )
        else:
            print(f"   {name:17s} steps 0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    for run_dir in parser.parse_args().runs:
        read_run(run_dir)


if __name__ == "__main__":
    main()
