"""Exclusive Python time per function on one thread of a mapping trace (stacks on), split inside and outside the
scheduler's omni.step spans.

usage: python3 python_hotspots.py <mapping trace.json.gz> [--thread TID] [--top 40]
Without --thread, the thread that carries the omni.step spans. python_function events nest by time on their thread;
a function's exclusive time is its duration minus its direct children's (Python frames, aten ops and CUDA calls
alike). Under a stack trace every frame pays the profiler's overhead, so the times are read as shares, not as served
milliseconds.
"""

import argparse
import gzip
import json
from collections import Counter

STEP_PREFIX = "omni.step"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--thread", type=int, default=None)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt", encoding="utf-8") as handle:
        events = json.load(handle)["traceEvents"]
    timed = [e for e in events if e.get("ph") == "X" and "dur" in e]
    steps = [e for e in timed if e.get("cat") == "user_annotation" and str(e.get("name")).startswith(STEP_PREFIX)]
    thread = args.thread if args.thread is not None else Counter(e["tid"] for e in steps).most_common(1)[0][0]
    spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in steps if e["tid"] == thread)
    on_thread = sorted(
        (e for e in timed if e["tid"] == thread and e.get("cat") in ("python_function", "cpu_op", "cuda_runtime", "cuda_driver")),
        key=lambda e: (e["ts"], -e["dur"]),
    )
    exclusive: Counter[tuple[str, bool]] = Counter()
    calls: Counter[str] = Counter()
    stack: list[list] = []

    def in_step(ts: float) -> bool:
        low, high = 0, len(spans)
        while low < high:
            mid = (low + high) // 2
            if spans[mid][1] < ts:
                low = mid + 1
            else:
                high = mid
        return low < len(spans) and spans[low][0] <= ts <= spans[low][1]

    def close(frame: list) -> None:
        event, child_time = frame
        if event.get("cat") == "python_function":
            name = str(event["name"])
            exclusive[(name, in_step(event["ts"]))] += event["dur"] - child_time
            calls[name] += 1
        else:
            pass

    for event in on_thread:
        while stack and stack[-1][0]["ts"] + stack[-1][0]["dur"] <= event["ts"]:
            close(stack.pop())
        if stack:
            stack[-1][1] += event["dur"]
        else:
            pass
        stack.append([event, 0.0])
    while stack:
        close(stack.pop())
    total_in = sum(v for (_, inside), v in exclusive.items() if inside)
    total_out = sum(v for (_, inside), v in exclusive.items() if not inside)
    print(f"thread {thread}: {len(spans)} steps; exclusive Python {total_in / 1e3:.1f} ms in steps, {total_out / 1e3:.1f} ms between")
    for label, inside, total in (("in steps", True, total_in), ("between steps", False, total_out)):
        rows = sorted(((name, v) for (name, flag), v in exclusive.items() if flag is inside), key=lambda r: -r[1])
        print(f"\n## {label}\n| function | exclusive ms | share | calls |\n| --- | --- | --- | --- |")
        for name, value in rows[: args.top]:
            print(f"| {name[:110]} | {value / 1e3:.2f} | {value / total:.1%} | {calls[name]} |")


if __name__ == "__main__":
    main()
