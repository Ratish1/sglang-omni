"""Python self time per thread role over a with-stack trace window.

Threads are grouped by role (the sglang_omni file that appears most on the thread). Per
role: threads, python time on the threads inside the omni.step window, and the functions
ranked by self time (duration minus direct children). With-stack tracing inflates host
time, so this names the code that holds the interpreter; it never times it.

usage: python thread_stacks.py STACK_TRACE.json.gz [--top 18]
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict


def short(name: str) -> str:
    for marker in ("sglang_omni/", "site-packages/", "lib/python3.12/"):
        if marker in name:
            return name.split(marker, 1)[1]
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--top", type=int, default=18)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt") as handle:
        events = [e for e in json.load(handle)["traceEvents"] if e.get("ph") == "X"]
    steps = sorted(
        float(e["ts"])
        for e in events
        if e.get("cat") == "user_annotation" and e["name"].startswith("omni.step ")
    )
    t0, t1 = steps[0], steps[-1]
    by_tid: dict = defaultdict(list)
    for event in events:
        if event.get("cat") == "python_function" and t0 <= float(event["ts"]) < t1:
            by_tid[event.get("tid")].append(event)

    roles: dict[str, list] = defaultdict(list)
    self_by_tid: dict = {}
    for tid, calls in by_tid.items():
        calls.sort(key=lambda e: (float(e["ts"]), -float(e["dur"])))
        self_us: Counter = Counter()
        files: Counter = Counter()
        stack: list[list] = []
        frames = []
        for event in calls:
            start = float(event["ts"])
            end = min(start + float(event["dur"]), t1)
            while stack and stack[-1][1] <= start:
                stack.pop()
            if stack:
                stack[-1][2] -= end - start
            frame = [start, end, end - start, short(event["name"])]
            stack.append(frame)
            frames.append(frame)
            if "sglang_omni/" in event["name"]:
                files[frame[3].split("(")[0]] += 1
        for frame in frames:
            self_us[frame[3]] += max(frame[2], 0.0)
        self_by_tid[tid] = self_us
        roles[files.most_common(1)[0][0] if files else "no sglang_omni frame"].append(
            tid
        )

    print(f"trace {args.trace}\nwindow {(t1 - t0) / 1e3:.1f} ms (with stack; inflated)")
    ranked = sorted(
        roles.items(),
        key=lambda item: -sum(sum(self_by_tid[t].values()) for t in item[1]),
    )
    for role, tids in ranked:
        merged: Counter = Counter()
        for tid in tids:
            merged.update(self_by_tid[tid])
        total = sum(merged.values())
        print(
            f"\nrole {role}: {len(tids)} threads, python self time {total / 1e3:.1f} ms = {100 * total / (t1 - t0):.1f}% of the window"
        )
        for name, value in merged.most_common(args.top):
            print(f"  {value / 1e3:>9.1f} ms{100 * value / total:>6.1f}%  {name[:140]}")


if __name__ == "__main__":
    main()
