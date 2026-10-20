"""Host time of the scheduler thread by call path, from a formal trace with stacks (graphs on, the served config).

usage: python3 host_frames.py <trace.json.gz> [--depth 8] [--min-ms 0.05]
       python3 host_frames.py --self-test
The profiler's python_function events carry their own call tree ("Python id", "Python parent id"). On the thread that
carries the omni.step spans, every frame below the event loop frame is aggregated by its call path (names from the
loop's direct child down), and the tree is printed with inclusive milliseconds per decode step and calls per step.
A frame's inclusive time contains the native calls it makes and any wait for the interpreter lock inside it.
--self-test profiles a known program (f spins 2 ms and calls g; g spins 1 ms and runs a torch op) and checks that
the reader recovers f at 3 ms and g at 1 ms per iteration.
"""

import argparse
import gzip
import json
import os
import re
import sys
import tempfile
import time
from collections import defaultdict

STEP = re.compile(r"omni\.step (\w+) (\w+) fwd=(\d+) bs=(\d+)")
LOOP_FRAME = re.compile(r"omni_scheduler\.py\(\d+\): event_loop_\w+")


def load(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["traceEvents"]


def aggregate(frames: list[dict], root_ids: list[int], depth: int) -> dict[tuple[str, ...], list[float]]:
    """Inclusive time and calls per call path under the given roots (paths start at the roots' children)."""
    children: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        children[frame["args"]["Python parent id"]].append(frame)
    totals: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0, 0])
    stack = [(child, (str(child["name"]),)) for root in root_ids for child in children.get(root, [])]
    while stack:
        frame, path = stack.pop()
        totals[path][0] += frame["dur"]
        totals[path][1] += 1
        if len(path) < depth:
            stack.extend((child, (*path, str(child["name"]))) for child in children.get(frame["args"]["Python id"], []))
        else:
            pass
    return totals


def print_tree(totals: dict[tuple[str, ...], list[float]], per: int, min_ms: float) -> None:
    def walk(prefix: tuple[str, ...], indent: int) -> None:
        kids = sorted((p for p in totals if len(p) == len(prefix) + 1 and p[: len(prefix)] == prefix), key=lambda p: -totals[p][0])
        for path in kids:
            total, calls = totals[path]
            if total / per / 1e3 < min_ms:
                continue
            else:
                pass
            print(f"{'  ' * indent}{total / per / 1e3:8.3f} ms  {calls / per:6.1f}x  {path[-1][:120]}")
            walk(path, indent + 1)

    walk((), 0)


def self_test() -> None:
    import torch
    from torch.profiler import ProfilerActivity, profile

    def spin(seconds: float) -> None:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            pass

    def g() -> None:
        spin(0.001)
        torch.ones(1000).sum()

    def f() -> None:
        spin(0.002)
        g()

    def loop() -> None:
        for _ in range(20):
            f()

    with profile(activities=[ProfilerActivity.CPU], with_stack=True) as prof:
        loop()
    path = os.path.join(tempfile.mkdtemp(), "self_test.json")
    prof.export_chrome_trace(path)
    events = load(path)
    frames = [e for e in events if e.get("cat") == "python_function" and e.get("ph") == "X"]
    roots = [e["args"]["Python id"] for e in frames if re.search(r"\): loop$", str(e["name"]))]
    totals = aggregate(frames, roots, depth=4)
    f_path = next(p for p in totals if len(p) == 1 and p[0].endswith(": f"))
    g_path = next(p for p in totals if len(p) == 2 and p[0] == f_path[0] and p[1].endswith(": g"))
    f_ms = totals[f_path][0] / totals[f_path][1] / 1e3
    g_ms = totals[g_path][0] / totals[g_path][1] / 1e3
    print(f"self-test: f {f_ms:.3f} ms x{totals[f_path][1]}, g {g_ms:.3f} ms x{totals[g_path][1]} (expect 3 and 1, 20 calls)")
    assert totals[f_path][1] == 20 and totals[g_path][1] == 20
    assert abs(f_ms - 3.0) < 0.4 and abs(g_ms - 1.0) < 0.3
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="?")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--min-ms", type=float, default=0.05)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    else:
        pass
    events = load(args.trace)
    steps = [e for e in events if e.get("cat") == "user_annotation" and STEP.match(str(e.get("name")))]
    thread = steps[0]["tid"]
    decode_steps = sum(1 for s in steps if STEP.match(s["name"]).group(1) == "decode")
    frames = [e for e in events if e.get("cat") == "python_function" and e.get("ph") == "X" and e["tid"] == thread]
    roots = [e["args"]["Python id"] for e in frames if LOOP_FRAME.search(str(e["name"]))]
    if not roots:
        sys.exit("no event loop frame on the scheduler thread")
    else:
        pass
    totals = aggregate(frames, roots, args.depth)
    kinds = defaultdict(int)
    for s in steps:
        kinds[(STEP.match(s["name"]).group(1), int(STEP.match(s["name"]).group(4)))] += 1
    print(f"scheduler thread {thread}: {len(steps)} steps ({dict(sorted(kinds.items()))}); per decode step over {decode_steps}")
    print_tree(totals, max(decode_steps, 1), args.min_ms)


if __name__ == "__main__":
    main()
