"""Host time of the scheduler thread by call path, from a formal trace with stacks (graphs on, the served config).

usage: python3 host_frames.py <trace.json.gz> [--depth 8] [--min-ms 0.05]
       python3 host_frames.py --self-test
The profiler's python_function events carry their own call tree ("Python id", "Python parent id"). On the thread that
carries the omni.step spans, the event loop frame's children are cut into loop iterations at each
process_admin_requests call (the first call of an iteration of event_loop_normal), each iteration labeled by the
omni.step span inside it (decode or prefill, its batch size). For each label, every frame below the loop is aggregated
by its call path (names from the loop's direct child down) and printed as a tree with inclusive milliseconds per
iteration and calls per iteration; the share of the loop's time the iterations cover is printed as a check. A frame's
inclusive time contains the native calls it makes and any wait for the interpreter lock inside it.
--self-test profiles a known program (f spins 2 ms and calls g; g spins 1 ms and runs a torch op) and checks that
the reader recovers the two spins at 2 and 1 ms per call (wall-clock exact by construction), 20 calls each, and every
parent's inclusive time covering its children's.
"""

import argparse
import gzip
import json
import os
import re
import statistics
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


def children_of(frames: list[dict]) -> dict[int, list[dict]]:
    children: dict[int, list[dict]] = defaultdict(list)
    for frame in frames:
        children[frame["args"]["Python parent id"]].append(frame)
    return children


def aggregate(
    children: dict[int, list[dict]], top: list[dict], depth: int
) -> dict[tuple[str, ...], list[float]]:
    """Inclusive time and calls per call path under the given top frames (paths start at them)."""
    totals: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0, 0])
    stack = [(frame, (str(frame["name"]),)) for frame in top]
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
    children = children_of(frames)
    roots = [e["args"]["Python id"] for e in frames if re.search(r"\): loop$", str(e["name"]))]
    totals = aggregate(children, [child for root in roots for child in children.get(root, [])], depth=4)

    def mean_ms(path: tuple[str, ...]) -> float:
        return totals[path][0] / totals[path][1] / 1e3

    f_path = next(p for p in totals if len(p) == 1 and p[0].endswith(": f"))
    g_path = next(p for p in totals if len(p) == 2 and p[0] == f_path[0] and p[1].endswith(": g"))
    f_spin = next(p for p in totals if len(p) == 2 and p[0] == f_path[0] and p[1].endswith(": spin"))
    g_spin = next(p for p in totals if len(p) == 3 and p[:2] == g_path and p[2].endswith(": spin"))
    print(
        f"self-test: f {mean_ms(f_path):.3f} (spin {mean_ms(f_spin):.3f}), g {mean_ms(g_path):.3f} (spin "
        f"{mean_ms(g_spin):.3f}) ms per call, calls {totals[f_path][1]}, {totals[g_path][1]}; spins expected 2 and 1"
    )
    assert totals[f_path][1] == 20 and totals[g_path][1] == 20
    assert abs(mean_ms(f_spin) - 2.0) < 0.15 and abs(mean_ms(g_spin) - 1.0) < 0.15
    assert mean_ms(f_path) >= mean_ms(f_spin) + mean_ms(g_path) and mean_ms(g_path) >= mean_ms(g_spin)
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
    steps = sorted(
        (e for e in events if e.get("cat") == "user_annotation" and STEP.match(str(e.get("name")))), key=lambda e: e["ts"]
    )
    thread = steps[0]["tid"]
    frames = [e for e in events if e.get("cat") == "python_function" and e.get("ph") == "X" and e["tid"] == thread]
    children = children_of(frames)
    loops = [e for e in frames if LOOP_FRAME.search(str(e["name"]))]
    if len(loops) != 1:
        sys.exit(f"expected one event loop frame on the scheduler thread, found {len(loops)}")
    else:
        pass
    loop = loops[0]
    loop_children = sorted(children.get(loop["args"]["Python id"], []), key=lambda e: e["ts"])
    iterations: list[list[dict]] = []
    for child in loop_children:
        if str(child["name"]).endswith(": process_admin_requests") or not iterations:
            iterations.append([child])
        else:
            iterations[-1].append(child)
    step_starts = [s["ts"] for s in steps]
    groups: dict[str, list[list[dict]]] = defaultdict(list)
    unlabeled = 0
    for iteration in iterations:
        start, end = iteration[0]["ts"], max(f["ts"] + f["dur"] for f in iteration)
        inside = [steps[i] for i in range(len(steps)) if start <= step_starts[i] <= end]
        if len(inside) != 1:
            unlabeled += 1
            continue
        else:
            pass
        kind, _, _, batch = STEP.match(inside[0]["name"]).groups()
        groups[f"{kind} bs {int(batch):02d}"].append(iteration)
    loop_children_ms = sum(f["dur"] for f in loop_children) / 1e3
    labeled_ms = sum(f["dur"] for its in groups.values() for it in its for f in it) / 1e3
    window = (steps[-1]["ts"] + steps[-1]["dur"] - steps[0]["ts"]) / 1e3
    print(
        f"scheduler thread {thread}: {len(steps)} steps, {len(iterations)} loop iterations, {unlabeled} without exactly "
        f"one step (window edges); loop children {loop_children_ms:.1f} ms of which labeled {labeled_ms:.1f} ms; "
        f"steps span {window:.1f} ms"
    )
    for label, its in sorted(groups.items()):
        wall = statistics.median(max(f["ts"] + f["dur"] for f in it) - it[0]["ts"] for it in its) / 1e3
        print(f"\n## {label}: {len(its)} iterations, median iteration wall {wall:.3f} ms")
        print_tree(aggregate(children, [f for it in its for f in it], args.depth), len(its), args.min_ms)


if __name__ == "__main__":
    main()
