"""Known-answer checks of the 10-20 profile's readers against the traces they read.

usage: python3 verify_readers.py /data/prof_20261020
1. Gates per formal and mapping trace: graph capture, compile, lazy loads in the window (the patterns trace_streams.py
   uses), compiled-region executions counted apart, graph launches (a mapping trace must have none).
2. Graph identity in step_anatomy.py: for decode steps of W2 (bs 1) and W4 (bs about 63), the top kernels of the
   step's first and second graph launch, to show which is the talker and which the predictor.
3. step_anatomy.py's host time inside traced ops uses a 64-event lookback; recompute it exactly for 60 steps of W4
   and report the largest difference.
"""

import bisect
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

GATES = {
    "graph capture": ("cudaStreamBeginCapture", "cudaGraphInstantiate", "cudaStreamEndCapture"),
    "compile": ("dynamo_timed", "compile_fx", "OutputGraph.call_user_compiler"),
    "lazy load": ("Lazy Function Loading", "cuModuleLoad", "cuLibraryLoad"),
}
STEP = re.compile(r"omni\.step (\w+) (\w+) fwd=(\d+) bs=(\d+)")


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["traceEvents"]


def union_us(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for start, stop in sorted(intervals):
        if start > end:
            total, end = total + stop - start, stop
        elif stop > end:
            total, end = total + stop - end, stop
        else:
            pass
    return total


def gates(events: list[dict]) -> str:
    counts: Counter[str] = Counter()
    for e in events:
        name = str(e.get("name", ""))
        for gate, patterns in GATES.items():
            if any(p in name for p in patterns):
                counts[gate] += 1
            else:
                pass
        if name.startswith("Torch-Compiled Region"):
            counts["compiled region runs"] += 1
        elif e.get("cat") == "cuda_runtime" and name.startswith("cudaGraphLaunch"):
            counts["graph launches"] += 1
        else:
            pass
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"


def graph_identity(events: list[dict], label: str) -> None:
    steps = sorted((e for e in events if e.get("cat") == "user_annotation" and STEP.match(str(e.get("name")))), key=lambda e: e["ts"])
    decode = [s for s in steps if STEP.match(s["name"]).group(1) == "decode"]
    thread = steps[0]["tid"]
    graph_calls = sorted(
        (e for e in events if e.get("cat") == "cuda_runtime" and e["tid"] == thread and str(e["name"]).startswith("cudaGraphLaunch")),
        key=lambda e: e["ts"],
    )
    starts = [e["ts"] for e in graph_calls]
    kernels_by_corr: dict[int, list[str]] = {}
    for e in events:
        if e.get("cat") == "kernel":
            kernels_by_corr.setdefault(e.get("args", {}).get("correlation"), []).append(e["name"])
        else:
            pass
    for step in decode[len(decode) // 2: len(decode) // 2 + 2]:
        calls = graph_calls[bisect.bisect_left(starts, step["ts"]): bisect.bisect_right(starts, step["ts"] + step["dur"])]
        for index, call in enumerate(calls, 1):
            names = Counter(n[:48] for n in kernels_by_corr.get(call.get("args", {}).get("correlation"), []))
            print(f"  {label} {step['name']}: graph{index} {sum(names.values())} kernels, top {names.most_common(4)}")


def lookback_error(events: list[dict]) -> None:
    steps = sorted((e for e in events if e.get("cat") == "user_annotation" and STEP.match(str(e.get("name")))), key=lambda e: e["ts"])
    thread = steps[0]["tid"]
    traced = sorted((e["ts"], e["ts"] + e["dur"]) for e in events if e.get("ph") == "X" and e["tid"] == thread and e.get("cat") in ("cpu_op", "cuda_runtime", "cuda_driver"))
    traced_starts = [s for s, _ in traced]
    worst = 0.0
    for step in steps[:60]:
        start, end = step["ts"], step["ts"] + step["dur"]

        def clipped(low: int) -> float:
            spans = [(max(a, start), min(b, end)) for a, b in traced[low:bisect.bisect_right(traced_starts, end)] if b > start]
            return union_us([s for s in spans if s[1] > s[0]])

        approximate = clipped(max(0, bisect.bisect_left(traced_starts, start) - 64))
        exact = clipped(0)
        worst = max(worst, abs(exact - approximate))
    print(f"  W4 lookback: largest difference over 60 steps {worst / 1e3:.4f} ms")


def main() -> None:
    root = Path(sys.argv[1])
    for mode in ("formal", "mapping"):
        for cell in ("W1", "W2", "W3", "W4"):
            trace = next((root / mode / cell).glob("trace*.trace.json.gz"))
            print(f"{mode} {cell}: {gates(load(trace))}")
    for cell in ("W2", "W4"):
        events = load(next((root / "formal" / cell).glob("trace*.trace.json.gz")))
        graph_identity(events, cell)
        if cell == "W4":
            lookback_error(events)
        else:
            pass


if __name__ == "__main__":
    main()
