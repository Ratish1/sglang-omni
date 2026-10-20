"""Per graph replay of kit trace pairs: the device span a caller waits for, and the kernels that differ.

Run: python3 kit_replay_span_read.py <run_a> <run_b>   (code2wav_kit_pair.py output directories)
Every device event of a replay carries its cudaGraphLaunch's correlation id. Per replay of each run's formal trace:
the span from its first device event's start to its last one's end, the summed device time inside it and its event
count; the median replay of each run is listed kernel by kernel (start offset, duration), and the names whose time
per replay differs between the two runs are printed, largest first.
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEVICE_CATEGORIES = ("kernel", "gpu_memcpy", "gpu_memset")


def replays(run: Path) -> list[list[dict]]:
    trace = next((run / "formal").rglob("*.trace.json.gz"))
    events = json.loads(gzip.open(trace).read())["traceEvents"]
    launches = {
        event["args"]["correlation"]
        for event in events
        if event.get("cat") == "cuda_runtime" and event.get("name") == "cudaGraphLaunch"
    }
    by_launch: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        if event.get("cat") in DEVICE_CATEGORIES and event.get("args", {}).get("correlation") in launches:
            by_launch[event["args"]["correlation"]].append(event)
    return [sorted(group, key=lambda event: event["ts"]) for _, group in sorted(by_launch.items())]


def span(replay: list[dict]) -> float:
    return max(event["ts"] + event["dur"] for event in replay) - replay[0]["ts"]


def short(name: str) -> str:
    return name.split("<")[0].split("(")[0][:70]


def main() -> None:
    runs = [Path(arg) for arg in sys.argv[1:3]]
    medians = []
    for run in runs:
        groups = replays(run)
        spans = [span(group) for group in groups]
        busy = [sum(event["dur"] for event in group) for group in groups]
        print(
            f"{run.name}: {len(groups)} replays, span median {statistics.median(spans):.1f} us, device time median "
            f"{statistics.median(busy):.1f} us, events per replay {statistics.median(len(group) for group in groups)}"
        )
        median_replay = sorted(groups, key=span)[len(groups) // 2]
        medians.append(median_replay)
    for run, replay in zip(runs, medians):
        print(f"\n{run.name} median replay, offset us, duration us, name")
        start = replay[0]["ts"]
        for event in replay:
            print(f"  {event['ts'] - start:8.1f} {event['dur']:7.1f}  {short(event['name'])}")
    totals = []
    for replay in medians:
        counter: Counter[str] = Counter()
        for event in replay:
            counter[short(event["name"])] += event["dur"]
        totals.append(counter)
    print("\nper replay time by name, b minus a (us)")
    names = set(totals[0]) | set(totals[1])
    for name in sorted(names, key=lambda key: -abs(totals[1][key] - totals[0][key]))[:25]:
        print(f"  {totals[1][name] - totals[0][name]:+8.1f}  a {totals[0][name]:7.1f}  b {totals[1][name]:7.1f}  {name}")


if __name__ == "__main__":
    main()
