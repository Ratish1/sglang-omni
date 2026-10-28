"""Two arms' request paths side by side from the request event recorder.

Reads every events_*.jsonl under each event dir (run_bench_boot.sh with EVENTS=1), keeps each
request's first timestamp per (stage, event), takes a slice of the requests by admission
order (START:STOP as in Python, e.g. :20 or -20:, one arm of a boot that served several), and
prints for every (stage, event) its median and p90 offset from admission in ms in each dir and
the median's difference, ordered by the first dir's median, so the first event where the arms
part names the stage that owns the difference.

usage: python events_path_offsets.py EVENT_DIR_A EVENT_DIR_B START:STOP
"""

from __future__ import annotations

import collections
import glob
import json
import os
import statistics
import sys

ADMISSION = ("coordinator", "request_admission")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def offsets(event_dir: str, selected: slice) -> dict[tuple[str, str], list[float]]:
    first: dict[str, dict[tuple[str, str], int]] = collections.defaultdict(dict)
    for path in glob.glob(
        os.path.join(event_dir, "**", "events_*.jsonl"), recursive=True
    ):
        with open(path) as handle:
            for line in handle:
                event = json.loads(line)
                key = (event["stage"] or "-", event["event_name"])
                request = first[event["request_id"]]
                if key not in request or event["timestamp_ns"] < request[key]:
                    request[key] = event["timestamp_ns"]
    admitted = sorted(
        (request[ADMISSION], request_id)
        for request_id, request in first.items()
        if ADMISSION in request
    )[selected]
    result: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for admission_ns, request_id in admitted:
        for key, timestamp_ns in first[request_id].items():
            result[key].append((timestamp_ns - admission_ns) / 1e6)
    return result


def main() -> None:
    start, stop = (int(bound) if bound else None for bound in sys.argv[3].split(":"))
    selected = slice(start, stop)
    arm_a = offsets(sys.argv[1], selected)
    arm_b = offsets(sys.argv[2], selected)
    print(f"A {sys.argv[1]}\nB {sys.argv[2]}\nrequests {sys.argv[3]} by admission")
    print(
        f"{'stage:event':58s} {'n':>3s} {'A med':>9s} {'B med':>9s} {'B-A':>8s} "
        f"{'A p90':>9s} {'B p90':>9s}"
    )
    for key in sorted(arm_a, key=lambda item: statistics.median(arm_a[item])):
        a_values = arm_a[key]
        b_values = arm_b.get(key, [])
        a_median = statistics.median(a_values)
        if b_values:
            b_median = statistics.median(b_values)
            print(
                f"{key[0] + ':' + key[1]:58s} {len(a_values):3d} {a_median:9.1f} "
                f"{b_median:9.1f} {b_median - a_median:+8.1f} "
                f"{percentile(a_values, 0.9):9.1f} {percentile(b_values, 0.9):9.1f}"
            )
        else:
            print(
                f"{key[0] + ':' + key[1]:58s} {len(a_values):3d} {a_median:9.1f} {'-':>9s}"
            )


if __name__ == "__main__":
    main()
