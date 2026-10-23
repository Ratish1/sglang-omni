"""Request states of a c64 events cell: first time of each event after admission, over the opening burst and the rest.

usage: python3 burst_states.py <run dir> [<run dir> ...]
"""

import glob
import json
import statistics
import sys
from collections import defaultdict

BURST = 64


def load(run):
    first = defaultdict(dict)
    for path in glob.glob(f"{run}/c64_events/events/*.jsonl"):
        with open(path) as source:
            for line in source:
                record = json.loads(line)
                key = (record["stage"], record["event_name"])
                seen = first[record["request_id"]]
                if key not in seen or record["timestamp_ns"] < seen[key]:
                    seen[key] = record["timestamp_ns"]
                else:
                    pass
    return first


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def split(first, burst):
    admission = ("coordinator", "request_admission")
    admitted = sorted(
        (events[admission], request)
        for request, events in first.items()
        if admission in events
    )
    chosen = [
        request for _, request in (admitted[:BURST] if burst else admitted[BURST:])
    ]
    offsets = defaultdict(list)
    for request in chosen:
        start = first[request][admission]
        for key, ns in first[request].items():
            offsets[key].append((ns - start) / 1e6)
    return offsets


runs = sys.argv[1:]
data = {run: load(run) for run in runs}
for burst in (True, False):
    print(
        f"opening {BURST} requests" if burst else "rest of the pass",
        "- ms after admission, median / p90",
    )
    splits = {run: split(data[run], burst) for run in runs}
    keys = sorted(
        splits[runs[0]], key=lambda key: statistics.median(splits[runs[0]][key])
    )
    print(
        "  "
        + "event".ljust(52)
        + "".join(run.rstrip("/").split("/")[-1].ljust(18) for run in runs)
    )
    for key in keys:
        cells = []
        for run in runs:
            values = splits[run].get(key, [])
            if values:
                cells.append(
                    f"{statistics.median(values):7.1f} / {quantile(values, 0.9):7.1f}".ljust(
                        18
                    )
                )
            else:
                cells.append("-".ljust(18))
        print("  " + f"{key[0]}:{key[1]}"[:52].ljust(52) + "".join(cells))
