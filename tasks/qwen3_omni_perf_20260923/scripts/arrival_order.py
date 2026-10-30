"""The order in which one stage's inputs arrive per request, in each boot's request events.

For every request it keeps the stage's first timestamp per (event, sending stage) key, from the
stage's first event through its first scheduler event, and sorts the keys by time. It prints, per
event dir, how many requests show each arrival order, and the median and p90 of every key's offset
from the stage's first event, so two boots' orders and gaps compare side by side.

usage: python arrival_order.py STAGE EVENT_DIR [EVENT_DIR ...]
"""

from __future__ import annotations

import collections
import glob
import json
import sys


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def stage_keys(event_dir: str, stage: str) -> dict[str, dict[str, int]]:
    """Per request, the first timestamp of each (event, sending stage) key of the stage."""
    first: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for path in glob.glob(f"{event_dir}/*.jsonl"):
        with open(path) as events_file:
            for line in events_file:
                if f'"{stage}"' not in line:
                    continue
                else:
                    pass
                event = json.loads(line)
                if event["stage"] != stage:
                    continue
                else:
                    pass
                metadata = event.get("metadata") or {}
                key = f"{event['event_name']}<{metadata.get('from_stage', '')}"
                request_keys = first[event["request_id"]]
                if key not in request_keys:
                    request_keys[key] = event["timestamp_ns"]
                else:
                    pass
    return first


def inputs_through_first_scheduler_event(keys: dict[str, int]) -> list[tuple[str, int]]:
    ordered = sorted(keys.items(), key=lambda item: item[1])
    kept = []
    for key, timestamp in ordered:
        kept.append((key, timestamp))
        if key.startswith("scheduler_"):
            break
        else:
            pass
    return kept


def main() -> None:
    stage = sys.argv[1]
    for event_dir in sys.argv[2:]:
        per_request = stage_keys(event_dir, stage)
        orders = collections.Counter()
        offsets: dict[str, list[float]] = collections.defaultdict(list)
        for keys in per_request.values():
            kept = inputs_through_first_scheduler_event(keys)
            if not kept:
                continue
            else:
                pass
            start = kept[0][1]
            orders[" > ".join(key for key, _ in kept)] += 1
            for key, timestamp in kept:
                offsets[key].append((timestamp - start) / 1e6)
        print(f"== {event_dir} {stage}: {len(per_request)} requests")
        for order, count in orders.most_common():
            print(f"  {count:5d}  {order}")
        for key, values in sorted(
            offsets.items(), key=lambda item: percentile(item[1], 0.5)
        ):
            print(
                f"  {key:60s} n {len(values):5d} median {percentile(values, 0.5):8.3f} ms"
                f" p90 {percentile(values, 0.9):8.3f} ms"
            )


if __name__ == "__main__":
    main()
