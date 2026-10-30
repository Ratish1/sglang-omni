"""A scheduler stage's occupancy per boot from request events: running and queued rows, completions, model time.

Per request it takes the stage's scheduler_queue_enter, model_path_start and model_path_end. Over
the middle 80 % of the boot's span it prints the time-averaged running rows (model start to end),
queued rows (queue enter to model start) and completions a second, then the median and p90 of a
request's model time, so two boots of one stage compare on load and on per-request speed.

usage: python stage_occupancy.py STAGE EVENT_DIR [EVENT_DIR ...]
"""

from __future__ import annotations

import glob
import json
import sys


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def time_average(
    intervals: list[tuple[float, float]], low: float, high: float
) -> float:
    edges = []
    for start, end in intervals:
        if end > low and start < high:
            edges.append((max(start, low), 1))
            edges.append((min(end, high), -1))
        else:
            pass
    area = 0.0
    level = 0
    last = low
    for timestamp, step in sorted(edges):
        area += level * (timestamp - last)
        level += step
        last = timestamp
    return area / (high - low)


def main() -> None:
    stage = sys.argv[1]
    for event_dir in sys.argv[2:]:
        starts: dict[str, float] = {}
        ends: dict[str, float] = {}
        enters: dict[str, float] = {}
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
                    request_id = event["request_id"]
                    seconds = event["timestamp_ns"] / 1e9
                    name = event["event_name"]
                    if name == "model_path_start":
                        starts.setdefault(request_id, seconds)
                    elif name == "model_path_end":
                        ends.setdefault(request_id, seconds)
                    elif name == "scheduler_queue_enter":
                        enters.setdefault(request_id, seconds)
                    else:
                        pass
        request_ids = [rid for rid in starts if rid in ends and rid in enters]
        if not request_ids:
            print(f"{event_dir} {stage}: no complete requests")
            continue
        else:
            pass
        first = min(starts[rid] for rid in request_ids)
        last = max(ends[rid] for rid in request_ids)
        low = first + 0.1 * (last - first)
        high = last - 0.1 * (last - first)
        running = time_average(
            [(starts[rid], ends[rid]) for rid in request_ids], low, high
        )
        queued = time_average(
            [(enters[rid], starts[rid]) for rid in request_ids], low, high
        )
        completions = sum(1 for rid in request_ids if low <= ends[rid] < high) / (
            high - low
        )
        model_ms = [(ends[rid] - starts[rid]) * 1000 for rid in request_ids]
        print(
            f"{event_dir} {stage}: running {running:6.2f} queued {queued:6.2f} done/s {completions:6.2f}"
            f" model median {percentile(model_ms, 0.5):8.1f} ms p90 {percentile(model_ms, 0.9):8.1f} ms"
            f" span {last - first:6.1f} s n {len(request_ids)}"
        )


if __name__ == "__main__":
    main()
