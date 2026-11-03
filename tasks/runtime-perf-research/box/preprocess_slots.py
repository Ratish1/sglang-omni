"""Where a request's preprocessing time goes and what the stage's worker bound costs, read from one events cell per boot.

Per request: queue (admission to preprocess_start), template and setup, media wait, processor, the rest of the work. Over
the pass: the mean number of requests in each phase, the queue time spent with a worker parked on media, which a media
pool outside the workers would give back, and the same arrivals replayed with media outside and two processor slots.

usage: python3 preprocess_slots.py run/boot/c16_events[,run/boot/c16_events] [...]
"""

import glob
import heapq
import json
import statistics
import sys
from collections import defaultdict

PHASES = (
    ("preprocess_start", "queue", "admission"),
    ("preprocess_media_start", "setup", "preprocess_start"),
    ("preprocess_media_end", "media", "preprocess_media_start"),
    ("preprocess_processor_start", "between", "preprocess_media_end"),
    ("preprocess_processor_end", "processor", "preprocess_processor_start"),
    ("preprocess_end", "after", "preprocess_processor_end"),
)
EVENTS = {
    "request_admission": "admission",
    "preprocess_start": "preprocess_start",
    "preprocess_media_start": "preprocess_media_start",
    "preprocess_media_end": "preprocess_media_end",
    "preprocess_processor_start": "preprocess_processor_start",
    "preprocess_processor_end": "preprocess_processor_end",
    "preprocess_end": "preprocess_end",
}
PROCESSOR_SLOTS = 2


def read_requests(cell: str) -> dict[str, dict[str, float]]:
    requests: dict[str, dict[str, float]] = defaultdict(dict)
    for path in glob.glob(f"{cell}/events/*.jsonl"):
        with open(path) as events:
            for line in events:
                record = json.loads(line)
                key = EVENTS.get(record.get("event_name"))
                if key is None:
                    continue
                stamps = requests[record["request_id"]]
                stamps.setdefault(key, record["timestamp_ns"] / 1e6)
    return {
        request_id: stamps
        for request_id, stamps in requests.items()
        if len(stamps) == len(EVENTS)
    }


def overlap_ms(start: float, end: float, spans: list[tuple[float, float]]) -> float:
    """Time in [start, end] covered by at least one span."""
    covered = 0.0
    cursor = start
    for span_start, span_end in sorted(spans):
        if span_end <= cursor:
            continue
        if span_start >= end:
            break
        covered += min(span_end, end) - max(span_start, cursor)
        cursor = min(span_end, end)
    return covered


def replay_media_outside(requests: dict[str, dict[str, float]]) -> list[float]:
    """Each request's wait for a processor slot with its media started at admission, slot time its processor to its end."""
    ready = sorted(
        (
            stamps["admission"]
            + stamps["preprocess_media_start"]
            - stamps["preprocess_start"]
            + stamps["preprocess_media_end"]
            - stamps["preprocess_media_start"],
            stamps["preprocess_end"] - stamps["preprocess_media_end"],
        )
        for stamps in requests.values()
    )
    slots = [0.0] * PROCESSOR_SLOTS
    waits = []
    for ready_at, slot_time in ready:
        free_at = heapq.heappop(slots)
        start = max(ready_at, free_at)
        waits.append(start - ready_at)
        heapq.heappush(slots, start + slot_time)
    return waits


def read_arm(cells: list[str]) -> None:
    durations: dict[str, list[float]] = defaultdict(list)
    queue_beside_media: list[float] = []
    queue_all: list[float] = []
    replay_waits: list[float] = []
    busy: dict[str, float] = defaultdict(float)
    window = 0.0
    for cell in cells:
        requests = read_requests(cell)
        media_spans = [
            (stamps["preprocess_media_start"], stamps["preprocess_media_end"])
            for stamps in requests.values()
        ]
        first = min(stamps["admission"] for stamps in requests.values())
        last = max(stamps["preprocess_end"] for stamps in requests.values())
        window += last - first
        for stamps in requests.values():
            for end_key, name, start_key in PHASES:
                duration = stamps[end_key] - stamps[start_key]
                durations[name].append(duration)
                busy[name] += duration
            queue_all.append(stamps["preprocess_start"] - stamps["admission"])
            queue_beside_media.append(
                overlap_ms(stamps["admission"], stamps["preprocess_start"], media_spans)
            )
        replay_waits.extend(replay_media_outside(requests))
    print(f"{','.join(cells)}: {len(queue_all)} requests, window {window / 1e3:.1f} s")
    print(f"{'phase':12s}{'median':>10s}{'mean':>10s}{'p95':>10s}{'in phase':>10s}")
    for _, name, _ in PHASES:
        values = sorted(durations[name])
        print(
            f"{name:12s}{statistics.median(values):10.1f}{statistics.fmean(values):10.1f}"
            f"{values[int(len(values) * 0.95)]:10.1f}{busy[name] / window:10.2f}"
        )
    print(
        f"queue mean {statistics.fmean(queue_all):.1f} ms, of it with a worker on media "
        f"{statistics.fmean(queue_beside_media):.1f} ms"
    )
    print(
        f"replay, media outside, {PROCESSOR_SLOTS} processor slots: slot wait mean "
        f"{statistics.fmean(replay_waits):.1f} ms, p95 "
        f"{sorted(replay_waits)[int(len(replay_waits) * 0.95)]:.1f} ms"
    )


def main() -> None:
    for argument in sys.argv[1:]:
        read_arm(argument.split(","))


if __name__ == "__main__":
    main()
