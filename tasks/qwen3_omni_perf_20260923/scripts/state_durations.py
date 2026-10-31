"""Per boot, the median and p90 of a SeedTTS request's state durations from the request event recorder (ms).

thinker reply: thinker prefill start to the thinker's model path end; talker wait: talker queue enter to its model path
start; talker generation: talker prefill start to its model path end; first audio: admission to code2wav's first audio;
done: admission to code2wav's stage complete. Each event's first timestamp per request.

usage: python3 state_durations.py BOOT_DIR [BOOT_DIR ...]   (each with an events/ directory)
"""

import collections
import glob
import json
import os
import sys
from pathlib import Path

SPANS = (
    (
        "thinker reply",
        ("thinker", "scheduler_prefill_start"),
        ("thinker", "model_path_end"),
    ),
    (
        "talker wait",
        ("talker_ar", "scheduler_queue_enter"),
        ("talker_ar", "model_path_start"),
    ),
    (
        "talker generation",
        ("talker_ar", "scheduler_prefill_start"),
        ("talker_ar", "model_path_end"),
    ),
    (
        "first audio",
        ("coordinator", "request_admission"),
        ("code2wav", "code2wav_first_audio"),
    ),
    ("done", ("coordinator", "request_admission"), ("code2wav", "stage_complete")),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


for boot_dir in sys.argv[1:]:
    first: dict[str, dict[tuple[str, str], float]] = collections.defaultdict(dict)
    for path in glob.glob(
        os.path.join(boot_dir, "events", "**", "events_*.jsonl"), recursive=True
    ):
        for line in Path(path).read_text().splitlines():
            event = json.loads(line)
            key = (event["stage"] or "-", event["event_name"])
            seen = first[event["request_id"]]
            if key not in seen or event["timestamp_ns"] < seen[key]:
                seen[key] = event["timestamp_ns"]
            else:
                pass
    cells = []
    for name, start, end in SPANS:
        durations = [
            (events[end] - events[start]) / 1e6
            for events in first.values()
            if start in events and end in events
        ]
        cells.append(
            f"{name} {percentile(durations, 0.5):7.1f} / {percentile(durations, 0.9):7.1f}"
        )
    print(
        f"{boot_dir.rstrip('/').split('/')[-2]}/{boot_dir.rstrip('/').split('/')[-1]:14s} n {len(first)} | "
        + " | ".join(cells)
    )
