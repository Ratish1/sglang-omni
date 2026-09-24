"""First-audio path of Qwen3-Omni speech requests from the request event recorder.

Reads every events_*.jsonl under the event dir (run_bench_boot.sh with EVENTS=1), keeps each
request's first timestamp per (stage, event), and prints each segment of the path from
admission to the first audio chunk leaving code2wav: count, median, p90, p99 in ms.
Wall clocks of processes on one host are comparable (time.time_ns in every process).

usage: python events_first_audio.py EVENT_DIR
"""

from __future__ import annotations

import collections
import glob
import json
import os
import statistics
import sys

POINTS = [
    ("admission", "coordinator", "request_admission"),
    ("thinker_done", "thinker", "stage_complete"),
    ("talker_build_start", "talker_ar", "scheduler_request_build_start"),
    ("talker_build_end", "talker_ar", "scheduler_request_build_end"),
    ("talker_prefill_start", "talker_ar", "scheduler_prefill_start"),
    ("talker_prefill_end", "talker_ar", "scheduler_prefill_end"),
    ("talker_first_frame_sent", "talker_ar", "stage_first_stream_chunk_sent"),
    ("code2wav_first_ingest", "code2wav", "code2wav_first_ingest"),
    ("code2wav_window_ready", "code2wav", "code2wav_first_window_ready"),
    ("code2wav_decode_start", "code2wav", "code2wav_decode_start"),
    ("code2wav_decode_end", "code2wav", "code2wav_decode_end"),
    ("code2wav_first_audio", "code2wav", "code2wav_first_audio"),
    ("audio_sent", "code2wav", "stage_first_stream_chunk_sent"),
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    event_dir = sys.argv[1]
    first: dict[str, dict[tuple[str, str], int]] = collections.defaultdict(dict)
    for path in glob.glob(
        os.path.join(event_dir, "**", "events_*.jsonl"), recursive=True
    ):
        with open(path) as handle:
            for line in handle:
                event = json.loads(line)
                key = (event["stage"], event["event_name"])
                request = first[event["request_id"]]
                if key not in request or event["timestamp_ns"] < request[key]:
                    request[key] = event["timestamp_ns"]
    names = [name for name, _, _ in POINTS]
    rows = []
    for request in first.values():
        times = {name: request.get((stage, event)) for name, stage, event in POINTS}
        if times["admission"] is not None and times["audio_sent"] is not None:
            rows.append(times)
    print(f"requests with a first audio chunk: {len(rows)} of {len(first)}")
    print(f"{'segment':50s} {'n':>5s} {'median_ms':>10s} {'p90_ms':>9s} {'p99_ms':>9s}")
    segments = list(zip(names, names[1:])) + [("admission", "audio_sent")]
    for begin, end in segments:
        values = [
            (row[end] - row[begin]) / 1e6
            for row in rows
            if row[begin] is not None and row[end] is not None
        ]
        if not values:
            print(f"{begin + ' -> ' + end:50s} {0:5d}")
            continue
        print(
            f"{begin + ' -> ' + end:50s} {len(values):5d} {statistics.median(values):10.1f} "
            f"{percentile(values, 0.9):9.1f} {percentile(values, 0.99):9.1f}"
        )


if __name__ == "__main__":
    main()
