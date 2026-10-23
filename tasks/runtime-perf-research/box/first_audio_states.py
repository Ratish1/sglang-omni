"""First audio of a Qwen3-Omni speech events cell, split into consecutive request states, main against a branch.

Each request's first time of each event after its admission is read from <boot>/<cell>/events/*.jsonl; the states
are the differences between consecutive chain events, so they sum to the server-side first audio (admission to
code2wav's first audio chunk sent). Printed per arm over all requests (median and mean) and over each boot's slowest
5 % by that first audio (mean), boots of one arm pooled.

usage: python3 first_audio_states.py --cell c16_events --main run/a1,run/a2 --branch run/b1,run/b2
"""

import argparse
import glob
import json
import statistics
from collections import defaultdict

CHAIN = (
    ("coordinator", "request_admission", "admission"),
    ("preprocessing", "preprocess_start", "preprocessing queue"),
    ("preprocessing", "preprocess_end", "preprocessing work"),
    ("audio_encoder", "stage_input_received", "hop to the audio encoder"),
    ("audio_encoder", "encoder_start", "audio encoder queue"),
    ("audio_encoder", "encoder_end", "audio encoder"),
    ("thinker", "scheduler_request_build_start", "hop to the thinker, build wait"),
    ("thinker", "scheduler_prefill_start", "thinker build and queue"),
    ("thinker", "scheduler_prefill_end", "thinker prefill"),
    ("thinker", "scheduler_first_emit", "first token emit"),
    ("talker_ar", "scheduler_request_build_start", "thinker chunks until talker build"),
    ("talker_ar", "scheduler_prefill_start", "talker build and queue"),
    ("talker_ar", "stage_first_stream_chunk_sent", "talker prefill and first frame"),
    ("code2wav", "code2wav_first_window_ready", "frames until code2wav's first window"),
    ("code2wav", "stage_first_stream_chunk_sent", "code2wav first audio"),
)
TAIL = 0.05


def first_times(boot: str, cell: str) -> dict[str, dict[tuple[str, str], int]]:
    first: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for path in glob.glob(f"{boot}/{cell}/events/*.jsonl"):
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


def state_rows(boot: str, cell: str) -> list[list[float]]:
    """Per complete request, the state durations in ms in CHAIN order (first entry is the total)."""
    rows = []
    for events in first_times(boot, cell).values():
        stamps = [events.get((stage, name)) for stage, name, _ in CHAIN]
        if any(stamp is None for stamp in stamps):
            continue
        else:
            pass
        states = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
        rows.append([(stamps[-1] - stamps[0]) / 1e6, *states])
    return rows


def arm_rows(
    boots: list[str], cell: str
) -> tuple[list[list[float]], list[list[float]]]:
    every, tail = [], []
    for boot in boots:
        rows = state_rows(boot, cell)
        rows.sort(key=lambda row: row[0])
        every.extend(rows)
        tail.extend(rows[int(len(rows) * (1 - TAIL)) :])
    return every, tail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="c16_events")
    parser.add_argument("--main", required=True)
    parser.add_argument("--branch", required=True)
    args = parser.parse_args()
    main_every, main_tail = arm_rows(args.main.split(","), args.cell)
    branch_every, branch_tail = arm_rows(args.branch.split(","), args.cell)
    print(
        f"requests: main {len(main_every)} (tail {len(main_tail)}), branch {len(branch_every)} (tail {len(branch_tail)})"
    )
    print(
        f"  {'state (ms)':40s}{'main p50':>10s}{'branch p50':>11s}{'main mean':>11s}{'branch mean':>12s}{'change':>9s}"
        f"{'main tail':>11s}{'branch tail':>12s}{'change':>9s}"
    )
    labels = ["server first audio", *(label for _, _, label in CHAIN[1:])]
    for index, label in enumerate(labels):
        main_values = [row[index] for row in main_every]
        branch_values = [row[index] for row in branch_every]
        main_tail_mean = statistics.fmean(row[index] for row in main_tail)
        branch_tail_mean = statistics.fmean(row[index] for row in branch_tail)
        main_mean = statistics.fmean(main_values)
        branch_mean = statistics.fmean(branch_values)
        print(
            f"  {label:40s}{statistics.median(main_values):10.1f}{statistics.median(branch_values):11.1f}"
            f"{main_mean:11.1f}{branch_mean:12.1f}{branch_mean - main_mean:+9.1f}"
            f"{main_tail_mean:11.1f}{branch_tail_mean:12.1f}{branch_tail_mean - main_tail_mean:+9.1f}"
        )


if __name__ == "__main__":
    main()
