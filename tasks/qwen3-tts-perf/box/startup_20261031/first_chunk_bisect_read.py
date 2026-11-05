"""Reads first_chunk_tail.py pairs of one block: per arm against main, the client's first chunk with the opening burst of 64 apart, and
the median of each request state from the events. The difference per pair is what a component's bisect is read by.

usage: python3 first_chunk_bisect_read.py BLOCK_DIR ARM [ARM ...]   (each pair is main_<arm>_<n> and <arm>_<n>, n from 0)
"""

import collections
import glob
import json
import statistics
import sys
from pathlib import Path

BURST = 64
GAPS = (
    (
        "preprocess",
        ("preprocessing", "stage_dispatch"),
        ("preprocessing", "stage_complete"),
    ),
    (
        "engine",
        ("tts_engine", "stage_input_received"),
        ("tts_engine", "scheduler_prefill_start"),
    ),
    (
        "prefill",
        ("tts_engine", "scheduler_prefill_start"),
        ("tts_engine", "scheduler_prefill_end"),
    ),
    (
        "queue",
        ("vocoder", "qwen3_tts_vocoder_decode_enqueued"),
        ("vocoder", "qwen3_tts_vocoder_decode_dispatched"),
    ),
    (
        "launch",
        ("vocoder", "qwen3_tts_vocoder_decode_dispatched"),
        ("vocoder", "qwen3_tts_vocoder_decode_launched"),
    ),
    (
        "decode",
        ("vocoder", "qwen3_tts_vocoder_decode_launched"),
        ("vocoder", "qwen3_tts_vocoder_decode_resolved"),
    ),
)


def read_arm(root: Path) -> dict[str, float]:
    records = json.loads(
        (root / "stream" / "bench" / "speed_results.json").read_text()
    )["per_request"]
    first_chunks = [row["audio_ttfp_s"] * 1000 for row in records if row["is_success"]]
    after_burst = first_chunks[BURST:]
    row = {
        "mean": statistics.mean(first_chunks),
        "p50": statistics.median(first_chunks),
        "burst": statistics.mean(first_chunks[:BURST]),
        "rest": statistics.mean(after_burst),
        "rest50": statistics.median(after_burst),
    }
    milestones: dict[str, dict[tuple[str, str], int]] = collections.defaultdict(dict)
    for path in glob.glob(str(root / "events" / "events_*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                key = (event["stage"], event["event_name"])
                stamps = milestones[event["request_id"]]
                stamps[key] = min(
                    stamps.get(key, event["timestamp_ns"]), event["timestamp_ns"]
                )
    for name, earlier, later in GAPS:
        row[name] = statistics.median(
            (stamps[later] - stamps[earlier]) / 1e6
            for stamps in milestones.values()
            if earlier in stamps and later in stamps
        )
    return row


def main() -> None:
    block = Path(sys.argv[1])
    columns = ("mean", "p50", "burst", "rest", "rest50", *(gap[0] for gap in GAPS))
    print(f"{'pair':18s} " + " ".join(f"{column:>8s}" for column in columns))
    for arm in sys.argv[2:]:
        differences: dict[str, list[float]] = collections.defaultdict(list)
        pair = 0
        while (block / f"{arm}_{pair}" / "stream").exists() and (
            block / f"main_{arm}_{pair}" / "stream"
        ).exists():
            base = read_arm(block / f"main_{arm}_{pair}")
            change = read_arm(block / f"{arm}_{pair}")
            for column in columns:
                differences[column].append(change[column] - base[column])
            print(
                f"{arm + ' ' + str(pair):18s} "
                + " ".join(f"{differences[column][-1]:+8.2f}" for column in columns),
                flush=True,
            )
            pair += 1
        if pair:
            print(
                f"{arm + ' mean':18s} "
                + " ".join(
                    f"{statistics.mean(differences[column]):+8.2f}"
                    for column in columns
                ),
                flush=True,
            )
        else:
            pass


if __name__ == "__main__":
    main()
