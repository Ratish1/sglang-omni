#!/usr/bin/env python3
"""Per boot, the median preprocessing time of its video requests in one events cell. Video preprocessing decodes frames
on a thread team across the card's socket, so a boot whose socket another tenant loads runs it 45 to 65 % slower while
the host load average stays low and single-threaded speech preprocessing stays level. A boot off its tree's usual value
(main e0482a08c 569 to 589 ms, stack 433 to 450 ms at speech 8/s beside video 1/s) voids its pair.

usage: python3 video_preprocess_gate.py --cell c64_events RUN_DIR BOOT [BOOT ...]
"""

import argparse
import glob
import json
import os


def video_preprocess_ms(boot_dir: str) -> list[float]:
    events: dict[str, dict[tuple[str, str], int]] = {}
    for path in glob.glob(os.path.join(boot_dir, "events", "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                if '"preprocessing"' not in line and '"image_encoder"' not in line:
                    continue
                else:
                    pass
                record = json.loads(line)
                events.setdefault(record["request_id"], {})[
                    (record["stage"], record["event_name"])
                ] = record["timestamp_ns"]
    start, end = ("preprocessing", "preprocess_start"), (
        "preprocessing",
        "preprocess_end",
    )
    return [
        (stamps[end] - stamps[start]) / 1e6
        for stamps in events.values()
        if start in stamps
        and end in stamps
        and ("image_encoder", "stage_input_received") in stamps
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="c64_events")
    parser.add_argument("run_dir")
    parser.add_argument("boots", nargs="+")
    args = parser.parse_args()
    for boot in args.boots:
        durations = sorted(
            video_preprocess_ms(os.path.join(args.run_dir, boot, args.cell))
        )
        if durations:
            print(
                f"gate {boot}: videos {len(durations)}, preprocess p50 {durations[len(durations) // 2]:.0f} ms"
            )
        else:
            print(f"gate {boot}: no video requests")


if __name__ == "__main__":
    main()
