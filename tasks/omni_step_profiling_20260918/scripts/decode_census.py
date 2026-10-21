"""Which vocoder decodes a served run made, from a decode_digest/sitecustomize.py census (OMNI_DECODE_CENSUS=1).

usage: python3 decode_census.py DIGEST_DIR [DIGEST_DIR ...]

Per directory, the vocoder process (the one with the most records): decodes and frames (rows by width) by kind, mode
and width, largest first, and each mode's share of the decodes and of the frames.
"""

import collections
import json
import sys
from pathlib import Path


def main() -> None:
    for directory in sys.argv[1:]:
        files = sorted(Path(directory).glob("*.jsonl"), key=lambda path: path.stat().st_size, reverse=True)
        entries = [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]
        decodes: collections.Counter = collections.Counter()
        frames: collections.Counter = collections.Counter()
        by_mode_decodes: collections.Counter = collections.Counter()
        by_mode_frames: collections.Counter = collections.Counter()
        for entry in entries:
            rows, width = entry["codes_shape"][0], entry["codes_shape"][-1]
            mode = entry.get("mode") or ("compiled" if entry.get("compiled") else "eager")
            key = (entry["kind"], mode, width)
            decodes[key] += 1
            frames[key] += rows * width
            by_mode_decodes[(entry["kind"], mode)] += 1
            by_mode_frames[(entry["kind"], mode)] += rows * width
        total_decodes, total_frames = sum(decodes.values()), sum(frames.values())
        print(f"{directory}: {total_decodes} decodes, {total_frames} row frames")
        for (kind, mode), count in by_mode_decodes.most_common():
            print(f"  {kind} {mode}: {count} decodes ({100 * count / total_decodes:.1f} %), "
                  f"{by_mode_frames[(kind, mode)]} row frames ({100 * by_mode_frames[(kind, mode)] / total_frames:.1f} %)")
        for key, count in decodes.most_common(25):
            print(f"    {key[0]} {key[1]} width {key[2]}: {count} decodes, {frames[key]} row frames")


if __name__ == "__main__":
    main()
