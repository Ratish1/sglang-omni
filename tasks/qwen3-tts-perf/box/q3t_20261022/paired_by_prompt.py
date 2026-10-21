"""Per-prompt paired read of pair.py runs: arm b minus arm a on each prompt both arms finished below the frame cap.

usage: python3 paired_by_prompt.py LABEL [LABEL ...]   (labels under PAIR_ROOT)

Per label: prompts paired, prompts capped in either arm, median and mean of b - a latency and first audio in ms, and
the share of prompts on which b's first audio is later.
"""

import csv
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/q3t_20261022/pairs"))
CAP_TOKENS = 2048


def rows(label: str, arm: str) -> dict[str, dict[str, str]]:
    with open(ROOT / label / arm / "bench" / "results.csv", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle) if row["is_success"] == "True"}


def main() -> None:
    print("label | paired | capped | latency b-a median / mean ms | first audio b-a median / mean ms | b later")
    for label in sys.argv[1:]:
        a, b = rows(label, "a"), rows(label, "b")
        capped = {
            key
            for side in (a, b)
            for key, row in side.items()
            if int(row["completion_tokens"]) >= CAP_TOKENS
        }
        keys = sorted((a.keys() & b.keys()) - capped)
        latency = [1000 * (float(b[key]["latency_s"]) - float(a[key]["latency_s"])) for key in keys]
        first = [1000 * (float(b[key]["audio_ttfp_s"]) - float(a[key]["audio_ttfp_s"])) for key in keys]
        later = sum(value > 0 for value in first) / len(first)
        print(
            f"{label} | {len(keys)} | {len(capped)} | {statistics.median(latency):+.2f} / {statistics.mean(latency):+.2f}"
            f" | {statistics.median(first):+.2f} / {statistics.mean(first):+.2f} | {100 * later:.0f} %"
        )


if __name__ == "__main__":
    main()
