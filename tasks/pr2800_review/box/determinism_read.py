"""Whether repeated passes of a text benchmark from videoamme_pair.py return the same answers, text for text.

Run: python3 determinism_read.py <benchmark> <out>/<pass>/<arm> [<out>/<pass>/<arm> ...]
Each argument names one pass's directory holding <benchmark>.json. Prints each pass's correct count and, for every
pair of passes, how many items have byte-identical responses and how many flip correctness each way.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path


def records(directory: Path, benchmark: str) -> dict[str, dict]:
    rows = json.loads((directory / f"{benchmark}.json").read_text())["per_sample"]
    return {str(row["sample_id"]): row for row in rows}


def main() -> None:
    benchmark, directories = sys.argv[1], [Path(argument) for argument in sys.argv[2:]]
    rows = {directory: records(directory, benchmark) for directory in directories}
    for directory in directories:
        correct = sum(bool(row["is_correct"]) for row in rows[directory].values())
        print(f"{directory}: {correct} / {len(rows[directory])} correct")
    for first, second in itertools.combinations(directories, 2):
        items = sorted(set(rows[first]) & set(rows[second]))
        identical = sum(rows[first][item]["raw_response"] == rows[second][item]["raw_response"] for item in items)
        only_first = sum(rows[first][item]["is_correct"] and not rows[second][item]["is_correct"] for item in items)
        only_second = sum(rows[second][item]["is_correct"] and not rows[first][item]["is_correct"] for item in items)
        print(
            f"{first.parent.name}/{first.name} vs {second.parent.name}/{second.name}: {identical} / {len(items)} "
            f"identical, right only on the first {only_first}, only on the second {only_second}"
        )


if __name__ == "__main__":
    main()
