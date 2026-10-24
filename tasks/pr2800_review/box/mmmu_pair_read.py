"""MMMU or Video-AMME results from videoamme_pair.py, paired item by item.

Run: python3 mmmu_pair_read.py <out> <pass>/<arm> <pass>/<arm> [<pass>/<arm> ...]
Each argument names one result, <out>/<pass>/<arm>/mmmu.json (or videoamme.json). Prints each result's correct count, and for every pair
of results the items whose raw responses are identical, the items only one side gets right with the exact McNemar p,
and each item whose parsed answers differ.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path


def records(out: Path, result: str) -> dict[str, dict]:
    name = "mmmu.json" if (out / result / "mmmu.json").exists() else "videoamme.json"
    rows = json.loads((out / result / name).read_text())["per_sample"]
    return {row["sample_id"]: row for row in rows}


def mcnemar_p(only_a: int, only_b: int) -> float:
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    else:
        pass
    tail = sum(math.comb(discordant, k) for k in range(min(only_a, only_b) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def main() -> None:
    out, results = Path(sys.argv[1]), sys.argv[2:]
    rows = {result: records(out, result) for result in results}
    for result in results:
        correct = sum(row["is_correct"] for row in rows[result].values())
        print(f"{result}: {correct} / {len(rows[result])} correct")
    for result_a, result_b in itertools.combinations(results, 2):
        items = sorted(set(rows[result_a]) & set(rows[result_b]))
        identical = [item for item in items if rows[result_a][item]["raw_response"] == rows[result_b][item]["raw_response"]]
        only_a = [item for item in items if rows[result_a][item]["is_correct"] and not rows[result_b][item]["is_correct"]]
        only_b = [item for item in items if rows[result_b][item]["is_correct"] and not rows[result_a][item]["is_correct"]]
        print(
            f"{result_a} vs {result_b}: {len(items)} items, raw responses identical {len(identical)}, "
            f"right only on {result_a} {len(only_a)}, only on {result_b} {len(only_b)}, "
            f"McNemar p {mcnemar_p(len(only_a), len(only_b)):.3f}"
        )
        for item in items:
            row_a, row_b = rows[result_a][item], rows[result_b][item]
            if row_a["predicted"] != row_b["predicted"]:
                print(
                    f"  {item} expected {row_a['expected']!r}: {row_a['predicted']!r} "
                    f"({'right' if row_a['is_correct'] else 'wrong'}, {row_a['completion_tokens']} tokens) / "
                    f"{row_b['predicted']!r} ({'right' if row_b['is_correct'] else 'wrong'}, "
                    f"{row_b['completion_tokens']} tokens)"
                )
            else:
                pass


if __name__ == "__main__":
    main()
