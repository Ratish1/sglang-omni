"""Pool duplex_ab.py's seeded pairs per set: every pair of a group (base dir, other dir) drawn with the same seed,
the per-utterance WER and SIM deltas of natural-stop pairs pooled over the group's seeds, with corpus error, failures
over 50 % and length stops summed per arm.

usage: python3 pooled_pairs.py GROUP BASE_DIR OTHER_DIR [BASE_DIR OTHER_DIR ...] [-- GROUP ...]
"""

import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

WER_FAILURE = 0.5


def load(out_dir: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (out_dir / "scored.jsonl").read_text(encoding="utf-8").splitlines() if line]


def interval(deltas: list[float], scale: float) -> str:
    mean = statistics.fmean(deltas) * scale
    half = 1.96 * statistics.stdev(deltas) * scale / math.sqrt(len(deltas))
    return f"{mean:+.3f} +- {half:.3f}"


def pool(group: str, dirs: list[Path]) -> None:
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[str, list[int]] = defaultdict(lambda: [0] * 8)
    for base_dir, other_dir in zip(dirs[::2], dirs[1::2]):
        by_id = {(str(record["set"]), str(record["id"])): record for record in load(other_dir)}
        for base in load(base_dir):
            set_name = str(base["set"])
            other = by_id[(set_name, str(base["id"]))]
            tally = counts[set_name]
            for offset, record in ((0, base), (4, other)):
                tally[offset] += record.get("finish_reason") == "length"
                if "word_errors" in record and record.get("finish_reason") == "stop":
                    tally[offset + 1] += int(record["word_errors"])
                    tally[offset + 2] += int(record["reference_words"])
                    tally[offset + 3] += float(record["wer"]) > WER_FAILURE
                else:
                    pass
            if base.get("finish_reason") == "stop" and other.get("finish_reason") == "stop":
                for key in ("wer", "similarity"):
                    if key in base and key in other:
                        sums[set_name][key].append(float(other[key]) - float(base[key]))
                    else:
                        pass
            else:
                pass
    for set_name in sorted(counts):
        tally = counts[set_name]
        print(
            f"{group} {set_name}: length {tally[0]} / {tally[4]} | corpus error {100 * tally[1] / tally[2]:.2f} / "
            f"{100 * tally[5] / tally[6]:.2f} % | over 50 % {tally[3]} / {tally[7]} | paired WER points "
            f"{interval(sums[set_name]['wer'], 100)} (n {len(sums[set_name]['wer'])}) | paired SIM "
            f"{interval(sums[set_name]['similarity'], 1)}",
            flush=True,
        )


def main() -> None:
    groups = " ".join(sys.argv[1:]).split(" -- ")
    for spec in groups:
        group, *paths = spec.split()
        pool(group, [Path(path) for path in paths])


if __name__ == "__main__":
    main()
