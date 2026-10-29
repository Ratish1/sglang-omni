"""Tally failures per arm and set over scored duplex_ab.py runs: natural stops, length stops, natural stops over 50 %
WER, and babble (a natural stop over 50 % WER whose transcript repeats one short token six or more times, "A A A",
"hehehe", "when when when"); then, per shape, an exact two-sided binomial test of the second arm's share of the
babble against its share of natural stops.

usage: python3 babble_tally.py ARM=DIR[,DIR...] ARM=DIR[,DIR...]
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

WER_FAILURE = 0.5
BABBLE = re.compile(r"(\b\w{1,4}\b)(?:[\s,.-]+\1\b){5,}|(\w{1,3})\2{6,}", re.IGNORECASE)


def binomial_two_sided(successes: int, trials: int, share: float) -> float:
    def mass(k: int) -> float:
        return math.comb(trials, k) * share**k * (1 - share) ** (trials - k)

    observed = mass(successes)
    return min(1.0, sum(mass(k) for k in range(trials + 1) if mass(k) <= observed * (1 + 1e-9)))


def main() -> None:
    arms = [spec.split("=", 1) for spec in sys.argv[1:]]
    tallies: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    examples: dict[str, list[str]] = defaultdict(list)
    for arm, dirs in arms:
        for run_dir in dirs.split(","):
            for line in (Path(run_dir) / "scored.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                tally = tallies[(str(record["shape"]), str(record["set"]), arm)]
                if record.get("finish_reason") == "length":
                    tally[1] += 1
                elif record.get("finish_reason") == "stop" and "wer" in record:
                    tally[0] += 1
                    if float(record["wer"]) > WER_FAILURE:
                        tally[2] += 1
                        if BABBLE.search(str(record.get("transcript", ""))):
                            tally[3] += 1
                            examples[arm].append(f"{Path(run_dir).name} {record['id']}: {str(record['transcript'])[:70]}")
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
    names = [arm for arm, _ in arms]
    shapes = sorted({(shape, set_name) for shape, set_name, _ in tallies})
    per_shape: dict[str, list[list[int]]] = defaultdict(lambda: [[0, 0] for _ in names])
    for shape, set_name in shapes:
        cells = []
        for index, arm in enumerate(names):
            natural, length, over, babble = tallies[(shape, set_name, arm)]
            cells.append(f"{arm} natural {natural}, length {length}, over 50 % {over}, babble {babble}")
            per_shape[shape][index][0] += natural
            per_shape[shape][index][1] += babble
        print(f"{shape} {set_name}: " + " | ".join(cells), flush=True)
    for shape, counts in sorted(per_shape.items()):
        (natural_a, babble_a), (natural_b, babble_b) = counts[0], counts[1]
        trials = babble_a + babble_b
        share = natural_b / (natural_a + natural_b)
        p_value = binomial_two_sided(babble_b, trials, share) if trials else 1.0
        print(
            f"{shape} all sets: babble {names[0]} {babble_a} in {natural_a} ({1e5 * babble_a / natural_a:.1f} per 1e5), "
            f"{names[1]} {babble_b} in {natural_b} ({1e5 * babble_b / natural_b:.1f} per 1e5), two-sided p {p_value:.3f}",
            flush=True,
        )
    for arm in names:
        for example in examples[arm]:
            print(f"  {arm}: {example}", flush=True)


if __name__ == "__main__":
    main()
