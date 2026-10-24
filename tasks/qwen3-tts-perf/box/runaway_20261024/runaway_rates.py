"""Runaway rates per prompt, served trees against the official implementation: a served request ran away when its
audio reaches 48 s (600 frames; these prompts speak in under 10 s), a reference row when it reached max_new_tokens
without a codec EOS. Per prompt and source: runaways of requests with a Wilson 95 % interval, then two-sided Fisher
exact p for each served tree against the reference and between the served trees.

usage: PAIR_ROOT=<pairs> python3 runaway_rates.py --pairs LABEL [LABEL ...] --reference JSON [JSON ...]
"""

import argparse
import itertools
import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ["PAIR_ROOT"])
RUNAWAY_SECONDS = 48.0
REFERENCE = "reference qwen_tts"


def wilson(runaways: int, sent: int) -> tuple[float, float]:
    z = 1.96
    rate = runaways / sent
    denominator = 1 + z * z / sent
    centre = (rate + z * z / (2 * sent)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / sent + z * z / (4 * sent * sent)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def fisher(first: tuple[int, int], second: tuple[int, int]) -> float:
    (k1, n1), (k2, n2) = first, second
    total = k1 + k2

    def probability(k: int) -> float:
        return math.comb(n1, k) * math.comb(n2, total - k) / math.comb(n1 + n2, total)

    observed = probability(k1)
    support = range(max(0, total - n2), min(total, n1) + 1)
    return min(1.0, sum(probability(k) for k in support if probability(k) <= observed * (1 + 1e-9)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=[])
    parser.add_argument("--reference", nargs="+", default=[])
    args = parser.parse_args()
    counts: dict[str, dict[str, list[int]]] = {}
    for label in args.pairs:
        for arm in ("a", "b"):
            tree = (ROOT / label / arm / "tree.txt").read_text(encoding="utf-8").split()[0]
            results = json.loads((ROOT / label / arm / "bench" / "speed_results.json").read_text(encoding="utf-8"))
            for request in results["per_request"]:
                tally = counts.setdefault(tree, {}).setdefault(request["id"].rsplit("_r", 1)[0], [0, 0])
                tally[0] += float(request.get("audio_duration_s") or 0) >= RUNAWAY_SECONDS
                tally[1] += 1
    for path in args.reference:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for record in data["records"]:
            tally = counts.setdefault(REFERENCE, {}).setdefault(record["prompt"], [0, 0])
            tally[0] += record["frames"] >= data["max_new_tokens"] - 1
            tally[1] += 1
    prompts = sorted({prompt for by_prompt in counts.values() for prompt in by_prompt})
    for prompt in prompts + ["all"]:
        print(prompt)
        totals = {}
        for source, by_prompt in counts.items():
            if prompt == "all":
                totals[source] = (sum(t[0] for t in by_prompt.values()), sum(t[1] for t in by_prompt.values()))
            elif prompt in by_prompt:
                totals[source] = tuple(by_prompt[prompt])
            else:
                continue
            runaways, sent = totals[source]
            low, high = wilson(runaways, sent)
            print(f"  {source}: {runaways} of {sent} ({runaways / sent:.1%}, 95 % {low:.1%} to {high:.1%})")
        for first, second in itertools.combinations(sorted(totals), 2):
            print(f"  {first} against {second}: Fisher p {fisher(totals[first], totals[second]):.3f}")


if __name__ == "__main__":
    main()
