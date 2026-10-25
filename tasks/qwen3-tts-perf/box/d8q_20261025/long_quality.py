"""Main against the PR on long-reference pairs, quality only: per pair and pooled, main / PR for the corpus WER over all
samples, the corpus WER without samples past 50 %, the samples past 50 %, the share of target words spoken, and the
speaker similarity of the generated audio to each sample's own reference; then the per-prompt similarity difference
(PR minus main) over every prompt scored in both arms and over the prompts where neither arm inserted 5 or more words
(an output that first speaks the repeated reference's transcript), with the number of pairs where the PR is lower.

usage: python3 long_quality.py PAIR_DIR [PAIR_DIR ...]   (each with a/bench and b/bench; arm a main, arm b the PR)
"""

import json
import math
import statistics
import sys
from pathlib import Path

LEAK_INSERTIONS = 5


def per_sample(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    else:
        pass
    return {row["id"]: row for row in json.load(open(path))["per_sample"] if row["is_success"]}


def corpus_wer(rows: list[dict[str, float]]) -> float:
    errors = sum(row["substitutions"] + row["deletions"] + row["insertions"] for row in rows)
    words = sum(row["hits"] + row["substitutions"] + row["deletions"] for row in rows)
    return 100.0 * errors / words


def target_spoken(rows: list[dict[str, float]]) -> float:
    return sum(row["hits"] for row in rows) / sum(row["hits"] + row["substitutions"] + row["deletions"] for row in rows)


def interval(values: list[float]) -> str:
    if len(values) < 2:
        return f"{values[0]:+.3f} (n 1)" if values else "n 0"
    else:
        pass
    mean = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return f"{mean:+.3f} (95 % {mean - half:+.3f} to {mean + half:+.3f}, n {len(values)})"


def main() -> None:
    pooled_all: list[float] = []
    pooled_clean: list[float] = []
    lower = 0
    scored_pairs = 0
    for pair in (Path(argument) for argument in sys.argv[1:]):
        wer = {arm: per_sample(pair / arm / "bench" / "wer_results.json") for arm in ("a", "b")}
        similarity = {arm: per_sample(pair / arm / "bench" / "similarity_results.json") for arm in ("a", "b")}
        line = [pair.name]
        for arm in ("a", "b"):
            rows = list(wer[arm].values())
            kept = [row for row in rows if row["wer"] <= 0.5]
            line.append(
                f"{'main' if arm == 'a' else 'PR'}: WER all {corpus_wer(rows):.2f} %, without >50 % "
                f"{corpus_wer(kept):.2f} %, >50 % {len(rows) - len(kept)}, target spoken {target_spoken(rows):.4f}"
            )
        both = sorted(set(similarity["a"]) & set(similarity["b"]))
        if both:
            scored_pairs += 1
            main_sim = statistics.fmean(similarity["a"][sample]["speaker_similarity"] for sample in both)
            pr_sim = statistics.fmean(similarity["b"][sample]["speaker_similarity"] for sample in both)
            lower += pr_sim < main_sim
            deltas = [
                similarity["b"][sample]["speaker_similarity"] - similarity["a"][sample]["speaker_similarity"]
                for sample in both
            ]
            clean = [
                delta
                for sample, delta in zip(both, deltas)
                if sample in wer["a"]
                and sample in wer["b"]
                and wer["a"][sample]["insertions"] < LEAK_INSERTIONS
                and wer["b"][sample]["insertions"] < LEAK_INSERTIONS
            ]
            pooled_all.extend(deltas)
            pooled_clean.extend(clean)
            line.append(
                f"SIM main {main_sim:.2f} / PR {pr_sim:.2f} (PR minus main {pr_sim - main_sim:+.2f} over {len(both)}; "
                f"clean prompts {interval(clean)})"
            )
        else:
            line.append("SIM not scored")
        print(" | ".join(line))
    print(f"pooled SIM, PR minus main per prompt: all {interval(pooled_all)}; clean {interval(pooled_clean)}")
    print(f"pairs where the PR's SIM is lower: {lower} of {scored_pairs}")


if __name__ == "__main__":
    main()
