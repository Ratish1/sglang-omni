"""SeedTTS WER of two arms, paired by sample: corpus WER per boot, the arm difference with a bootstrap over samples,
each arm's boot-to-boot difference as its own band, and the samples that carry the difference.

usage: python3 wer_paired.py <base arm>=<bench dir>,<bench dir> <pr arm>=<bench dir>,<bench dir>
"""

import json
import random
import sys
from pathlib import Path

BOOTSTRAP_DRAWS = 2000
TOP_SAMPLES = 8


def boot_errors(bench: str) -> dict[str, tuple[int, int]]:
    rows = json.loads((Path(bench) / "wer_results.json").read_text(encoding="utf-8"))[
        "per_sample"
    ]
    return {
        row["id"]: (
            row["substitutions"] + row["deletions"] + row["insertions"],
            row["substitutions"] + row["deletions"] + row["hits"],
        )
        for row in rows
    }


def corpus_wer(errors: list[int], words: list[int]) -> float:
    return 100.0 * sum(errors) / sum(words)


def main() -> None:
    arms = []
    for spec in sys.argv[1:3]:
        arm, benches = spec.split("=")
        arms.append((arm, [boot_errors(bench) for bench in benches.split(",")]))
    (base_name, base_boots), (pr_name, pr_boots) = arms
    sample_ids = sorted(
        set.intersection(*(set(boot) for boot in base_boots + pr_boots))
    )
    words = [base_boots[0][sample_id][1] for sample_id in sample_ids]
    base_errors = [
        sum(boot[sample_id][0] for boot in base_boots) / len(base_boots)
        for sample_id in sample_ids
    ]
    pr_errors = [
        sum(boot[sample_id][0] for boot in pr_boots) / len(pr_boots)
        for sample_id in sample_ids
    ]
    print(f"samples {len(sample_ids)}")
    for name, boots in arms:
        per_boot = [
            corpus_wer([boot[sample_id][0] for sample_id in sample_ids], words)
            for boot in boots
        ]
        aa = per_boot[1] - per_boot[0] if len(per_boot) == 2 else float("nan")
        print(
            f"{name}: corpus WER per boot {' / '.join(f'{value:.3f}' for value in per_boot)}, boot difference {aa:+.3f} pp"
        )
    difference = corpus_wer(pr_errors, words) - corpus_wer(base_errors, words)
    rng = random.Random(3451)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        picked = [rng.randrange(len(sample_ids)) for _ in sample_ids]
        picked_words = [words[index] for index in picked]
        draws.append(
            corpus_wer([pr_errors[index] for index in picked], picked_words)
            - corpus_wer([base_errors[index] for index in picked], picked_words)
        )
    draws.sort()
    low = draws[int(0.025 * BOOTSTRAP_DRAWS)]
    high = draws[int(0.975 * BOOTSTRAP_DRAWS)]
    print(
        f"{pr_name} - {base_name}: {difference:+.3f} pp, 95 % bootstrap over samples [{low:+.3f}, {high:+.3f}]"
    )
    total_words = sum(words)
    gaps = sorted(
        zip(sample_ids, base_errors, pr_errors),
        key=lambda row: abs(row[2] - row[1]),
        reverse=True,
    )
    print(
        f"largest per-sample error differences (mean errors over boots; {total_words} reference words):"
    )
    for sample_id, base_error, pr_error in gaps[:TOP_SAMPLES]:
        print(f"  {sample_id}: {base_name} {base_error:.1f}, {pr_name} {pr_error:.1f}")
    without_top = [row for row in gaps[TOP_SAMPLES:]]
    kept_words = [base_boots[0][row[0]][1] for row in without_top]
    trimmed = corpus_wer([row[2] for row in without_top], kept_words) - corpus_wer(
        [row[1] for row in without_top], kept_words
    )
    print(f"difference without those {TOP_SAMPLES} samples: {trimmed:+.3f} pp")


if __name__ == "__main__":
    main()
