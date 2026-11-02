"""Two scored cells paired by sample: WER, bad samples, speaker similarity, a sign test and a bootstrap interval on WER.

A sample's word errors are summed over its passes. The sign test counts samples with fewer errors on A against more;
the interval resamples samples, then each sample's passes, so it carries both the corpus and the repeat draws.

usage: python3 cells_paired.py RUN_A RUN_B OUT_JSON   (each run holds seedtts_en/wer_results.json, sim/seedtts_en/sim_summary.json)
"""

import collections
import json
import math
import random
import sys
from pathlib import Path

BOOTSTRAPS = 4000
SEED = 20261102


def load(run: Path) -> dict:
    rows = json.loads((run / "seedtts_en" / "wer_results.json").read_text())["per_sample"]
    errors = collections.defaultdict(list)
    words = {}
    bad = 0
    bad_below = []
    for row in rows:
        sample = row["id"].split("__r")[0]
        count = row["substitutions"] + row["deletions"] + row["insertions"]
        errors[sample].append(count)
        words[sample] = len(row["ref_norm"].split())
        if row["wer"] > 0.5:
            bad += 1
        else:
            bad_below.append((count, words[sample]))
    similarity = json.loads((run / "sim" / "seedtts_en" / "sim_summary.json").read_text())["similarity"]["speaker_similarity_mean"]
    return {
        "errors": errors,
        "words": words,
        "rows": len(rows),
        "bad": bad,
        "wer": 100 * sum(map(sum, errors.values())) / sum(words[k] * len(errors[k]) for k in errors),
        "wer_without_bad": 100 * sum(c for c, _ in bad_below) / sum(w for _, w in bad_below),
        "similarity": similarity,
    }


def sign_test(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    else:
        return min(1.0, 2 * sum(math.comb(total, k) for k in range(0, min(wins, losses) + 1)) / 2**total)


def main() -> None:
    a = load(Path(sys.argv[1]))
    b = load(Path(sys.argv[2]))
    shared = sorted(set(a["errors"]) & set(b["errors"]))
    better = sum(sum(a["errors"][k]) * len(b["errors"][k]) < sum(b["errors"][k]) * len(a["errors"][k]) for k in shared)
    worse = sum(sum(a["errors"][k]) * len(b["errors"][k]) > sum(b["errors"][k]) * len(a["errors"][k]) for k in shared)
    generator = random.Random(SEED)
    deltas = []
    for _ in range(BOOTSTRAPS):
        picked = generator.choices(shared, k=len(shared))
        errors_a = errors_b = words = 0
        for sample in picked:
            passes = len(a["errors"][sample])
            errors_a += sum(generator.choices(a["errors"][sample], k=passes))
            errors_b += sum(generator.choices(b["errors"][sample], k=passes))
            words += a["words"][sample] * passes
        deltas.append(100 * (errors_a - errors_b) / words)
    deltas.sort()
    report = {
        "a": sys.argv[1],
        "b": sys.argv[2],
        "samples": len(shared),
        "rows": [a["rows"], b["rows"]],
        "wer": [a["wer"], b["wer"]],
        "wer_without_bad": [a["wer_without_bad"], b["wer_without_bad"]],
        "bad": [a["bad"], b["bad"]],
        "similarity": [a["similarity"], b["similarity"]],
        "samples_better_on_a": better,
        "samples_better_on_b": worse,
        "sign_p": sign_test(better, worse),
        "wer_a_minus_b_95": [deltas[int(0.025 * BOOTSTRAPS)], deltas[int(0.975 * BOOTSTRAPS) - 1]],
    }
    Path(sys.argv[3]).write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
