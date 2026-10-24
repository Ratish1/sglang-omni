"""The TTS CI's quality metrics read from same-time pairs: per arm the corpus WER, the corpus WER without samples past
50 %, requests at the token cap, and the speaker similarity mean over all samples and over the first 50 in dataset order
(the CI's similarity check scores --max-samples 50); then b minus a per sample over those 50 and over all.

usage: PAIR_ROOT=<pairs> PYTHONPATH=<omni tree> python3 ci_metrics.py LABEL [LABEL ...]
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples

ROOT = Path(os.environ["PAIR_ROOT"])
CI_SIMILARITY_SAMPLES = 50


def paired(a: dict[str, float], b: dict[str, float], ids: list[str]) -> str:
    deltas = [b[sample_id] - a[sample_id] for sample_id in ids if sample_id in a and sample_id in b]
    mean = statistics.fmean(deltas)
    error = statistics.stdev(deltas) / math.sqrt(len(deltas))
    return f"{mean:+.3f} (95 % {mean - 1.96 * error:+.3f} to {mean + 1.96 * error:+.3f}, n {len(deltas)})"


def main() -> None:
    first = [sample.sample_id for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, CI_SIMILARITY_SAMPLES, split="en")]
    for label in sys.argv[1:]:
        similarity = {}
        for arm in ("a", "b"):
            bench = ROOT / label / arm / "bench"
            wer = json.loads((bench / "wer_results.json").read_text(encoding="utf-8"))["summary"]
            speed = json.loads((bench / "speed_results.json").read_text(encoding="utf-8"))["summary"]
            path = bench / "similarity_results.json"
            if path.exists():
                rows = json.loads(path.read_text(encoding="utf-8"))["per_sample"]
                similarity[arm] = {row["id"]: row["speaker_similarity"] for row in rows if row["is_success"]}
                scored = [similarity[arm][sample_id] for sample_id in first if sample_id in similarity[arm]]
                sim_text = (
                    f"SIM all {statistics.fmean(similarity[arm].values()):.2f}, "
                    f"first 50 {statistics.fmean(scored):.2f} over {len(scored)}"
                )
            else:
                sim_text = "SIM unscored"
            print(
                f"{label} {arm}: WER {wer['wer_corpus'] * 100:.2f} %, without >50 % {wer['wer_below_50_corpus'] * 100:.2f} %, "
                f">50 % {wer['n_above_50_pct_wer']}, cap hits {speed['max_token_hits']}, {sim_text}"
            )
        if len(similarity) == 2:
            print(
                f"{label} b minus a: SIM first 50 {paired(similarity['a'], similarity['b'], first)}, "
                f"all {paired(similarity['a'], similarity['b'], sorted(similarity['a']))}"
            )
        else:
            pass


if __name__ == "__main__":
    main()
