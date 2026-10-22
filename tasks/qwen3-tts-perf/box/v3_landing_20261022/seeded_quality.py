"""Seeded pairs of main against a vocoder-only change: with the same seed the talker draws the same codes in both
arms, so every difference in the audio is the vocoder's.

usage: python3 seeded_quality.py LABEL [LABEL ...]   (labels under PAIR_ROOT, arm a main, arm b the change)

Per label: prompts whose audio has the same length in both arms (same frame count, the codes' proxy), the waveform
SNR of b against a over those prompts, and the paired per-prompt speaker similarity and WER differences (b minus a)
with their standard errors; then the same over every label together.
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/v3_landing_20261022/pairs"))


def per_sample(path: Path, field: str) -> dict[str, float]:
    return {row["id"]: float(row[field]) for row in json.load(open(path))["per_sample"] if row["is_success"]}


def word_counts(path: Path) -> dict[str, tuple[int, int]]:
    """Per prompt: errors and reference words."""
    counts = {}
    for row in json.load(open(path))["per_sample"]:
        if row["is_success"]:
            errors = row["substitutions"] + row["deletions"] + row["insertions"]
            counts[row["id"]] = (errors, row["substitutions"] + row["deletions"] + row["hits"])
        else:
            pass
    return counts


def paired(differences: list[float]) -> str:
    mean = statistics.fmean(differences)
    error = statistics.stdev(differences) / math.sqrt(len(differences))
    return f"{mean:+.4f} (se {error:.4f}, 95 % {mean - 1.96 * error:+.4f} to {mean + 1.96 * error:+.4f})"


def main() -> None:
    all_sim, all_wer, all_snr = [], [], []
    for label in sys.argv[1:]:
        arms = {arm: ROOT / label / arm / "bench" for arm in ("a", "b")}
        sims = {arm: per_sample(path / "similarity_results.json", "speaker_similarity") for arm, path in arms.items()}
        wers = {arm: per_sample(path / "wer_results.json", "wer") for arm, path in arms.items()}
        counts = {arm: word_counts(path / "wer_results.json") for arm, path in arms.items()}
        same_length, snrs = 0, []
        common = sorted(set(sims["a"]) & set(sims["b"]))
        for prompt in common:
            a, _ = soundfile.read(arms["a"] / "audio" / f"{prompt}.wav", dtype="float64")
            b, _ = soundfile.read(arms["b"] / "audio" / f"{prompt}.wav", dtype="float64")
            if len(a) == len(b):
                same_length += 1
                noise = float(np.sum((a - b) ** 2))
                snrs.append(float("inf") if noise == 0 else 10 * math.log10(float(np.sum(a**2)) / noise))
            else:
                pass
        sim_diff = [sims["b"][prompt] - sims["a"][prompt] for prompt in common]
        wer_common = sorted(set(wers["a"]) & set(wers["b"]))
        wer_diff = [wers["b"][prompt] - wers["a"][prompt] for prompt in wer_common]
        corpus = {
            arm: 100 * sum(counts[arm][p][0] for p in wer_common) / sum(counts[arm][p][1] for p in wer_common)
            for arm in arms
        }
        finite = sorted(snr for snr in snrs if math.isfinite(snr))
        print(f"{label}: {len(common)} prompts scored in both, {same_length} of the same length")
        print(f"  waveform SNR b against a: median {statistics.median(finite):.1f} dB, 5th percentile "
              f"{finite[len(finite) // 20]:.1f} dB, identical {len(snrs) - len(finite)}")
        print(f"  SIM a {statistics.fmean(sims['a'][p] for p in common):.4f}, b "
              f"{statistics.fmean(sims['b'][p] for p in common):.4f}, paired b - a {paired(sim_diff)}")
        print(f"  WER corpus a {corpus['a']:.3f} %, b {corpus['b']:.3f} %; per prompt b - a {paired(wer_diff)}; "
              f"prompts whose WER differs {sum(d != 0 for d in wer_diff)}")
        all_sim += sim_diff
        all_wer += wer_diff
        all_snr += finite
    print(f"all: SIM paired b - a {paired(all_sim)}; WER per prompt b - a {paired(all_wer)}; "
          f"SNR median {statistics.median(all_snr):.1f} dB over {len(all_snr)} prompts")


if __name__ == "__main__":
    main()
