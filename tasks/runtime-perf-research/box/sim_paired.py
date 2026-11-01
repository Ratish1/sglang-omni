"""Speaker similarity paired by prompt: each prompt's mean over an arm's passes, the per-prompt difference between
arms, its mean with a 95 % interval (normal over prompts), and the same split per boot so the boot spread shows.

usage: python3 sim_paired.py <root> <arm a boots a1,a2> <arm b boots b1,b2>
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
arms = [sys.argv[2].split(","), sys.argv[3].split(",")]


def prompt_means(boots):
    values = defaultdict(list)
    for boot in boots:
        for rep in ("rep0", "rep1"):
            path = root / boot / "c64_clean/bench" / rep / "similarity_results.json"
            for sample in json.loads(path.read_text())["per_sample"]:
                if sample["is_success"] and sample["speaker_similarity"] is not None:
                    values[sample["id"]].append(sample["speaker_similarity"])
    return {prompt: statistics.mean(scores) for prompt, scores in values.items()}


def paired(a_boots, b_boots, label):
    a, b = prompt_means(a_boots), prompt_means(b_boots)
    shared = sorted(set(a) & set(b))
    diffs = [b[prompt] - a[prompt] for prompt in shared]
    mean = statistics.mean(diffs)
    half = 1.96 * statistics.stdev(diffs) / len(diffs) ** 0.5
    worse = sum(1 for d in diffs if d < 0)
    print(f"{label}: {len(shared)} prompts, arm a {statistics.mean(a[p] for p in shared):.4f}, "
          f"arm b {statistics.mean(b[p] for p in shared):.4f}, difference {mean:+.4f} (95 % {mean - half:+.4f} to "
          f"{mean + half:+.4f}), prompts lower in b {worse}")


paired(arms[0], arms[1], "pooled")
for a_boot in arms[0]:
    for b_boot in arms[1]:
        paired([a_boot], [b_boot], f"{a_boot} against {b_boot}")
paired([arms[0][0]], [arms[0][1]], f"{arms[0][0]} against {arms[0][1]} (same arm)")
paired([arms[1][0]], [arms[1][1]], f"{arms[1][0]} against {arms[1][1]} (same arm)")
