"""WER and speaker similarity of two arms over many seeded runs, pooled.

Each seeded c1 run of an arm is deterministic, so every seed is one independent draw of the
whole corpus for both arms. Reads run_ab_pairs.sh round directories (<round>_a, <round>_b,
each with bench/wer_results.json and bench/similarity_results.json) and prints:

  per seed   corpus SIM and WER (below 50%) of both arms and the difference
  seeds      the mean difference, its standard error and t, how many seeds are lower, a
             bootstrap 95% interval over seeds
  utterances every utterance averaged over all seeds in each arm, then paired: mean and
             standard error of the per-utterance difference, how many utterances are lower
  voices     per reference voice (the prompt id), the averaged difference; the voices with
             the largest drops and gains, with their utterance counts
  wer        per-utterance WER pooled the same way, and the utterances whose mean WER moves
             most

usage: python seed_pool_stats.py ROUND_DIR_PREFIX... (e.g. /data/p4ab/c1_seeded
       /data/p4seeds/c1_seed_11 ...)
"""

from __future__ import annotations

import collections
import json
import random
import statistics
import sys


def load(round_dir: str, arm: str):
    bench = f"{round_dir}_{arm}/bench"
    with open(f"{bench}/similarity_results.json") as handle:
        sim = json.load(handle)
    with open(f"{bench}/wer_results.json") as handle:
        wer = json.load(handle)
    per_sim = {
        row["id"]: row["speaker_similarity"]
        for row in sim["per_sample"]
        if row.get("speaker_similarity") is not None
    }
    per_wer = {
        row["id"]: row["wer"] for row in wer["per_sample"] if row.get("wer") is not None
    }
    return (
        sim["summary"]["speaker_similarity_mean"],
        wer["summary"]["wer_below_50_corpus"] * 100,
        per_sim,
        per_wer,
    )


def mean_se(values):
    mean = statistics.mean(values)
    se = (
        statistics.stdev(values) / len(values) ** 0.5
        if len(values) > 1
        else float("nan")
    )
    return mean, se


def main() -> None:
    rounds = sys.argv[1:]
    seeds = []
    for round_dir in rounds:
        a, b = load(round_dir, "a"), load(round_dir, "b")
        seeds.append((round_dir.rsplit("/", 1)[-1], a, b))
    print(
        f"{'round':>18} {'SIM A':>8} {'SIM B':>8} {'dSIM':>7} {'WER A':>7} {'WER B':>7} {'dWER':>7}"
    )
    for name, a, b in seeds:
        print(
            f"{name:>18} {a[0]:>8.3f} {b[0]:>8.3f} {b[0] - a[0]:>+7.3f} {a[1]:>7.3f} {b[1]:>7.3f} {b[1] - a[1]:>+7.3f}"
        )
    for label, index in (("SIM", 0), ("WER points", 1)):
        deltas = [b[index] - a[index] for _, a, b in seeds]
        mean, se = mean_se(deltas)
        rng = random.Random(0)
        boots = sorted(
            statistics.mean(rng.choice(deltas) for _ in deltas) for _ in range(20000)
        )
        print(
            f"\nseeds {label}: mean difference {mean:+.4f}, se {se:.4f}, t {mean / se:+.2f}, "
            f"lower in {sum(d < 0 for d in deltas)} of {len(deltas)}, bootstrap 95% "
            f"[{boots[500]:+.4f}, {boots[19500]:+.4f}]"
        )
        arm_a = [a[index] for _, a, _ in seeds]
        arm_b = [b[index] for _, _, b in seeds]
        print(
            f"  A across seeds {min(arm_a):.3f} to {max(arm_a):.3f} (sd {statistics.stdev(arm_a):.3f}); "
            f"B {min(arm_b):.3f} to {max(arm_b):.3f} (sd {statistics.stdev(arm_b):.3f})"
        )
    for label, index in (("SIM", 2), ("WER", 3)):
        pooled = {
            "a": collections.defaultdict(list),
            "b": collections.defaultdict(list),
        }
        for _, a, b in seeds:
            for arm, data in (("a", a), ("b", b)):
                for key, value in data[index].items():
                    pooled[arm][key].append(value)
        keys = sorted(set(pooled["a"]) & set(pooled["b"]))
        diffs = {
            key: statistics.mean(pooled["b"][key]) - statistics.mean(pooled["a"][key])
            for key in keys
        }
        mean, se = mean_se(list(diffs.values()))
        print(
            f"\nutterances {label}: {len(keys)} utterances averaged over {len(seeds)} seeds per arm; "
            f"paired mean difference {mean:+.4f}, se {se:.4f}, t {mean / se:+.2f}, "
            f"lower in {sum(d < 0 for d in diffs.values())}"
        )
        if label == "SIM":
            voices = collections.defaultdict(list)
            for key, value in diffs.items():
                voices[key.split("-")[0]].append(value)
            ranked = sorted(
                (statistics.mean(v), name, len(v)) for name, v in voices.items()
            )
            print(
                f"  voices: {len(voices)}; largest drops and gains (mean difference, utterances)"
            )
            for value, name, count in ranked[:8]:
                print(f"    {value:+.3f}  {name}  {count}")
            print("    ...")
            for value, name, count in ranked[-5:]:
                print(f"    {value:+.3f}  {name}  {count}")
            voice_means = [value for value, _, _ in ranked]
            vm, vse = mean_se(voice_means)
            print(
                f"  per voice: mean {vm:+.4f}, se {vse:.4f}, lower in {sum(v < 0 for v in voice_means)} of {len(voice_means)}"
            )
        else:
            moved = sorted(diffs.items(), key=lambda item: item[1])
            print("  utterances whose mean WER moves most (B minus A):")
            for key, value in moved[:3] + moved[-5:]:
                print(f"    {value:+.4f}  {key}")


if __name__ == "__main__":
    main()
