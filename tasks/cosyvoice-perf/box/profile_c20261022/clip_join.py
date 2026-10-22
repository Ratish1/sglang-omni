"""The quality draws' second arm minus first, split by whether a clip's reference tokens differ between ONNX Runtime
and L3's graphs (tokenizer_flips.py): per family the corpus WER delta carried by each set of clips (errors over the
cell's reference words, so the two parts sum to the delta) and the SIM delta of each set, with standard errors over
the draws; then the clips whose errors moved most over every draw, each with its flip count and the draws it was
worse and better in. A seeded cell under one seed in both orders is one draw, as in quality_draws.py.
usage: python3 clip_join.py <run home> <flips json> <pair> [<pair> ...]"""

import json
import statistics
import sys
from pathlib import Path

SEEDED = ("c1", "c1f", "b1f")


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def per_clip(wer, sim):
    errors = {
        row["id"]: (row["substitutions"] + row["deletions"] + row["insertions"], row["substitutions"] + row["deletions"] + row["hits"])
        for row in wer["per_sample"]
        if row.get("wer") is not None
    }
    sims = {row["id"]: float(row["speaker_similarity"]) for row in sim["per_sample"] if row.get("speaker_similarity") is not None}
    return errors, sims


def stats(values):
    mean = statistics.mean(values)
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0
    return mean, se


def main():
    home, flips = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    flipped = {clip for clip, row in flips.items() if row["ort_vs_graph"] > 0}
    draws = {}
    for name in sys.argv[3:]:
        for order in ("a", "b"):
            gpus = {"first": "gpu0", "second": "gpu1"} if order == "a" else {"first": "gpu1", "second": "gpu0"}
            for cell_dir in sorted((home / name / order / gpus["first"]).iterdir()):
                cell = cell_dir.name
                arms = {}
                for arm, gpu in gpus.items():
                    directory = home / name / order / gpu / cell
                    wer, sim = load(directory / "wer_results.json"), load(directory / "similarity_results.json")
                    if wer and sim:
                        arms[arm] = per_clip(wer, sim)
                if len(arms) < 2:
                    continue
                family = cell.partition("_s")[0]
                key = (name, cell) if family in SEEDED else (name, order, cell)
                draws.setdefault(family, {}).setdefault(key, []).append(arms)
    print(f"references with ONNX Runtime against graph token differences: {len(flipped)} of {len(flips)}")
    for family, keyed in sorted(draws.items()):
        parts = {"flipped": [], "rest": []}
        sim_parts = {"flipped": [], "rest": []}
        moves = {}
        for runs in keyed.values():
            for label, clips_of in (("flipped", lambda ids: ids & flipped), ("rest", lambda ids: ids - flipped)):
                wer_deltas, sim_deltas = [], []
                for arms in runs:
                    (first_errors, first_sims), (second_errors, second_sims) = arms["first"], arms["second"]
                    words = sum(count for _, count in first_errors.values())
                    ids = clips_of(first_errors.keys() & second_errors.keys())
                    wer_deltas.append(100 * sum(second_errors[clip][0] - first_errors[clip][0] for clip in ids) / words)
                    sim_ids = clips_of(first_sims.keys() & second_sims.keys())
                    sim_deltas.append(statistics.mean(second_sims[clip] - first_sims[clip] for clip in sim_ids))
                parts[label].append(statistics.mean(wer_deltas))
                sim_parts[label].append(statistics.mean(sim_deltas))
            common = set.intersection(*(run["first"][0].keys() & run["second"][0].keys() for run in runs))
            for clip in common:
                delta = sum(run["second"][0][clip][0] - run["first"][0][clip][0] for run in runs) / len(runs)
                moves.setdefault(clip, []).append(delta)
        print()
        print(f"{family}: {len(keyed)} draws")
        for label in ("flipped", "rest"):
            wer_mean, wer_se = stats(parts[label])
            sim_mean, sim_se = stats(sim_parts[label])
            print(f"  {label}: corpus WER delta part {wer_mean:+.3f} points (se {wer_se:.3f}), SIM delta {sim_mean:+.3f} (se {sim_se:.3f})")
        ranked = sorted(moves.items(), key=lambda item: -abs(sum(item[1])))[:12]
        for clip, values in ranked:
            print(
                f"    {clip[:44]}: error delta sum {sum(values):+.1f} over {len(values)} draws, worse {sum(v > 0 for v in values)}, "
                f"better {sum(v < 0 for v in values)}, flips {flips.get(clip, {}).get('ort_vs_graph', 'n/a')}"
            )


if __name__ == "__main__":
    main()
