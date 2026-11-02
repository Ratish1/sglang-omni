"""Seeded quality of a vocoder change against block 79's records of #3418's head, with no new base run. One server on one
card, SeedTTS EN 1.7B Base at concurrency 1, seed 1234, the same driver as block 79, non-streaming then streaming. Both
modes' files are hashed against block 79's. Non-streaming is scored (the same talker codes through another decode), then
paired per request: SIM with a bootstrap interval, WER over all files, and the rows that moved most.

usage: python3 whole_seeded_quality.py --tree TREE --base-root /data/.../block79/identity --root DIR [--card 1] [--port 20800]
"""

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from seeded_tail import audio_hashes, score_card, serve_phase  # noqa: E402

RUNS = (("c1n", 1, 1234, False), ("c1s", 1, 1234, True))


def load(out: Path) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
    bench = out / "bench"
    similarity = json.loads((bench / "similarity_results.json").read_text())["per_sample"]
    with (bench / "wer_results.csv").open() as handle:
        wer = {row["id"]: row for row in csv.DictReader(handle)}
    return {row["id"]: float(row["speaker_similarity"]) for row in similarity}, wer


def word_counts(table: dict[str, dict[str, str]], keys: list[str]) -> tuple[int, int]:
    errors = sum(int(table[key]["substitutions"]) + int(table[key]["deletions"]) + int(table[key]["insertions"]) for key in keys)
    words = sum(int(table[key]["substitutions"]) + int(table[key]["deletions"]) + int(table[key]["hits"]) for key in keys)
    return errors, words


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--card", type=int, default=1)
    parser.add_argument("--port", type=int, default=20800)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    serve_phase("seeded", [("arm", args.tree, args.card, args.port)], list(RUNS), args.root)
    for run, _, _, _ in RUNS:
        base, arm = audio_hashes(args.base_root / run / "branch"), audio_hashes(args.root / run / "arm")
        print(f"{run}: byte identical to block 79 {sum(base[name] == arm.get(name) for name in base)} of {len(base)} ({len(arm)} here)", flush=True)
    score_card(args.card, args.port + 200, args.tree, [args.root / "c1n" / "arm"], args.root)
    base_sim, base_wer = load(args.base_root / "c1n" / "branch")
    arm_sim, arm_wer = load(args.root / "c1n" / "arm")
    keys = sorted(set(base_sim) & set(arm_sim))
    deltas = [arm_sim[key] - base_sim[key] for key in keys]
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    print(f"c1n SIM over {len(keys)}: base {statistics.mean(base_sim[key] for key in keys):.2f} arm {statistics.mean(arm_sim[key] for key in keys):.2f} "
          f"paired {statistics.mean(deltas):+.3f} (95 % {means[50]:+.3f} to {means[1949]:+.3f}); below 40: {sum(base_sim[key] < 40 for key in keys)} / {sum(arm_sim[key] < 40 for key in keys)}", flush=True)
    wer_keys = sorted(set(base_wer) & set(arm_wer))
    base_errors, words = word_counts(base_wer, wer_keys)
    arm_errors, _ = word_counts(arm_wer, wer_keys)
    print(f"c1n WER over {words} words: base {100 * base_errors / words:.2f} % ({base_errors}) arm {100 * arm_errors / words:.2f} % ({arm_errors}); "
          f"above 50 %: {sum(float(base_wer[key]['wer']) > 0.5 for key in wer_keys)} / {sum(float(arm_wer[key]['wer']) > 0.5 for key in wer_keys)}", flush=True)
    for key in sorted(keys, key=lambda item: arm_sim[item] - base_sim[item])[:8]:
        print(f"worst SIM {key}: {base_sim[key]:.2f} to {arm_sim[key]:.2f}; WER {base_wer[key]['wer']} to {arm_wer[key]['wer']}", flush=True)


if __name__ == "__main__":
    main()
