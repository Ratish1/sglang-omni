"""Seeded quality of a vocoder change against another tree at the same time. One server per card, SeedTTS EN at concurrency
1, seed 1234, non-streaming then streaming: both arms decode the same talker codes, so what moves is the decode alone. Both
modes' files are hashed, non-streaming is scored per card, then paired per request: SIM with a bootstrap interval, WER over
all files, and the rows that moved most. BOX_TTS_MODEL picks the checkpoint.

usage: python3 whole_seeded_pair.py --base TREE --arm TREE --root DIR [--port 16000] [--seed 1234] [--no-streaming]
"""

import argparse
import concurrent.futures
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from seeded_tail import MODEL, audio_hashes, score_card, serve_phase  # noqa: E402
from whole_seeded_quality import load, word_counts  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--compare-only", action="store_true", help="read the scored files already under --root")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    print(f"model {MODEL} seed {args.seed}", flush=True)
    if not args.compare_only:
        runs = [("c1n", 1, args.seed, False)] if args.no_streaming else [("c1n", 1, args.seed, False), ("c1s", 1, args.seed, True)]
        serve_phase("seeded", [("base", args.base, 0, args.port), ("arm", args.arm, 1, args.port + 200)], runs, args.root)
        for run, _, _, _ in runs:
            base, arm = audio_hashes(args.root / run / "base"), audio_hashes(args.root / run / "arm")
            print(f"{run}: byte identical {sum(base[name] == arm.get(name) for name in base)} of {len(base)} ({len(arm)} in the arm)", flush=True)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(
                lambda job: score_card(job[0], job[1], job[2], [args.root / "c1n" / job[3]], args.root),
                [(0, args.port + 400, args.base, "base"), (1, args.port + 600, args.arm, "arm")],
            ))
    else:
        pass
    base_sim, base_wer = load(args.root / "c1n" / "base")
    arm_sim, arm_wer = load(args.root / "c1n" / "arm")
    keys = sorted(set(base_sim) & set(arm_sim))
    deltas = [arm_sim[key] - base_sim[key] for key in keys]
    rng = random.Random(0)
    means = sorted(statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(2000))
    print(
        f"c1n SIM over {len(keys)}: base {statistics.mean(base_sim[key] for key in keys):.2f} arm "
        f"{statistics.mean(arm_sim[key] for key in keys):.2f} paired {statistics.mean(deltas):+.3f} "
        f"(95 % {means[50]:+.3f} to {means[1949]:+.3f}); below 40: {sum(base_sim[key] < 40 for key in keys)} / "
        f"{sum(arm_sim[key] < 40 for key in keys)}",
        flush=True,
    )
    wer_keys = sorted(set(base_wer) & set(arm_wer))
    base_errors, words = word_counts(base_wer, wer_keys)
    arm_errors, _ = word_counts(arm_wer, wer_keys)
    print(
        f"c1n WER over {words} words: base {100 * base_errors / words:.2f} % ({base_errors}) arm "
        f"{100 * arm_errors / words:.2f} % ({arm_errors}); above 50 %: "
        f"{sum(float(base_wer[key]['wer']) > 0.5 for key in wer_keys)} / "
        f"{sum(float(arm_wer[key]['wer']) > 0.5 for key in wer_keys)}",
        flush=True,
    )
    for key in sorted(keys, key=lambda item: arm_sim[item] - base_sim[item])[:8]:
        print(f"worst SIM {key}: {base_sim[key]:.2f} to {arm_sim[key]:.2f}; WER {base_wer[key]['wer']} to {arm_wer[key]['wer']}", flush=True)


if __name__ == "__main__":
    main()
