"""Byte identity of two trees' served audio: one server pair (main on card 0, the branch on card 1), SeedTTS EN 1.7B
Base at concurrency 1 with seed 1234, non-streaming then streaming, then every request's WAV compared by SHA-256.
Each prompt draws the same random numbers in both arms, and concurrency 1 keeps batch composition fixed, so any
differing file is a change in the served audio.

usage: python3 seeded_identity.py --main TREE --branch TREE --root /data/next_20261101/<block> [--port 24000]
"""

import argparse
from pathlib import Path

from seeded_tail import audio_hashes, serve_phase

RUNS = (("c1n", 1, 1234, False), ("c1s", 1, 1234, True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=24000)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    serve_phase("o1", [("main", args.main, 0, args.port), ("branch", args.branch, 1, args.port + 200)], list(RUNS),
                args.root)
    for run, _, _, _ in RUNS:
        first, second = audio_hashes(args.root / run / "main"), audio_hashes(args.root / run / "branch")
        same = sum(first[name] == second.get(name) for name in first)
        differing = sorted(name for name in first if first[name] != second.get(name))
        print(f"{run} seed 1234: byte identical {same} of {len(first)} ({len(second)} in the branch); differing "
              f"{differing[:10]}", flush=True)


if __name__ == "__main__":
    main()
