"""Speaker similarity for a seeded pair whose scorer ran out of memory on a capped clip: per arm, the bench's
generated.json copied to bench_sim/ with every clip longer than --max-seconds in either arm marked unsuccessful, so
the scorer skips it in both, then the scorer's similarity-only pass on that copy and its similarity_results.json
placed in the arm's bench/ beside similarity_excluded.txt, which names the clips left out.

usage: PAIR_ROOT=<pairs> python3 similarity_excluding_caps.py <label> --tree <tree> --card 2 [--max-seconds 60]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--tree", required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    args = parser.parse_args()
    root = Path(os.environ["PAIR_ROOT"]) / args.label
    generated = {arm: json.loads((root / arm / "bench" / "generated.json").read_text()) for arm in ("a", "b")}
    excluded = sorted(
        {row["sample_id"] for rows in generated.values() for row in rows if row["audio_duration_s"] > args.max_seconds}
    )
    print(f"{args.label}: excluding {excluded}", flush=True)
    for arm, rows in generated.items():
        sim_dir = root / arm / "bench_sim"
        sim_dir.mkdir(exist_ok=True)
        for row in rows:
            if row["sample_id"] in excluded:
                row["is_success"] = False
                row["error"] = f"excluded: longer than {args.max_seconds} s in an arm of the pair"
            else:
                pass
        (sim_dir / "generated.json").write_text(json.dumps(rows))
        with open(sim_dir / "score--similarity-only.log", "w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL, "--lang", "en",
                 "--skip-gpu-cleanup", "--output-dir", str(sim_dir), "--similarity-only", "--device",
                 f"cuda:{args.card}"],
                cwd=args.tree, env=dict(os.environ, PYTHONPATH=args.tree), stdout=log, stderr=subprocess.STDOUT,
                check=True,
            )
        shutil.copy(sim_dir / "similarity_results.json", root / arm / "bench" / "similarity_results.json")
        (root / arm / "bench" / "similarity_excluded.txt").write_text("\n".join(excluded) + "\n")
        print(f"{args.label} {arm}: scored", flush=True)


if __name__ == "__main__":
    main()
