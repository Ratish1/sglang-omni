"""Speaker similarity of serve_arm.py arms on their natural stops: per arm without similarity_results.json (its scorer ran
out of memory on a clip at the generation cap), the bench's generated.json copied to bench_sim/ with every clip longer
than --max-seconds marked unsuccessful, the scorer's similarity-only pass run from the arm's tree on that copy, and its
similarity_results.json placed in the arm's bench/ beside similarity_excluded.txt, which names the clips left out.

usage: python3 similarity_natural_stops.py ARMS_DIR ARM [ARM ...] --card 0 [--max-seconds 60] [--batch-size 8]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
# the scorer pads each similarity batch to its longest clip; fewer clips a batch keep a long natural stop in memory
SCORER = (
    "import runpy, sys; import benchmarks.tasks.tts as tts; tts.SPEAKER_SIMILARITY_BATCH_SIZE = int(sys.argv.pop(1)); "
    "runpy.run_module('benchmarks.eval.benchmark_tts_seedtts', run_name='__main__', alter_sys=True)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", type=Path)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    for label in args.labels:
        arm = args.arms / label
        if (arm / "bench" / "similarity_results.json").exists():
            print(f"{label}: already scored", flush=True)
            continue
        else:
            pass
        tree = (arm / "tree.txt").read_text().split()[0]
        rows = json.loads((arm / "bench" / "generated.json").read_text())
        excluded = sorted(row["sample_id"] for row in rows if row["audio_duration_s"] > args.max_seconds)
        for row in rows:
            if row["sample_id"] in excluded:
                row["is_success"] = False
                row["error"] = f"excluded: longer than {args.max_seconds} s, at the generation cap"
            else:
                pass
        sim_dir = arm / "bench_sim"
        sim_dir.mkdir(exist_ok=True)
        (sim_dir / "generated.json").write_text(json.dumps(rows))
        with open(sim_dir / "score--similarity-only.log", "w", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, "-c", SCORER, str(args.batch_size), "--model", MODEL, "--lang", "en",
                 "--skip-gpu-cleanup", "--output-dir", str(sim_dir), "--similarity-only", "--device",
                 f"cuda:{args.card}"],
                cwd=tree, env=dict(os.environ, PYTHONPATH=tree), stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        shutil.copy(sim_dir / "similarity_results.json", arm / "bench" / "similarity_results.json")
        (arm / "bench" / "similarity_excluded.txt").write_text("".join(f"{sample}\n" for sample in excluded))
        print(f"{label}: scored, excluded {excluded}", flush=True)


if __name__ == "__main__":
    main()
