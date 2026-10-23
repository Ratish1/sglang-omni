"""SocialOmni level 1 (2000 speaker-attribution items, video with its own audio track, use_audio_in_video), arms a card
each at the same time, each on a fresh text-only server.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/socialomni_pair.py --arms vrt@0,upv@1
    --out /data/pr2800/<run> [--hook-arms upvq] [--concurrency 16]
Each arm serves the benchmark README's text-only Qwen3-Omni server (bf16 Instruct, 65536-token preprocessing and
thinker) on its card, and benchmarks/eval/benchmark_omni_socialomni.py sends the full level 1 from the arm's own
tree. An arm in --hook-arms loads socialomni_fps/sitecustomize.py, which times its videos at the sampled fps rounded
to the nearest 2 / k (k whole), so positions stay whole while frames keep their real time. Writes
<out>/<arm>/socialomni-*.json, <out>/<arm>/client.log and <out>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
DATASET = "alexisty/SocialOmni"
DATASET_REVISION = "3b76009b45090eaa54007454c93a831f3cc8e1e6"
SERVE_ARGS = [
    "--text-only", "--model-name", "qwen3-omni", "--preprocessing.factory.max_seq_len", "65536",
    "--thinker.factory.max_seq_len", "65536",
]
PORT = 31000
HOOK_DIR = BOX / "socialomni_fps"


def run_arm(out: Path, arm: str, card: int, dataset_root: str, is_hooked: bool, concurrency: int) -> dict:
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = PORT + 200 * card
    env = pair_check.card_env(card, arm_tree)
    if is_hooked:
        env["PYTHONPATH"] = f"{HOOK_DIR}:{arm_tree}"
    else:
        pass
    pair_check.log(out, f"{arm} card {card} serving{' with the fps hook' if is_hooked else ''}")
    server = pair_check.serve(card, arm_tree, MODEL, port, SERVE_ARGS, arm_out / "server.log", env=env)
    try:
        pair_check.log(out, f"{arm} card {card} healthy, client")
        rc = subprocess.run(
            [sys.executable, "-m", "benchmarks.eval.benchmark_omni_socialomni", "--dataset-root", dataset_root,
             "--model", "qwen3-omni", "--base-url", f"http://127.0.0.1:{port}", "--level", "level1",
             "--max-concurrency", str(concurrency), "--disable-tqdm", "--output-dir", str(arm_out)],
            cwd=arm_tree, env=pair_check.card_env(card, arm_tree), stdout=open(arm_out / "client.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(card, server)
    results = sorted(arm_out.glob("socialomni-*.json"))
    if not results:
        pair_check.log(out, f"{arm} card {card} client rc {rc}, no results")
        return dict(rc=rc)
    else:
        pass
    level1 = json.loads(results[-1].read_text())["summary"]["level1"]
    pair_check.log(out, f"{arm} card {card} client rc {rc} level1 {json.dumps(level1)[:400]}")
    return dict(rc=rc, level1=level1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True, help="arm@card,arm@card")
    parser.add_argument("--hook-arms", default="", help="arms that load the fps hook")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset_root = snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REVISION,
                                     allow_patterns=["README.md", "data/level_1/*"])
    hooked = set(filter(None, args.hook_arms.split(",")))
    placements = [(arm, int(card)) for arm, card in (entry.split("@") for entry in args.arms.split(","))]
    with ThreadPoolExecutor(max_workers=len(placements)) as pool:
        records = list(pool.map(
            lambda placement: run_arm(args.out, placement[0], placement[1], dataset_root, placement[0] in hooked,
                                      args.concurrency),
            placements))
    (args.out / "summary.json").write_text(
        json.dumps({arm: record for (arm, _), record in zip(placements, records)}, indent=1))
    pair_check.log(args.out, "done")


if __name__ == "__main__":
    main()
