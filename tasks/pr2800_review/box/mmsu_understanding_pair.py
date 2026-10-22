"""MMSU understanding in the CI's request shape on the full mmsu-ci-2000, two arms at the same time, one a card.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/mmsu_understanding_pair.py --arms upl@0,abr@1
    --out /data/pr2800/<run>
Each arm serves the CI's thinker-only config (examples/configs/qwen3_omni_mmmu_h100.yaml, test_qwen3_omni_mmsu_ci.py's
server) on its card and gets the CI's requests: audio and text in, text out, greedy, 32 tokens, c16, every
mmsu-ci-2000 question. The client runs in each tree's own code (this file in its --client mode). Writes
<out>/<arm>/mmsu.json and <out>/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

REPO_ID = "zhaochenyang20/mmsu-ci-2000"
CONFIG = "examples/configs/qwen3_omni_mmmu_h100.yaml"
MAX_TOKENS = 32
CONCURRENCY = 16
PORT = 25000


def client(port: int, out: Path) -> None:
    from benchmarks.eval.benchmark_omni_mmsu import run

    args = argparse.Namespace(
        base_url=None, host="127.0.0.1", port=port, model="qwen3-omni", modalities="text", output_dir=str(out),
        max_samples=None, task_names=None, categories=None, prompt=None, max_tokens=MAX_TOKENS, temperature=0.0,
        warmup=0, max_concurrency=CONCURRENCY, request_rate=float("inf"), timeout_s=300, save_audio=False,
        disable_tqdm=True, seed=None, fingerprint=False, repo_id=REPO_ID, lang="en", asr_device="cuda:0")
    results = asyncio.run(run(args))
    (out / "mmsu.json").write_text(json.dumps(results, indent=1))


def run_arm(out: Path, arm: str, card: int) -> dict[str, object]:
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = PORT + 200 * card
    server = pair_check.serve(card, arm_tree, pair_check.MODEL, port, ["--config", CONFIG], arm_out / "server.log")
    try:
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--client", "--port", str(port), "--out", str(arm_out)],
            cwd=arm_tree, env=pair_check.card_env(card, arm_tree), stdout=open(arm_out / "client.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(card, server)
    pair_check.log(out, f"{arm} card {card} client rc {rc}")
    if rc != 0:
        return dict(rc=rc)
    else:
        pass
    mmsu = json.loads((arm_out / "mmsu.json").read_text())
    pair_check.log(out, f"{arm} accuracy {mmsu['accuracy'].get('overall_accuracy')}")
    return dict(rc=rc, accuracy=mmsu["accuracy"], speed=mmsu["speed"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", help="arm@card,arm@card")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.client:
        client(args.port, args.out)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    placements = [(arm, int(card)) for arm, card in (entry.split("@") for entry in args.arms.split(","))]
    with ThreadPoolExecutor(max_workers=len(placements)) as pool:
        records = list(pool.map(lambda placement: run_arm(args.out, *placement), placements))
    summary = {arm: record for (arm, _), record in zip(placements, records)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    pair_check.log(args.out, "done")


if __name__ == "__main__":
    main()
