"""Video-AMME in CI stage 9's shape, two arms at the same time, one a card, each pass on a fresh server.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/videoamme_pair.py --arms upl@2,c99@3 --passes c1,c16
    --out /data/pr2800/<run>
Each arm serves one replica of the CI's FP8 colocated worker (examples/configs/qwen3_omni_colocated_h100_fp8.yaml,
--colocate, 32768-token preprocessing and thinker) on its card and gets test_qwen3_omni_videoamme_ci.py's requests:
video at 2 fps, at most 128 frames of at most 401408 pixels, a spoken audio question, text out, greedy, 256 tokens,
videoamme-ci-50. A pass boots its own server, since a repeated prompt is a prefix-cache hit that computes differently.
--benchmark mmmu sends test_qwen3_omni_mmmu_ci.py's requests instead (mmmu-ci-50, image and text in, text out,
greedy, two warmup requests) and writes mmmu.json in place of videoamme.json.
The client runs in each tree's own code (this file in its --client mode). Writes <out>/<pass>/<arm>/videoamme.json,
the server's KV pool line in <out>/<pass>/<arm>/kv_pool.txt and <out>/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import snapshot_download

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

MODEL = "marksverdhei/Qwen3-Omni-30B-A3B-FP8"
CI_CONFIG = "examples/configs/qwen3_omni_colocated_h100_fp8.yaml"
SERVE_ARGS = [
    "--colocate", "--preprocessing.factory.max_seq_len", "32768", "--thinker.factory.max_seq_len", "32768",
]
PASSES = {"c1": 1, "c8": 8, "c16": 16}
PORT = 29000


def client(benchmark: str, port: int, concurrency: int, out: Path) -> None:
    from benchmarks.dataset.prepare import DATASETS

    if benchmark == "videoamme":
        from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
        from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig

        config = VideoEvalConfig(
            model="qwen3-omni", host="127.0.0.1", port=port, max_samples=50, max_concurrency=concurrency,
            output_dir=str(out), repo_id=DATASETS["videoamme-ci-50"], video_fps=2, video_max_frames=128,
            video_max_pixels=401408, disable_tqdm=True, timeout_s=500)
        results = asyncio.run(run_videoamme_eval(config, compute_wer=False))
    else:
        from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval

        # note (ratish): test_qwen3_omni_mmmu_ci.py's request: greedy, 2048 tokens, two warmup requests.
        config = MMMUEvalConfig(
            model="qwen3-omni", host="127.0.0.1", port=port, max_concurrency=concurrency, output_dir=str(out),
            repo_id=DATASETS["mmmu-ci-50"], warmup=2, disable_tqdm=True)
        results = asyncio.run(run_mmmu_eval(config))
    (out / f"{benchmark}.json").write_text(json.dumps(results, indent=1, default=str))


def run_pass(
    out: Path, pass_name: str, arm: str, card: int, config: str, model: str, benchmark: str
) -> dict[str, object]:
    arm_out = out / pass_name / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = PORT + 200 * card
    pair_check.log(out, f"{pass_name} {arm} card {card} serving")
    server = pair_check.serve(
        card, arm_tree, model, port, ["--config", config, *SERVE_ARGS], arm_out / "server.log")
    try:
        pair_check.log(out, f"{pass_name} {arm} card {card} healthy, client")
        rc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--client", "--benchmark", benchmark, "--port",
             str(port), "--concurrency", str(PASSES[pass_name]), "--out", str(arm_out)],
            cwd=arm_tree, env=pair_check.card_env(card, arm_tree), stdout=open(arm_out / "client.log", "w"),
            stderr=subprocess.STDOUT).returncode
    finally:
        pair_check.stop(card, server)
    kv_lines = [line.strip() for line in (arm_out / "server.log").read_text().splitlines()
                if re.search(r"max_total_num_tokens|KV Cache is allocated", line)]
    (arm_out / "kv_pool.txt").write_text("\n".join(kv_lines) + "\n")
    if rc != 0:
        pair_check.log(out, f"{pass_name} {arm} card {card} client rc {rc}")
        return dict(rc=rc)
    else:
        pass
    summary = json.loads((arm_out / f"{benchmark}.json").read_text())["summary"]
    pair_check.log(
        out, f"{pass_name} {arm} card {card} accuracy {summary.get('accuracy')} failed {summary.get('failed')}"
    )
    return dict(rc=rc, accuracy=summary.get("accuracy"), failed=summary.get("failed"))


def run_chain(
    out: Path, passes: list[str], arm: str, card: int, config: str, model: str, benchmark: str
) -> dict[str, object]:
    return {pass_name: run_pass(out, pass_name, arm, card, config, model, benchmark) for pass_name in passes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", help="arm@card,arm@card")
    parser.add_argument("--passes", default="c1,c16", help="passes in order, each on a fresh server")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--config", default=CI_CONFIG, help="the pipeline config, relative to the tree or absolute")
    parser.add_argument("--model", default=MODEL, help="the checkpoint, served with --config's matching precision")
    parser.add_argument("--benchmark", choices=("videoamme", "mmmu"), default="videoamme")
    args = parser.parse_args()
    if args.client:
        client(args.benchmark, args.port, args.concurrency, args.out)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot_download(args.model)
    pair_check.log(args.out, f"{args.model} ready")
    passes = args.passes.split(",")
    placements = [(arm, int(card)) for arm, card in (entry.split("@") for entry in args.arms.split(","))]
    with ThreadPoolExecutor(max_workers=len(placements)) as pool:
        records = list(
            pool.map(
                lambda placement: run_chain(
                    args.out, passes, *placement, args.config, args.model, args.benchmark
                ),
                placements,
            )
        )
    summary = {arm: record for (arm, _), record in zip(placements, records)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    pair_check.log(args.out, "done")


if __name__ == "__main__":
    main()
