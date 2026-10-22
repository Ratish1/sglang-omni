"""Same-time pair of two trees serving Qwen3-Omni's colocated bf16 speech pipeline, one card each, then scoring.

usage: python3 omni_pair.py --label p6omni_c16_a --tree /workspace/main_4f4cf02cc --b-tree /workspace/main_dc8a48e11
       --a-card 2 --b-card 3 --port 11000

Each arm puts every GPU stage on its card with the stage gpu flags (no device env) and serves the H100 bf16 colocated
config as the Qwen3-Omni program's BENCHMARKS.md 2 does. Both boot and bench at once: Omni SeedTTS EN, full corpus, voice
clone, streaming, warmup 1, max concurrency 16, unseeded. With the servers stopped, each arm is scored in turn on its
card: WER from a Qwen3-ASR server (pair.py's scorer ports and rendezvous port), then speaker similarity.
Results under PAIR_ROOT/<label>/{a,b}.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from pair import ASR_MODEL, BENCH_TIMEOUT_S, ROOT, gpu_snapshot, stop_group, wait_healthy

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SERVE_ARGS = [
    "--config", "examples/configs/qwen3_omni_colocated_h100_bf16.yaml", "--colocate",
    "--preprocessing.factory.max_seq_len", "32768", "--thinker.factory.max_seq_len", "32768",
]
GPU_STAGES = ("image_encoder", "audio_encoder", "thinker", "talker_ar", "code2wav")
SEEDTTS = [sys.executable, "-m", "benchmarks.eval.benchmark_omni_seedtts", "--model", "qwen3-omni", "--lang", "en",
           "--disable-tqdm"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--b-tree", required=True)
    parser.add_argument("--a-card", type=int, required=True)
    parser.add_argument("--b-card", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()
    arms = {"a": (args.a_card, args.tree, args.port), "b": (args.b_card, args.b_tree, args.port + 200)}
    outs = {arm: ROOT / args.label / arm for arm in arms}
    servers = {}
    spawned = {}
    for arm, (card, tree, port) in arms.items():
        outs[arm].mkdir(parents=True, exist_ok=True)
        identity = subprocess.run(
            "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo WORKING_TREE_DIRTY",
            shell=True, cwd=tree, capture_output=True, text=True,
        ).stdout
        placement = [flag for stage in GPU_STAGES for flag in (f"--{stage}.gpu", str(card))]
        server_env = dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1")
        (outs[arm] / "tree.txt").write_text(
            f"{tree} card {card} serve args: {' '.join(SERVE_ARGS + placement)}\n{identity}", encoding="utf-8"
        )
        command = [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, *SERVE_ARGS, *placement,
                   "--host", "127.0.0.1", "--port", str(port)]
        with open(outs[arm] / "serve.log", "w", encoding="utf-8") as serve_log:
            spawned[arm] = time.monotonic()
            servers[arm] = subprocess.Popen(
                command, cwd=tree, env=server_env, stdout=serve_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
    try:
        for arm, (_, _, port) in arms.items():
            wait_healthy(port, servers[arm])
            (outs[arm] / "startup.txt").write_text(f"{time.monotonic() - spawned[arm]:.1f}\n", encoding="utf-8")
            print(f"{arm} healthy {time.strftime('%H:%M:%S')}", flush=True)
        for arm in arms:
            (outs[arm] / "gpu_healthy.txt").write_text(gpu_snapshot(), encoding="utf-8")
        benches = {}
        for arm, (_, tree, port) in arms.items():
            command = [*SEEDTTS, "--generate-only", "--voice-clone", "--stream", "--warmup", "1", "--max-concurrency",
                       str(args.concurrency), "--host", "127.0.0.1", "--port", str(port), "--output-dir",
                       str(outs[arm] / "bench")]
            with open(outs[arm] / "bench.log", "w", encoding="utf-8") as bench_log:
                benches[arm] = subprocess.Popen(
                    command, cwd=tree, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=tree),
                    stdout=bench_log, stderr=subprocess.STDOUT,
                )
        for arm, bench in benches.items():
            print(f"{arm} bench rc {bench.wait(timeout=BENCH_TIMEOUT_S)}", flush=True)
        for arm in arms:
            (outs[arm] / "gpu_after_bench.txt").write_text(gpu_snapshot(), encoding="utf-8")
    finally:
        for arm, server in servers.items():
            stop_group(server, arms[arm][2])
    for index, (arm, (card, tree, _)) in enumerate(arms.items()):
        score(outs[arm], card, args.port + 400 + 400 * index, tree)
    print("scored", flush=True)
    print("pair done", flush=True)


def score(out: Path, card: int, port: int, tree: str) -> None:
    """WER from a Qwen3-ASR server on the arm's card with its own rendezvous port, then speaker similarity there."""
    env = dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1")
    asr_env = dict(env, MASTER_PORT=str(port + 50))
    with open(out / "asr_serve.log", "w", encoding="utf-8") as serve_log:
        asr = subprocess.Popen(
            [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", ASR_MODEL, "--host", "127.0.0.1",
             "--port", str(port), "--asr.gpu", str(card)],
            cwd=tree, env=asr_env, stdout=serve_log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        wait_healthy(port, asr)
        with open(out / "score--transcribe-only.log", "w", encoding="utf-8") as score_log:
            subprocess.run(
                [*SEEDTTS, "--transcribe-only", "--host", "127.0.0.1", "--port", str(port), "--device", f"cuda:{card}",
                 "--output-dir", str(out / "bench")],
                cwd=tree, env=env, stdout=score_log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S, check=False,
            )
    finally:
        stop_group(asr, port)
    with open(out / "score--similarity-only.log", "w", encoding="utf-8") as score_log:
        subprocess.run(
            [*SEEDTTS, "--similarity-only", "--device", f"cuda:{card}", "--output-dir", str(out / "bench")],
            cwd=tree, env=env, stdout=score_log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S, check=False,
        )


if __name__ == "__main__":
    main()
