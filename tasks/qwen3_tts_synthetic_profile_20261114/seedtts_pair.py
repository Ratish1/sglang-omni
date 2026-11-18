"""Same-time SeedTTS EN pairs of two trees on two cards, then the order swapped: each order boots
both arms at once (one per card), runs every cell on both at once over the full 1,088 prompts,
unseeded, then scores the scored cell's WER (a Qwen3-ASR server on the arm's own card) and speaker
similarity, one arm after the other. Results under OUT/order{0,1}/{a,b}/<cell>.

usage: python3 seedtts_pair.py --a-tree MAIN --b-tree BRANCH --out DIR [--model base|customvoice]
"""

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from served_baseline import MODELS, wait_healthy

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
ASR_CONCURRENCY = 64
CELL_TIMEOUT_S = 3600
STOP_TIMEOUT_S = 300
# (cell, concurrency, stream); the protocol's non-streaming c1 and c16, and streaming where first audio is read
CELLS = (
    ("stream_c16", 16, True),
    ("stream_c1", 1, True),
    ("stream_c64", 64, True),
    ("whole_c16", 16, False),
    ("whole_c1", 1, False),
)
SCORED_CELL = "stream_c16"
# note (ratish): both H100s of host-221 sit on NUMA node 1, so each server takes half of that socket and
# the clients take node 0.
CARD_CPUS = {0: "32-47,96-111", 1: "48-63,112-127"}
CLIENT_CPUS = "0-31,64-95"


def serve(
    tree: Path, model: str, card: int, port: int, log: Path, env_extra: dict[str, str],
    server_args: tuple[str, ...] = (),
) -> subprocess.Popen:
    command = [
        "taskset", "-c", CARD_CPUS[card],
        sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", model, "--host", "127.0.0.1",
        "--port", str(port), "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card), *server_args,
    ]
    with log.open("w") as handle:
        return subprocess.Popen(
            command, cwd=tree, env={**os.environ, "PYTHONPATH": str(tree), "SGLANG_OMNI_STRICT_PORT": "1", **env_extra},
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )


def stop(server: subprocess.Popen) -> None:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
    else:
        pass
    try:
        server.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=30)


def bench(tree: Path, model: str, extra: list[str], port: int, concurrency: int, stream: bool, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset", "-c", CLIENT_CPUS,
        sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--meta", "zhaochenyang20/seed-tts-eval-arrow",
        "--model", model, "--use-existing-server", "--host", "127.0.0.1", "--port", str(port), "--generate-only",
        "--lang", "en", "--warmup", str(concurrency), "--max-concurrency", str(concurrency),
        "--output-dir", str(out / "bench"), *extra, *(["--stream"] if stream else []),
    ]
    with (out / "bench.log").open("w") as log:
        subprocess.run(
            command, cwd=tree, env={**os.environ, "PYTHONPATH": str(tree), "CUDA_VISIBLE_DEVICES": ""},
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=CELL_TIMEOUT_S,
        )
    return json.loads((out / "bench/speed_results.json").read_text())["summary"]


def score(tree: Path, model: str, extra: list[str], card: int, port: int, out: Path) -> None:
    """WER from an ASR server on the arm's own card, then speaker similarity on that card."""
    env = {**os.environ, "PYTHONPATH": str(tree), "SGLANG_OMNI_STRICT_PORT": "1"}
    scorer = [
        sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--meta", "zhaochenyang20/seed-tts-eval-arrow",
        "--model", model, "--lang", "en", "--skip-gpu-cleanup", "--output-dir", str(out / "bench"), *extra,
    ]
    with (out / "asr_serve.log").open("w") as log:
        asr = subprocess.Popen(
            ["taskset", "-c", CARD_CPUS[card], sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path",
             ASR_MODEL, "--host", "127.0.0.1", "--port", str(port), "--asr.gpu", str(card)],
            cwd=tree, env={**env, "MASTER_PORT": str(port + 50)}, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_healthy(port, asr)
        with (out / "score_transcribe.log").open("w") as log:
            subprocess.run(
                [*scorer, "--transcribe-only", "--use-existing-server", "--host", "127.0.0.1", "--port", str(port),
                 "--asr-concurrency", str(ASR_CONCURRENCY)],
                cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=CELL_TIMEOUT_S, check=False,
            )
    finally:
        stop(asr)
    if "--no-ref-audio" in extra:
        return
    else:
        pass
    with (out / "score_similarity.log").open("w") as log:
        subprocess.run(
            [*scorer, "--similarity-only", "--device", f"cuda:{card}"],
            cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=CELL_TIMEOUT_S, check=False,
        )


def run_order(
    order: int, trees: dict[str, Path], cards: dict[str, int], model: str, extra: list[str], port: int, out: Path,
    cells: tuple[tuple[str, int, bool], ...], should_score: bool,
) -> None:
    ports = {"a": port, "b": port + 200}
    servers = {}
    try:
        for arm in ("a", "b"):
            arm_out = out / f"order{order}" / arm
            arm_out.mkdir(parents=True, exist_ok=True)
            identity = subprocess.check_output(["git", "rev-parse", "HEAD", "HEAD:sglang_omni"], cwd=trees[arm], text=True)
            (arm_out / "tree.txt").write_text(f"{trees[arm]} card {cards[arm]}\n{identity}")
            servers[arm] = serve(trees[arm], model, cards[arm], ports[arm], arm_out / "serve.log", {})
        for arm in ("a", "b"):
            wait_healthy(ports[arm], servers[arm])
        print(f"{time.strftime('%T')} order {order} healthy, a card {cards['a']}, b card {cards['b']}", flush=True)
        for cell, concurrency, stream in cells:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    arm: pool.submit(bench, trees[arm], model, extra, ports[arm], concurrency, stream,
                                     out / f"order{order}" / arm / cell)
                    for arm in ("a", "b")
                }
                summaries = {arm: future.result() for arm, future in futures.items()}
            for arm, summary in summaries.items():
                (out / f"order{order}" / arm / f"{cell}.json").write_text(json.dumps(summary, indent=1))
            print(f"{time.strftime('%T')} order {order} {cell} done", flush=True)
    finally:
        for server in servers.values():
            stop(server)
    if should_score:
        for index, arm in enumerate(("a", "b")):
            score(trees[arm], model, extra, cards[arm], port + 400 + 400 * index, out / f"order{order}" / arm / SCORED_CELL)
        print(f"{time.strftime('%T')} order {order} scored", flush=True)
    else:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="base")
    parser.add_argument("--port", type=int, default=21000)
    parser.add_argument(
        "--rounds", type=int, default=0,
        help="run only --rounds-cell, this many rounds on the same servers, unscored",
    )
    parser.add_argument("--rounds-cell", choices=[cell for cell, _, _ in CELLS], default="stream_c16")
    args = parser.parse_args()
    model, extra = MODELS[args.model]
    trees = {"a": args.a_tree, "b": args.b_tree}
    if args.rounds > 0:
        _, concurrency, stream = next(entry for entry in CELLS if entry[0] == args.rounds_cell)
        cells = tuple((f"{args.rounds_cell}_r{index + 1}", concurrency, stream) for index in range(args.rounds))
    else:
        cells = CELLS
    for order, cards in enumerate(({"a": 0, "b": 1}, {"a": 1, "b": 0})):
        run_order(order, trees, cards, model, extra, args.port, args.out, cells, args.rounds == 0)
    print(f"{time.strftime('%T')} seedtts_pair done", flush=True)


if __name__ == "__main__":
    main()
