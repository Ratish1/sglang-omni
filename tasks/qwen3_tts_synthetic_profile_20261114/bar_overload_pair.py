"""Pipeshift's bar past the knee, two trees at once on two cards: per seed, both servers boot (each
pinned to its card's half of the socket), Nari's client (unmodified, pinned to the other socket) runs
every rate against both at once, and the cards swap for the next seed. Each rate writes the client's
own summary.json under OUT/seed<seed>/<arm>/rps<rate>.

usage: python3 bar_overload_pair.py --a-tree MAIN --b-tree BRANCH --out DIR [--rates 60 70 80] [--seeds 0 1 2]
"""

import argparse
import concurrent.futures
import os
import subprocess
import threading
import time
from pathlib import Path

from seedtts_pair import CARD_CPUS, CLIENT_CPUS, serve, stop
from served_baseline import MODELS, wait_healthy

CLIENT_DIR = Path("/data/tts_bench_src")
CLIENT_COMMIT = "a2d65e1"
DATASET = CLIENT_DIR / "tts_bench/datasets/seed-tts-eval.jsonl"
RATE_TIMEOUT_S = 1800
SETTLE_S = 30
MEMORY_SAMPLE_S = 5


def prepare_client() -> None:
    if DATASET.exists():
        return
    else:
        pass
    subprocess.run(["git", "clone", "-q", "https://github.com/nari-labs/benchmarks", str(CLIENT_DIR)], check=True)
    subprocess.run(["git", "-C", str(CLIENT_DIR), "checkout", "-q", CLIENT_COMMIT], check=True)
    subprocess.run(["uv", "sync", "-q", "--all-groups"], cwd=CLIENT_DIR / "tts_bench", check=True)
    subprocess.run(["uv", "run", "-q", "bench", "dataset", "prepare"], cwd=CLIENT_DIR / "tts_bench", check=True)


def sample_gpu_memory(out: Path, stop_event: threading.Event) -> None:
    """Every card's used memory every MEMORY_SAMPLE_S while a rate runs, one CSV row per sample."""
    with out.open("w") as handle:
        while not stop_event.is_set():
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            ).stdout.split()
            handle.write(f"{time.time():.0f},{','.join(used)}\n")
            handle.flush()
            stop_event.wait(MEMORY_SAMPLE_S)


def run_rate(port: int, model: str, rate: int, seed: int, out: Path, duration_s: int) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset", "-c", CLIENT_CPUS, "uv", "run", "-q", "bench", "run", "--target", "sglang-omni",
        "--base-url", f"http://127.0.0.1:{port}", "--model", model, "--voice", "Ryan", "--language", "English",
        "--dataset", "datasets/seed-tts-eval.jsonl", "--rps", str(rate), "--warmup", "30s", "--duration", f"{duration_s}s",
        "--arrival", "poisson", "--seed", str(seed), "--timeout", "120s", "--max-in-flight", "4096",
        "--output", str(out),
    ]
    with (out.parent / f"bench_rps{rate}.log").open("w") as log:
        return subprocess.run(
            command, cwd=CLIENT_DIR / "tts_bench", stdout=log, stderr=subprocess.STDOUT, timeout=RATE_TIMEOUT_S,
            check=False,
        ).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rates", type=int, nargs="+", default=[60, 70, 80])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--port", type=int, default=22000)
    parser.add_argument("--duration", type=int, default=120, help="measured seconds per rate")
    args = parser.parse_args()
    prepare_client()
    model, _ = MODELS["customvoice"]
    trees = {"a": args.a_tree, "b": args.b_tree}
    ports = {"a": args.port, "b": args.port + 200}
    for index, seed in enumerate(args.seeds):
        cards = {"a": 0, "b": 1} if index % 2 == 0 else {"a": 1, "b": 0}
        seed_out = args.out / f"seed{seed}"
        servers = {}
        try:
            for arm in ("a", "b"):
                (seed_out / arm).mkdir(parents=True, exist_ok=True)
                identity = subprocess.check_output(["git", "rev-parse", "HEAD", "HEAD:sglang_omni"], cwd=trees[arm], text=True)
                (seed_out / arm / "tree.txt").write_text(f"{trees[arm]} card {cards[arm]} cpus {CARD_CPUS[cards[arm]]}\n{identity}")
                servers[arm] = serve(trees[arm], model, cards[arm], ports[arm], seed_out / arm / "serve.log", {})
            for arm in ("a", "b"):
                wait_healthy(ports[arm], servers[arm])
            print(f"{time.strftime('%T')} seed {seed} healthy, a card {cards['a']}, b card {cards['b']}", flush=True)
            for rate in args.rates:
                stop_event = threading.Event()
                (seed_out / f"cards_rps{rate}.txt").write_text(f"a card {cards['a']}, b card {cards['b']}\n")
                sampler = threading.Thread(
                    target=sample_gpu_memory, args=(seed_out / f"gpu_memory_rps{rate}.csv", stop_event), daemon=True
                )
                sampler.start()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {
                        arm: pool.submit(run_rate, ports[arm], model, rate, seed, seed_out / arm / f"rps{rate}", args.duration)
                        for arm in ("a", "b")
                    }
                    codes = {arm: future.result() for arm, future in futures.items()}
                stop_event.set()
                sampler.join()
                print(f"{time.strftime('%T')} seed {seed} rps {rate} rc {codes}", flush=True)
                time.sleep(SETTLE_S)
        finally:
            for server in servers.values():
                stop(server)
    print(f"{time.strftime('%T')} bar_overload_pair done", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("UV_NO_PROGRESS", "1")
    main()
