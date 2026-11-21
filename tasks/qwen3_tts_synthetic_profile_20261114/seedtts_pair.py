"""Same-time SeedTTS EN pairs of two trees on two cards, then the order swapped: each order boots
both arms at once (one per card), runs every cell on both at once over the full 1,088 prompts,
unseeded, then scores the scored cell's WER (a Qwen3-ASR server on the arm's own card) and speaker
similarity, one arm after the other. Results under OUT/order{0,1}/{a,b}/<cell>.

usage: python3 seedtts_pair.py --a-tree MAIN --b-tree BRANCH --out DIR [--model base|customvoice]
"""

import argparse
import concurrent.futures
import functools
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

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
# cells only --rounds runs, where a host-side change acts but the standard pair does not look
ROUNDS_ONLY_CELLS = (("whole_c64", 64, False),)
def node_ranges(node: int) -> list[tuple[int, int]]:
    """The NUMA node's CPU ranges, physical cores then their hyperthread siblings."""
    text = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    return [(int(part.split("-")[0]), int(part.split("-")[-1])) for part in text.split(",")]


def format_ranges(ranges: list[tuple[int, int]]) -> str:
    return ",".join(f"{low}-{high}" for low, high in ranges)


@functools.cache
def cpu_layout() -> tuple[dict[int, str], str]:
    """Each visible card's server takes a half of its own NUMA node; the client takes the nodes with no card, or the halves left."""
    bus_ids = subprocess.run(
        ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], capture_output=True, text=True, check=True
    ).stdout.split()
    card_nodes = {}
    for card, bus_id in enumerate(bus_ids):
        domain, rest = bus_id.split(":", 1)
        card_nodes[card] = int(Path(f"/sys/bus/pci/devices/{domain[-4:].lower()}:{rest.lower()}/numa_node").read_text())
    nodes = sorted(int(path.name[4:]) for path in Path("/sys/devices/system/node").glob("node[0-9]*"))
    card_cpus = {}
    leftover = []
    for node in nodes:
        ranges = node_ranges(node)
        halves = [
            [(low, low + (high - low + 1) // 2 - 1) for low, high in ranges],
            [(low + (high - low + 1) // 2, high) for low, high in ranges],
        ]
        node_cards = [card for card, card_node in card_nodes.items() if card_node == node]
        for card, half in zip(node_cards, halves):
            card_cpus[card] = format_ranges(half)
        leftover.append((node, node_cards, halves[len(node_cards):] if len(node_cards) < 2 else []))
    free_nodes = [format_ranges(node_ranges(node)) for node, node_cards, _ in leftover if not node_cards]
    if free_nodes:
        client = ",".join(free_nodes)
    else:
        client = ",".join(format_ranges(half) for _, _, halves in leftover for half in halves)
    return card_cpus, client


def card_cpus(card: int) -> str:
    return cpu_layout()[0][card]


def client_cpus() -> str:
    return cpu_layout()[1]


def serve(
    tree: Path, model: str, card: int, port: int, log: Path, env_extra: dict[str, str],
    server_args: tuple[str, ...] = (),
) -> subprocess.Popen:
    command = [
        "taskset", "-c", card_cpus(card),
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
        "taskset", "-c", client_cpus(),
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
            ["taskset", "-c", card_cpus(card), sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path",
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
    order: int, arms: tuple[str, ...], trees: dict[str, Path], cards: dict[str, int], model: str,
    extras: dict[str, list[str]], port: int, out: Path, cells: tuple[tuple[str, int, bool], ...], should_score: bool,
) -> None:
    ports = {"a": port, "b": port + 200}
    servers = {}
    try:
        for arm in arms:
            arm_out = out / f"order{order}" / arm
            arm_out.mkdir(parents=True, exist_ok=True)
            identity = subprocess.check_output(["git", "rev-parse", "HEAD", "HEAD:sglang_omni"], cwd=trees[arm], text=True)
            (arm_out / "tree.txt").write_text(f"{trees[arm]} card {cards[arm]} client {' '.join(extras[arm])}\n{identity}")
            servers[arm] = serve(trees[arm], model, cards[arm], ports[arm], arm_out / "serve.log", {})
        for arm in arms:
            wait_healthy(ports[arm], servers[arm])
        print(f"{time.strftime('%T')} order {order} healthy, cards {cards}", flush=True)
        for cell, concurrency, stream in cells:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(arms)) as pool:
                futures = {
                    arm: pool.submit(bench, trees[arm], model, extras[arm], ports[arm], concurrency, stream,
                                     out / f"order{order}" / arm / cell)
                    for arm in arms
                }
                summaries = {arm: future.result() for arm, future in futures.items()}
            for arm, summary in summaries.items():
                (out / f"order{order}" / arm / f"{cell}.json").write_text(json.dumps(summary, indent=1))
            print(f"{time.strftime('%T')} order {order} {cell} done", flush=True)
        for arm in arms:
            # note (ratish): the vocoder's codec counters after every cell, window and tail replays among them.
            try:
                info = httpx.get(f"http://127.0.0.1:{ports[arm]}/model_info", timeout=30).json()
            except httpx.HTTPError as error:
                print(f"{time.strftime('%T')} order {order} {arm} model_info failed: {error}", flush=True)
            else:
                (out / f"order{order}" / arm / "model_info.json").write_text(json.dumps(info, indent=1))
    finally:
        for server in servers.values():
            stop(server)
    if should_score:
        for index, arm in enumerate(arms):
            score(
                trees[arm], model, extras[arm], cards[arm], port + 400 + 400 * index, out / f"order{order}" / arm / SCORED_CELL
            )
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
    parser.add_argument(
        "--rounds-cell", choices=[cell for cell, _, _ in CELLS + ROUNDS_ONLY_CELLS], default="stream_c16"
    )
    parser.add_argument("--a-client-arg", action="append", default=[], help="one more client argument for arm a")
    parser.add_argument("--b-client-arg", action="append", default=[], help="one more client argument for arm b")
    parser.add_argument(
        "--one-card", action="store_true",
        help="run each arm alone on --card, a then b in order 0 and b then a in order 1, when one card is held",
    )
    parser.add_argument("--card", type=int, default=0, help="the card --one-card runs on")
    parser.add_argument(
        "--cells", nargs="+", choices=[cell for cell, _, _ in CELLS], default=None,
        help="only these cells, in the protocol's order",
    )
    args = parser.parse_args()
    model, extra = MODELS[args.model]
    trees = {"a": args.a_tree, "b": args.b_tree}
    extras = {"a": [*extra, *args.a_client_arg], "b": [*extra, *args.b_client_arg]}
    if args.rounds > 0:
        _, concurrency, stream = next(entry for entry in CELLS + ROUNDS_ONLY_CELLS if entry[0] == args.rounds_cell)
        cells = tuple((f"{args.rounds_cell}_r{index + 1}", concurrency, stream) for index in range(args.rounds))
    else:
        cells = tuple(entry for entry in CELLS if args.cells is None or entry[0] in args.cells)
    for order, cards in enumerate(({"a": 0, "b": 1}, {"a": 1, "b": 0})):
        if args.one_card:
            for arm in ("a", "b") if order == 0 else ("b", "a"):
                run_order(
                    order, (arm,), trees, {arm: args.card}, model, extras, args.port, args.out, cells, args.rounds == 0
                )
        else:
            run_order(order, ("a", "b"), trees, cards, model, extras, args.port, args.out, cells, args.rounds == 0)
    print(f"{time.strftime('%T')} seedtts_pair done", flush=True)


if __name__ == "__main__":
    main()
