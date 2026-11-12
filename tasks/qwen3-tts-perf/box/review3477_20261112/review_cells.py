"""Review cells for #3477 (whole requests on the codec's window graphs): two trees at the same time, one server per card.

Each phase boots a server pair on SeedTTS EN 1.7B Base and runs its cells in turn on both servers at once, every cell over
all 1,088 prompts, warmup 1, unseeded unless a seed is named. Phases:
  prime    main alone on card 0 with no cells, so every later boot reads one compile cache
  o1       main card 0, head card 1
  o2       head card 0, main card 1
  band     main on both cards, the same-time band for the first chunk rows
  nograph  main card 0, head card 1, both with an incremental graph headroom no card has, so no codec graph captures
  nograph2 the same, cards swapped
  slots    head with 16 codec state slots card 0, head with its 64 card 1
  slots2   the same, cards swapped
Cells: n16 and n64 non-streaming; mix8 a streaming and a non-streaming client at once, 8 in flight each; mix16_48 the same
with 16 streaming and 48 non-streaming in flight; s64a to s64c streaming at 64 three times; n1 non-streaming one at a time,
seed 1234. In a mixed cell the non-streaming client repeats until the streaming client ends, so every stream runs under
non-streaming load; the non-streaming numbers are its first pass.

usage: python3 review_cells.py --main TREE --head TREE --root DIR --phases o1,o2,band --card-cpus "0=0-15,64-79;1=16-31,80-95"
       --client-cpus 32-63,96-127 [--port 20000]
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3_20261020"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from pair import BENCH_TIMEOUT_S, MODEL, stop_group, wait_healthy  # noqa: E402
from seeded_tail import identity  # noqa: E402

NO_GRAPH = ["--vocoder.factory.incremental_codec_cuda_graph_min_free_gb", "4096"]
SLOTS_16 = ["--vocoder.factory.codec_state_slots", "16"]
STREAM_64 = [(f"s64{letter}", [("stream", 64, None, True)]) for letter in "abc"]
MIXED = [
    ("mix8", [("stream", 8, None, True), ("whole", 8, None, False)]),
    ("mix16_48", [("stream", 16, None, True), ("whole", 48, None, False)]),
]
WHOLE = [("n16", [("whole", 16, None, False)]), ("n64", [("whole", 64, None, False)])]
WHOLE_1 = [("n1", [("whole", 1, 1234, False)])]
PHASES = {
    "prime": ((("prime", "main", 0, []),), []),
    "o1": ((("main", "main", 0, []), ("head", "head", 1, [])), WHOLE + MIXED + STREAM_64 + WHOLE_1),
    "o2": ((("head", "head", 0, []), ("main", "main", 1, [])), WHOLE + MIXED + STREAM_64),
    "band": ((("maina", "main", 0, []), ("mainb", "main", 1, [])), MIXED + STREAM_64),
    "nograph": ((("main", "main", 0, NO_GRAPH), ("head", "head", 1, NO_GRAPH)), WHOLE),
    "nograph2": ((("head", "head", 0, NO_GRAPH), ("main", "main", 1, NO_GRAPH)), WHOLE),
    "slots": ((("head16", "head", 0, SLOTS_16), ("head64", "head", 1, [])), WHOLE + [("n64b", [("whole", 64, None, False)])]),
    "slots2": ((("head64", "head", 0, []), ("head16", "head", 1, SLOTS_16)), WHOLE + [("n64b", [("whole", 64, None, False)])]),
}


def start(tree: str, card: int, port: int, cpus: str, extra: list[str], log: Path) -> subprocess.Popen:
    command = ["taskset", "-c", cpus, sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, "--host",
               "127.0.0.1", "--port", str(port), "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card), *extra]
    with open(log, "w", encoding="utf-8") as handle:
        return subprocess.Popen(command, cwd=tree, env=dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1"),
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def bench(tree: str, port: int, out: Path, concurrency: int, seed: int | None, stream: bool, client_cpus: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    command = ["taskset", "-c", client_cpus, sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL,
               "--use-existing-server", "--host", "127.0.0.1", "--port", str(port), "--generate-only",
               *(["--stream"] if stream else []), "--lang", "en", "--warmup", "1", "--max-concurrency", str(concurrency),
               *([] if seed is None else ["--seed", str(seed)]), "--output-dir", str(out / "bench")]
    with open(out / "bench.log", "w", encoding="utf-8") as log:
        subprocess.run(command, cwd=tree, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=tree), stdout=log,
                       stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S, check=False)
    subprocess.run(["rm", "-rf", str(out / "bench" / "audio")], check=False)


def run_arm_cell(tree: str, port: int, out: Path, jobs: list[tuple[str, int, int | None, bool]], client_cpus: str) -> None:
    """One arm's jobs at once; beside a streaming job, a non-streaming job repeats until the streaming job ends."""
    stream_done = threading.Event()
    is_mixed = len(jobs) > 1

    def run_job(job: tuple[str, int, int | None, bool]) -> None:
        name, concurrency, seed, stream = job
        if stream or not is_mixed:
            bench(tree, port, out / name, concurrency, seed, stream, client_cpus)
            if stream:
                stream_done.set()
            else:
                pass
        else:
            passes = 0
            while passes == 0 or not stream_done.is_set():
                passes += 1
                bench(tree, port, out / (name if passes == 1 else f"{name}_pass{passes}"), concurrency, seed, stream,
                      client_cpus)

    with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
        list(pool.map(run_job, jobs))


def line(out: Path) -> str:
    path = out / "bench" / "speed_results.json"
    if not path.exists():
        return "no speed_results.json"
    else:
        pass
    summary = json.loads(path.read_text())["summary"]
    wanted = ("completed_requests", "total_requests", "throughput_qps", "rtf_mean", "latency_mean_s", "latency_median_s",
              "latency_p95_s", "audio_ttfp_mean_s", "audio_ttfp_median_s", "audio_ttfp_p95_s", "audio_ttfp_p99_s")
    return " ".join(f"{key}={summary[key]:.4f}" if isinstance(summary[key], float) else f"{key}={summary[key]}"
                    for key in wanted if key in summary and summary[key] is not None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phases", required=True)
    parser.add_argument("--card-cpus", required=True)
    parser.add_argument("--client-cpus", required=True)
    parser.add_argument("--port", type=int, default=20000)
    args = parser.parse_args()
    trees = {"main": args.main, "head": args.head}
    card_cpus = {int(card): cpus for card, cpus in (item.split("=") for item in args.card_cpus.split(";"))}
    port = args.port
    for phase in args.phases.split(","):
        arms, cells = PHASES[phase]
        root = args.root / phase
        root.mkdir(parents=True, exist_ok=True)
        placed = [(label, trees[tree], card, port + 200 * index, extra) for index, (label, tree, card, extra) in enumerate(arms)]
        servers = []
        try:
            for label, tree, card, arm_port, extra in placed:
                print(f"{phase} {label}: card {card} cpus {card_cpus[card]} port {arm_port} extra {extra} {identity(tree)}", flush=True)
                servers.append((start(tree, card, arm_port, card_cpus[card], extra, root / f"serve_{label}.log"), arm_port))
            for server, arm_port in servers:
                wait_healthy(arm_port, server)
            print(f"{phase}: both healthy", flush=True)
            for cell, jobs in cells:
                with concurrent.futures.ThreadPoolExecutor(len(placed)) as pool:
                    list(pool.map(lambda arm: run_arm_cell(arm[1], arm[3], root / cell / arm[0], jobs, args.client_cpus), placed))
                for label, _, card, _, _ in placed:
                    for name, _, _, _ in jobs:
                        print(f"{phase} {cell} {label} card {card} {name}: {line(root / cell / label / name)}", flush=True)
        finally:
            for server, arm_port in servers:
                stop_group(server, arm_port)
        for label, _, _, _, _ in placed:
            log = (root / f"serve_{label}.log").read_text(errors="replace")
            print(f"{phase} {label}: tracebacks {log.count('Traceback')}, left context {log.count('left-context decoder')}, "
                  f"capture failed {log.count('capture_failed')}, whole request failed {log.count('whole request')}", flush=True)
        port += 400
    print("review cells done", flush=True)


if __name__ == "__main__":
    main()
