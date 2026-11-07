"""One boot of a tree on card 0, then each non-streaming SeedTTS cell on that server in turn, then stop.

usage: python3 arm_cells.py --label 17_H_1 --tree /workspace/t_head --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 12100
       [--cells n1:1:1234,n16:16,n64:64]

A cell is name:concurrency[:seed], all 1,088 SeedTTS EN prompts, warmup 1. Results under $ARM_ROOT/<label>/<cell>/
(bench.log, bench/results.csv; audio kept for seeded cells only, for byte identity). CARD_CPUS="0=<cpus>" starts the
server on the card's socket. tree.txt records the tree's commit and source hash.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3_20261020"))

from pair import BENCH_TIMEOUT_S, stop_group, wait_healthy  # noqa: E402

ROOT = Path(os.environ.get("ARM_ROOT", "/data/rowwait_20261107/arms"))


def run_cell(out: Path, tree: str, model: str, port: int, cell: str) -> str:
    name, concurrency, *seed = cell.split(":")
    cell_out = out / name
    cell_out.mkdir(parents=True, exist_ok=True)
    bench = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--model",
        model,
        "--use-existing-server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--generate-only",
        "--lang",
        "en",
        "--warmup",
        "1",
        "--max-concurrency",
        concurrency,
        "--output-dir",
        str(cell_out / "bench"),
        *(["--seed", seed[0]] if seed else []),
    ]
    with open(cell_out / "bench.log", "w", encoding="utf-8") as bench_log:
        returncode = subprocess.run(
            bench,
            cwd=tree,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=tree),
            stdout=bench_log,
            stderr=subprocess.STDOUT,
            timeout=BENCH_TIMEOUT_S,
            check=False,
        ).returncode
    if not seed:
        shutil.rmtree(cell_out / "bench" / "audio", ignore_errors=True)
    else:
        pass
    return f"{name} rc {returncode}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--cells", default="n1:1:1234,n16:16,n64:64")
    args = parser.parse_args()

    out = ROOT / args.label
    out.mkdir(parents=True, exist_ok=True)
    identity = subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo WORKING_TREE_DIRTY",
        shell=True,
        cwd=args.tree,
        capture_output=True,
        text=True,
    ).stdout
    (out / "tree.txt").write_text(
        f"{args.tree} {args.model}\n{identity}", encoding="utf-8"
    )
    cpus = dict(
        item.split("=") for item in os.environ.get("CARD_CPUS", "").split(";") if item
    ).get("0")
    command = [
        *(["taskset", "-c", cpus] if cpus else []),
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    server_env = dict(os.environ, PYTHONPATH=args.tree, SGLANG_OMNI_STRICT_PORT="1")
    with open(out / "serve.log", "w", encoding="utf-8") as serve_log:
        spawned = time.monotonic()
        server = subprocess.Popen(
            command,
            cwd=args.tree,
            env=server_env,
            stdout=serve_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    outcomes = []
    try:
        wait_healthy(args.port, server)
        (out / "startup.txt").write_text(
            f"{time.monotonic() - spawned:.1f}\n", encoding="utf-8"
        )
        for cell in args.cells.split(","):
            outcomes.append(run_cell(out, args.tree, args.model, args.port, cell))
    finally:
        stop_group(server, args.port)
    tracebacks = (out / "serve.log").read_text(errors="replace").count("Traceback")
    print(
        f"{args.label} {'; '.join(outcomes)}; serve tracebacks {tracebacks}; {identity.split()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
