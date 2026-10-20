"""Where an arm's req/s goes: interpreter-lock waits and CUDA API time per thread of the talker process.

Run on the box: python3 /data/pr2800/gil_wait_check.py --out /data/pr2800/<run> --arms main@0,flat@0 [--concurrency 64] [--graph-trace node] [--gpu-metrics] [--python-backtrace]
Each arm serves the colocated bf16 H100 command under nsys launch with CUDA and OS runtime tracing (pthread waits),
pinned to its card's cores; arms on one card run one after the other, arms on different cards at once. SeedTTS EN
streamed with voice clone at the given concurrency (16 by default) starts once the server is healthy; nsys records
15 s from 40 s into the pass. Writes <out>/<arm>/trace.{nsys-rep,sqlite} and <out>/summary.json with the stage pids.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from pair_check import (
    CARD_CPUS,
    CONFIG,
    MODEL,
    NODE,
    SERVER_PORT,
    ARMS,
    benchmark,
    card_env,
    log,
    stop,
    tree,
)

PID_LINE = re.compile(r"sglang_ar_started stage=(\w+) .* pid=(\d+)")


def run_arm(
    out: Path,
    index: int,
    card: int,
    arm: str,
    concurrency: int,
    graph_trace: str,
    gpu_metrics: bool,
    python_backtrace: bool,
) -> dict[str, object]:
    tree_name, extra, _ = ARMS[arm]
    tree_path = tree(tree_name)
    arm_dir = out / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    port = SERVER_PORT + 200 * index
    session = f"gil{index}"
    log(out, f"{arm} card {card} start")
    server = subprocess.Popen(
        ["numactl", f"--physcpubind={CARD_CPUS[card]}", f"--membind={NODE}",
         "nsys", "launch", f"--session-new={session}", "-t", "cuda,osrt", f"--cuda-graph-trace={graph_trace}",
         *(["--python-backtrace=cuda", "--cudabacktrace=all"] if python_backtrace else []),
         sys.executable, "-u", "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, "--config", CONFIG,
         "--colocate", *extra, "--host", "127.0.0.1", "--port", str(port)],
        cwd=tree_path,
        env={**card_env(card, tree_path), "SGLANG_OMNI_STRICT_PORT": "1", "SGLANG_OMNI_STARTUP_TIMEOUT": "1800"},
        stdout=open(arm_dir / "server.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    record: dict[str, object] = {"arm": arm, "card": card}
    try:
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                    break
                else:
                    pass
            except httpx.HTTPError:
                pass
            time.sleep(5)
        record["pids"] = {
            stage: int(pid)
            for stage, pid in PID_LINE.findall((arm_dir / "server.log").read_text(errors="replace"))
        }
        metrics = (
            f"--gpu-metrics-devices={card} --gpu-metrics-set=gh100 --gpu-metrics-frequency=20000 "
            if gpu_metrics
            else ""
        )
        sampling = "--sample=process-tree" if python_backtrace else "--sample=none"
        capture = subprocess.Popen(
            ["bash", "-c", f"sleep 40; nsys start --session={session} {sampling} --cpuctxsw=none {metrics}"
             f"-o {arm_dir / 'trace'} --force-overwrite=true; sleep 15; nsys stop --session={session}"],
            stdout=open(arm_dir / "capture.log", "w"),
            stderr=subprocess.STDOUT,
        )
        pass_name = f"c{concurrency}"
        record["rc"] = benchmark(
            card,
            ["--host", "127.0.0.1", "--port", str(port), "--model", "qwen3-omni", "--lang", "en",
             "--voice-clone", "--stream", "--warmup", "1", "--generate-only", "--disable-tqdm",
             "--max-concurrency", str(concurrency), "--output-dir", str(arm_dir / pass_name)],
            arm_dir / f"{pass_name}.log",
        )
        capture.wait()
        log(out, f"{arm} {pass_name} rc {record['rc']}")
    finally:
        stop(card, server)
    for wav in (arm_dir / f"c{concurrency}").rglob("*.wav"):
        wav.unlink()
    for report in sorted(arm_dir.glob("trace*.nsys-rep")):
        subprocess.run(["nsys", "export", "-t", "sqlite", "--force-overwrite=true", "-o",
                        str(report.with_suffix(".sqlite")), str(report)], check=False)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", required=True, help="'arm@card,arm@card'")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--graph-trace", choices=["graph", "node"], default="graph")
    parser.add_argument("--gpu-metrics", action="store_true", help="sample the card's clocks and units at 20 kHz")
    parser.add_argument(
        "--python-backtrace", action="store_true", help="a Python backtrace on every CUDA call, with CPU sampling on"
    )
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "driver.pgid").write_text(str(os.getpgrp()))
    placements = [(arm, int(card)) for arm, card in (entry.split("@") for entry in args.arms.split(","))]
    by_card: dict[int, list[tuple[int, str]]] = {}
    for index, (arm, card) in enumerate(placements, start=1):
        by_card.setdefault(card, []).append((index, arm))

    def run_card(card: int) -> list[dict[str, object]]:
        return [
            run_arm(
                out, index, card, arm, args.concurrency, args.graph_trace, args.gpu_metrics, args.python_backtrace
            )
            for index, arm in by_card[card]
        ]

    with ThreadPoolExecutor(max_workers=len(by_card)) as pool:
        futures = [pool.submit(run_card, card) for card in by_card]
    (out / "summary.json").write_text(
        json.dumps([record for future in futures for record in future.result()], indent=1)
    )
    log(out, "done")


if __name__ == "__main__":
    main()
