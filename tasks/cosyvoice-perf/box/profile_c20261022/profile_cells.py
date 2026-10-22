"""Capture torch profiler traces of profile_20261022/PLAN.md's cells from one Fun-CosyVoice3 server, or read its memory.
The Qwen3-TTS driver (tasks/qwen3-tts-perf/box/profile_20261020/profile_cells.py) with CosyVoice's model, graph and
compile switches, cells and paths.

usage: python3 profile_cells.py --tree /workspace/t_chain --card 0 --port 47000 --mode formal|mapping|memory \
       --out /data/c20261022b/prof/formal [--cells S_c16 P_c1 ...]
formal: the served config, stacks off. mapping: every graph and compile off, stacks on. memory: the served config with
the pool probe (pool_probe/: the Flow prefix pool's peak frames and failed allocations, each whole utterance HiFT
decode's peak), no profiler. Each boot archives the tree identity, the serve command and log, and samples
/proc/loadavg and the card's processes every 5 s. Every cell runs once unprofiled to warm its shapes, then once inside
/start_profile and /stop_profile (a window opened after the cell's lead for the long ones); the trace is complete when
its .trace.json sibling is gone (the server gzips in the background). cells.tsv holds each measured run's window.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
COSYVOICE = "/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"
HERE = Path(__file__).resolve().parent
CPUS = {"0": "0-15,64-79", "1": "16-31,80-95"}
MAPPING_ARGS = (
    "--tts_engine.engine.disable_cuda_graph true --vocoder.factory.enable_flow_cuda_graph false "
    "--vocoder.factory.enable_flow_prefix_cuda_graph false --vocoder.factory.enable_flow_whole_history_cuda_graph false "
    "--vocoder.factory.enable_hift_cuda_graph false --vocoder.factory.enable_dit_torch_compile false"
).split()
# cell: (concurrency, streaming, the long cells' list name or None for SeedTTS EN, measured SeedTTS requests)
CELLS = {
    "S_c16": (16, True, None, 320),
    "S_c64": (64, True, None, 640),
    "Sb_c16": (16, False, None, 320),
    "P_c1": (1, True, "P_c1", None),
    "P_c16": (16, True, "P_c16", None),
    "D_c1": (1, True, "D_c1", None),
    "D_c16": (16, True, "D_c16", None),
    "D_c64": (64, True, "D_c64", None),
    "Db_c16": (16, False, "Db_c16", None),
    "PD_c16": (16, True, "PD_c16", None),
}
# note (ratish): seconds from a measured run's start to its window, estimated from the served RTF and the AR's 32
# running requests so each window holds the cell's steady shape, not its ramp or tail; a cell absent here is traced
# whole.
WINDOW_LEAD_S = {
    "S_c16": 5.0,
    "S_c64": 8.0,
    "Sb_c16": 4.0,
    "P_c16": 2.0,
    "D_c1": 2.0,
    "D_c16": 8.0,
    "D_c64": 10.0,
    "Db_c16": 6.0,
    "PD_c16": 8.0,
}
# note (ratish): a mapping window only has to hold each call site once, and stacks make it large.
WINDOW_S = {"formal": 2.0, "mapping": 1.0}


def environment(tree: str, card: str, mode: str) -> dict[str, str]:
    probe = f"{HERE / 'pool_probe'}:" if mode == "memory" else ""
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=card, PYTHONPATH=f"{probe}{tree}:{COSYVOICE}", SGLANG_OMNI_STRICT_PORT="1",
        TORCHINDUCTOR_CACHE_DIR=f"{os.environ.get('PAIR_HOME', '/data/c20261022b')}/inductor_{Path(tree).name}",
    )
    if mode == "mapping":
        env["SGLANG_TORCH_PROFILER_WITH_STACK"] = "1"
    else:
        env.pop("SGLANG_TORCH_PROFILER_WITH_STACK", None)
    if mode == "memory":
        env["POOL_PROBE"] = "1"
        env["POOL_PROBE_OUT"] = str(Path(os.environ["PROFILE_OUT"]) / "pool")
    else:
        pass
    return env


def run(command: list[str], tree: str, card: str, log: Path, timeout: float = 3600) -> None:
    with open(log, "w", encoding="utf-8") as handle:
        subprocess.run(
            ["taskset", "-c", CPUS[card], *command], cwd=tree,
            env=dict(os.environ, PYTHONPATH=f"{tree}:{COSYVOICE}", CUDA_VISIBLE_DEVICES=""), stdout=handle,
            stderr=subprocess.STDOUT, check=True, timeout=timeout,
        )


def bench_command(port: int, cell: str, out: Path, meta: Path, warm: bool) -> list[str]:
    concurrency, streaming, listing, samples = CELLS[cell]
    if listing is None:
        offset = ["--sample-offset", "700", "--max-samples", "64"] if warm else ["--max-samples", str(samples)]
        workload = ["--warmup", "1", *offset]
    else:
        workload = ["--warmup", "0", "--meta", str(meta / f"{listing}_{'warm' if warm else 'run'}.lst")]
    return [
        sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL, "--use-existing-server",
        "--host", "127.0.0.1", "--port", str(port), "--generate-only", *(["--stream"] if streaming else []),
        "--lang", "en", "--max-concurrency", str(concurrency), "--output-dir", str(out), *workload,
    ]


def wait_healthy(port: int, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with {server.returncode}")
        else:
            pass
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2, trust_env=False).status_code == 200:
                return
            else:
                pass
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("server not healthy")


def sample_host(out: Path, stop: threading.Event) -> None:
    with open(out / "host.log", "a", encoding="utf-8") as log:
        while not stop.is_set():
            load = Path("/proc/loadavg").read_text(encoding="utf-8").split()[:3]
            cards = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip().replace("\n", "; ")
            log.write(f"{time.strftime('%H:%M:%S')} load {' '.join(load)} | cards {cards}\n")
            log.flush()
            stop.wait(5)


def start_profile(port: int, cell: str, cell_dir: Path) -> None:
    body = {"run_id": cell, "event_dir": str(cell_dir / "events"), "trace_path_template": str(cell_dir / "trace")}
    httpx.post(f"http://127.0.0.1:{port}/start_profile", json=body, timeout=60, trust_env=False).raise_for_status()


def stop_profile(port: int, cell: str, cell_dir: Path) -> None:
    httpx.post(
        f"http://127.0.0.1:{port}/stop_profile", json={"run_id": cell}, timeout=600, trust_env=False
    ).raise_for_status()
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if list(cell_dir.glob("trace*.trace.json.gz")) and not list(cell_dir.glob("trace*.trace.json")):
            return
        else:
            pass
        time.sleep(1)
    raise TimeoutError(f"trace of {cell} not complete")


def capture_cell(cell: str, args: argparse.Namespace, meta: Path, out: Path) -> None:
    cell_dir = out / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    run(bench_command(args.port, cell, cell_dir / "warm", meta, warm=True), args.tree, args.card, cell_dir / "warm.log")
    command = bench_command(args.port, cell, cell_dir / "bench", meta, warm=False)
    started = time.time()
    if args.mode == "memory":
        run(command, args.tree, args.card, cell_dir / "bench.log")
    elif cell in WINDOW_LEAD_S:
        with open(cell_dir / "bench.log", "w", encoding="utf-8") as log:
            bench = subprocess.Popen(
                ["taskset", "-c", CPUS[args.card], *command], cwd=args.tree,
                env=dict(os.environ, PYTHONPATH=f"{args.tree}:{COSYVOICE}", CUDA_VISIBLE_DEVICES=""), stdout=log,
                stderr=subprocess.STDOUT,
            )
            time.sleep(WINDOW_LEAD_S[cell])
            start_profile(args.port, cell, cell_dir)
            time.sleep(WINDOW_S[args.mode])
            stop_profile(args.port, cell, cell_dir)
            if bench.wait(timeout=3600) != 0:
                raise RuntimeError(f"{cell} bench failed")
            else:
                pass
    else:
        start_profile(args.port, cell, cell_dir)
        run(command, args.tree, args.card, cell_dir / "bench.log")
        stop_profile(args.port, cell, cell_dir)
    with open(out / "cells.tsv", "a", encoding="utf-8") as cells:
        cells.write(f"{cell}\t{started:.1f}\t{time.time():.1f}\n")
    print(f"{cell} {args.mode} done {time.strftime('%H:%M:%S')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--card", choices=sorted(CPUS), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=("formal", "mapping", "memory"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--meta", required=True, help="the folder long_cells_meta.py wrote")
    parser.add_argument("--cells", nargs="*", default=list(CELLS))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["PROFILE_OUT"] = str(out)
    identity = subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo DIRTY",
        shell=True, cwd=args.tree, capture_output=True, text=True,
    ).stdout
    serve = [
        "taskset", "-c", CPUS[args.card], sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL,
        "--host", "127.0.0.1", "--port", str(args.port), *(MAPPING_ARGS if args.mode == "mapping" else []),
    ]
    env = environment(args.tree, args.card, args.mode)
    (out / "boot.json").write_text(
        json.dumps({"tree": args.tree, "identity": identity.split(), "card": args.card, "mode": args.mode,
                    "serve": serve, "with_stack": env.get("SGLANG_TORCH_PROFILER_WITH_STACK")}, indent=1),
        encoding="utf-8",
    )
    stop_sampling = threading.Event()
    threading.Thread(target=sample_host, args=(out, stop_sampling), daemon=True).start()
    with open(out / "serve.log", "w", encoding="utf-8") as serve_log:
        server = subprocess.Popen(serve, cwd=args.tree, env=env, stdout=serve_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    try:
        wait_healthy(args.port, server)
        print(f"healthy {time.strftime('%H:%M:%S')}", flush=True)
        for cell in args.cells:
            capture_cell(cell, args, Path(args.meta), out)
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
        stop_sampling.set()


if __name__ == "__main__":
    main()
