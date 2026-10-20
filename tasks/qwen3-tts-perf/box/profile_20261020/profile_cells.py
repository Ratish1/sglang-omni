"""Capture torch profiler traces of the plan's cells (PLAN.md 2) from one Qwen3-TTS Base server, or its clean bench.

usage: python3 profile_cells.py --tree /workspace/sglang-omni --card 0 --port 31000 --mode formal|mapping|clean \
       --out /data/prof_20261020/formal [--cells W1 W2 W3 W4]
formal: the served config, stacks off. mapping: every graph and compile off, stacks on. clean: the served config
without a profiler, benches c1 (200), c16 and c64 on the full corpus. Each boot archives the tree identity, the serve
command and log, and samples /proc/loadavg and the card's processes every 5 s. Every cell runs once unprofiled to warm
its shapes, then once inside /start_profile (torch trace plus request events) and /stop_profile; the trace is complete
when its .trace.json sibling is gone (the server gzips in the background).
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

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MAPPING_ARGS = (
    "--tts_engine.engine.disable_cuda_graph true --vocoder.factory.initial_cuda_graph false "
    "--vocoder.factory.followup_cuda_graph false --vocoder.factory.incremental_codec_cuda_graph false "
    "--vocoder.factory.incremental_codec_compile false"
).split()
WINDOW_LEAD_S = {"W3": 10.0, "W4": 10.0}
# note (ratish): a mapping window only has to hold each call site once, and stacks make it large.
WINDOW_S = {"formal": 2.0, "mapping": 1.0}
LONG_TEXT_SENTENCES = 8
LONG_TEXT_REQUESTS = 3


def run(command: list[str], cwd: str, log: Path, timeout: float = 3600) -> None:
    with open(log, "w", encoding="utf-8") as handle:
        subprocess.run(
            command, cwd=cwd, env=dict(os.environ, PYTHONPATH=cwd), stdout=handle, stderr=subprocess.STDOUT,
            check=True, timeout=timeout,
        )


def bench_command(port: int, concurrency: int, out: Path, extra: list[str]) -> list[str]:
    return [
        sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL, "--use-existing-server",
        "--host", "127.0.0.1", "--port", str(port), "--generate-only", "--stream", "--lang", "en",
        "--max-concurrency", str(concurrency), "--output-dir", str(out), *extra,
    ]


def write_long_text_meta(tree: str, folder: Path) -> Path:
    """One median-length reference; each row's target joins LONG_TEXT_SENTENCES corpus sentences."""
    script = f"""
import shutil, soundfile
from pathlib import Path
from benchmarks.dataset.seedtts import SEEDTTS_DATASET_ID, load_seedtts_samples
samples = load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")
by_seconds = sorted(samples, key=lambda sample: soundfile.info(sample.ref_audio).duration)
reference = by_seconds[len(by_seconds) // 2]
folder = Path({str(folder)!r})
folder.mkdir(parents=True, exist_ok=True)
shutil.copyfile(reference.ref_audio, folder / "ref.wav")
rows = []
for row in range({LONG_TEXT_REQUESTS} + 1):
    picked = samples[row * {LONG_TEXT_SENTENCES}:(row + 1) * {LONG_TEXT_SENTENCES}]
    text = " ".join(sample.target_text for sample in picked)
    rows.append(f"long_{{row}}|{{reference.ref_text}}|ref.wav|{{text}}")
(folder / "meta.lst").write_text(rows[0] + "\\n", encoding="utf-8")
(folder / "meta_profiled.lst").write_text("\\n".join(rows[1:]) + "\\n", encoding="utf-8")
print(reference.sample_id, soundfile.info(reference.ref_audio).duration)
"""
    output = subprocess.run(
        [sys.executable, "-c", script], cwd=tree, env=dict(os.environ, PYTHONPATH=tree), capture_output=True,
        text=True, check=True,
    ).stdout
    (folder / "reference.txt").write_text(output, encoding="utf-8")
    return folder


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
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                capture_output=True, text=True,
            ).stdout.strip().replace("\n", "; ")
            log.write(f"{time.strftime('%H:%M:%S')} load {' '.join(load)} | apps {apps}\n")
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
        traces = list(cell_dir.glob("trace*.trace.json.gz"))
        pending = list(cell_dir.glob("trace*.trace.json"))
        if traces and not pending:
            return
        else:
            pass
        time.sleep(1)
    raise TimeoutError(f"trace of {cell} not complete")


def cell_workload(cell: str, meta: Path, warm: bool) -> tuple[int, list[str]]:
    if cell == "W1":
        offset = "600" if warm else "0"
        return 1, ["--warmup", "0", "--max-samples", "20", "--sample-offset", offset]
    elif cell == "W2":
        listing = meta / ("meta.lst" if warm else "meta_profiled.lst")
        return 1, ["--warmup", "0", "--meta", str(listing)]
    elif cell == "W3":
        return 16, ["--warmup", "1"]
    else:
        return 64, ["--warmup", "1"]


def capture_cell(cell: str, tree: str, port: int, meta: Path, out: Path, window_s: float) -> None:
    cell_dir = out / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    concurrency, extra = cell_workload(cell, meta, warm=True)
    if cell in WINDOW_LEAD_S:
        warm_extra = [*extra, "--max-samples", "128"]
    else:
        warm_extra = extra
    run(bench_command(port, concurrency, cell_dir / "warm", warm_extra), tree, cell_dir / "warm.log")
    concurrency, extra = cell_workload(cell, meta, warm=False)
    if cell in WINDOW_LEAD_S:
        with open(cell_dir / "bench.log", "w", encoding="utf-8") as log:
            bench = subprocess.Popen(
                bench_command(port, concurrency, cell_dir / "bench", extra), cwd=tree,
                env=dict(os.environ, PYTHONPATH=tree), stdout=log, stderr=subprocess.STDOUT,
            )
            time.sleep(WINDOW_LEAD_S[cell])
            start_profile(port, cell, cell_dir)
            time.sleep(window_s)
            stop_profile(port, cell, cell_dir)
            bench.wait(timeout=3600)
    else:
        start_profile(port, cell, cell_dir)
        run(bench_command(port, concurrency, cell_dir / "bench", extra), tree, cell_dir / "bench.log")
        stop_profile(port, cell, cell_dir)
    print(f"{cell} captured {time.strftime('%H:%M:%S')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=("formal", "mapping", "clean"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cells", nargs="*", default=["W1", "W2", "W3", "W4"])
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    identity = subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo DIRTY",
        shell=True, cwd=args.tree, capture_output=True, text=True,
    ).stdout
    serve = [
        sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, "--host", "127.0.0.1",
        "--port", str(args.port), "--tts_engine.gpu", str(args.card), "--vocoder.gpu", str(args.card),
        *(MAPPING_ARGS if args.mode == "mapping" else []),
    ]
    env = dict(os.environ, PYTHONPATH=args.tree, SGLANG_OMNI_STRICT_PORT="1")
    if args.mode == "mapping":
        env["SGLANG_TORCH_PROFILER_WITH_STACK"] = "1"
    else:
        env.pop("SGLANG_TORCH_PROFILER_WITH_STACK", None)
    (out / "boot.json").write_text(
        json.dumps({"tree": args.tree, "identity": identity.split(), "card": args.card, "mode": args.mode,
                    "serve": serve, "with_stack": env.get("SGLANG_TORCH_PROFILER_WITH_STACK")}, indent=1),
        encoding="utf-8",
    )
    stop_sampling = threading.Event()
    sampler = threading.Thread(target=sample_host, args=(out, stop_sampling), daemon=True)
    sampler.start()
    with open(out / "serve.log", "w", encoding="utf-8") as serve_log:
        server = subprocess.Popen(serve, cwd=args.tree, env=env, stdout=serve_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    try:
        wait_healthy(args.port, server)
        print(f"healthy {time.strftime('%H:%M:%S')}", flush=True)
        run(bench_command(args.port, 4, out / "warmup", ["--max-samples", "32", "--sample-offset", "700"]),
            args.tree, out / "warmup.log")
        if args.mode == "clean":
            for label, concurrency, extra in (("c1", 1, ["--max-samples", "200"]), ("c16", 16, []), ("c64", 64, [])):
                run(bench_command(args.port, concurrency, out / f"bench_{label}", ["--warmup", "1", *extra]),
                    args.tree, out / f"bench_{label}.log")
                print(f"{label} benched {time.strftime('%H:%M:%S')}", flush=True)
        else:
            meta = write_long_text_meta(args.tree, out / "meta_w2")
            for cell in args.cells:
                capture_cell(cell, args.tree, args.port, meta, out, WINDOW_S[args.mode])
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
        stop_sampling.set()


if __name__ == "__main__":
    main()
