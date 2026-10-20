"""Same-time pair of one tree served two ways (arm a: the given flags, arm b: those plus --b-args), or of two trees
(--b-tree), then scoring.

usage: python3 pair.py --label pre_post_c16_a --tree /workspace/t_8220d2207 --b-tree /workspace/t_08182f2bc
       --concurrency 16 --a-card 0 --b-card 1 --port 20000 [--no-stream] [--seed 1234] [--skip-score] [--wer-only]

Each arm is placed on its card with the stage gpu flags (the container sees only our cards, so no device env), both
are booted at once and benchmarked at once (SeedTTS EN, full corpus, streaming unless --no-stream, warmup 1), stopped
by process group, then scored one after the other: WER, then speaker similarity. With --seed it also counts
byte-identical samples. Each arm's spawn-to-healthy seconds go to startup.txt (both arms boot at once, so these share
the host's CPUs) and the card's processes and memory to gpu_healthy.txt and gpu_after_bench.txt.
Results under /data/async_decode_20261020/pairs/<label>/{a,b}.
"""

import argparse
import hashlib
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ROOT = Path("/data/async_decode_20261020/pairs")
HEALTH_TIMEOUT_S = 1800.0
STOP_TIMEOUT_S = 90.0
BENCH_TIMEOUT_S = 7200.0


def wait_healthy(port: int, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server on port {port} exited with {server.returncode}")
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
    raise TimeoutError(f"server on port {port} not healthy in {HEALTH_TIMEOUT_S} s")


def group_alive(pgid: int) -> bool:
    """A non-zombie member remains (the container's PID 1 does not reap exited members)."""
    listing = subprocess.run(["ps", "-eo", "pgid=,stat="], capture_output=True, text=True).stdout
    return any(
        fields[0] == str(pgid) and not fields[1].startswith("Z")
        for fields in (line.split() for line in listing.splitlines())
        if len(fields) == 2
    )


def stop_group(server: subprocess.Popen) -> None:
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        if not group_alive(server.pid):
            return
        else:
            pass
        time.sleep(1)
    os.killpg(server.pid, signal.SIGKILL)


def gpu_snapshot() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout


def audio_hashes(out: Path) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((out / "bench" / "audio").glob("*.wav"))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--a-card", type=int, required=True)
    parser.add_argument("--b-card", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--a-args", default="")
    parser.add_argument("--b-args", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--wer-only", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--config", default="")
    parser.add_argument("--bench-extra", default="")
    parser.add_argument("--b-env", default="", help="KEY=VALUE pairs set for arm b's server only")
    parser.add_argument(
        "--step-phases", action="store_true",
        help="each arm's server writes its loop phase totals to step_phases.jsonl (trees carrying the probe), and the "
        "bench window goes to bench_window.txt",
    )
    parser.add_argument("--b-pythonpath", default="", help="a directory put ahead of the tree for arm b's server")
    parser.add_argument(
        "--b-tree", default="",
        help="another source tree for arm b's server (its working directory and PYTHONPATH; python -m puts the working "
        "directory first on sys.path, so a PYTHONPATH entry alone cannot replace the tree)",
    )
    args = parser.parse_args()
    model = args.model
    config_args = ["--config", args.config] if args.config else []
    arms = {
        "a": (args.a_card, args.a_args, args.port),
        "b": (args.b_card, f"{args.a_args} {args.b_args}", args.port + 200),
    }
    outs = {arm: ROOT / args.label / arm for arm in arms}
    trees = {"a": args.tree, "b": args.b_tree or args.tree}
    servers = {}
    spawned = {}
    for arm, (card, extra, port) in arms.items():
        outs[arm].mkdir(parents=True, exist_ok=True)
        tree = trees[arm]
        identity = subprocess.run(
            "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo WORKING_TREE_DIRTY",
            shell=True, cwd=tree, capture_output=True, text=True,
        ).stdout
        placement = f"--tts_engine.gpu {card} --vocoder.gpu {card}"
        if "--preprocessing.process" in extra:
            placement += f" --preprocessing.gpu {card}"
        else:
            pass
        server_env = dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1")
        if args.step_phases:
            server_env["OMNI_STEP_PHASES"] = str(outs[arm] / "step_phases.jsonl")
        else:
            pass
        if arm == "b":
            server_env.update(item.split("=", 1) for item in shlex.split(args.b_env))
            if args.b_pythonpath:
                server_env["PYTHONPATH"] = f"{args.b_pythonpath}:{tree}"
            else:
                pass
        else:
            pass
        imported = subprocess.run(
            [sys.executable, "-c", "import sglang_omni; print(sglang_omni.__file__)"],
            cwd=tree, env=server_env, capture_output=True, text=True,
        ).stdout
        (outs[arm] / "tree.txt").write_text(
            f"{tree} card {card} serve args: {placement} {extra} env: "
            f"{args.b_env if arm == 'b' else ''} pythonpath: {server_env['PYTHONPATH']}\n{identity}imports {imported}",
            encoding="utf-8",
        )
        with open(outs[arm] / "serve.log", "w", encoding="utf-8") as serve_log:
            spawned[arm] = time.monotonic()
            servers[arm] = subprocess.Popen(
                [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", model, *config_args,
                 "--host", "127.0.0.1", "--port", str(port), *shlex.split(placement), *shlex.split(extra)],
                cwd=tree, env=server_env,
                stdout=serve_log, stderr=subprocess.STDOUT, start_new_session=True,
            )
    try:
        for arm, (_, _, port) in arms.items():
            wait_healthy(port, servers[arm])
            (outs[arm] / "startup.txt").write_text(f"{time.monotonic() - spawned[arm]:.1f}\n", encoding="utf-8")
            print(f"{arm} healthy {time.strftime('%H:%M:%S')}", flush=True)
        for arm in arms:
            (outs[arm] / "gpu_healthy.txt").write_text(gpu_snapshot(), encoding="utf-8")
        benches = {}
        bench_start = time.time()
        stream_args = [] if args.no_stream else ["--stream"]
        for arm, (_, _, port) in arms.items():
            command = [sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", model,
                       "--use-existing-server", "--host", "127.0.0.1", "--port", str(port), "--generate-only",
                       *stream_args, "--lang", "en", "--warmup", "1", "--max-concurrency", str(args.concurrency),
                       "--output-dir", str(outs[arm] / "bench"), *shlex.split(args.bench_extra)]
            if args.seed is not None:
                command += ["--seed", str(args.seed)]
            else:
                pass
            with open(outs[arm] / "bench.log", "w", encoding="utf-8") as bench_log:
                benches[arm] = subprocess.Popen(
                    command, cwd=args.tree, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=args.tree),
                    stdout=bench_log, stderr=subprocess.STDOUT,
                )
        for arm, bench in benches.items():
            print(f"{arm} bench rc {bench.wait(timeout=BENCH_TIMEOUT_S)}", flush=True)
            (outs[arm] / "bench_window.txt").write_text(f"{bench_start} {time.time()}\n", encoding="utf-8")
        for arm in arms:
            (outs[arm] / "gpu_after_bench.txt").write_text(gpu_snapshot(), encoding="utf-8")
        for arm, (_, _, port) in arms.items():
            model_info = httpx.get(f"http://127.0.0.1:{port}/model_info", timeout=30, trust_env=False)
            (outs[arm] / "model_info.json").write_text(model_info.text, encoding="utf-8")
    finally:
        for server in servers.values():
            stop_group(server)
    if args.seed is not None:
        a_hashes, b_hashes = audio_hashes(outs["a"]), audio_hashes(outs["b"])
        same = sum(a_hashes[name] == b_hashes.get(name) for name in a_hashes)
        print(f"byte identical {same} of {len(a_hashes)}", flush=True)
    else:
        pass
    if args.skip_score:
        pass
    else:
        phases = (("--transcribe-only", 400),) if args.wer_only else (("--transcribe-only", 400), ("--similarity-only", 600))
        for index, arm in enumerate(arms):
            for phase, offset in phases:
                with open(outs[arm] / f"score{phase}.log", "w", encoding="utf-8") as score_log:
                    subprocess.run(
                        [sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", model, phase,
                         "--lang", "en", "--port", str(args.port + offset + 400 * index),
                         "--output-dir", str(outs[arm] / "bench"), *shlex.split(args.bench_extra)],
                        cwd=args.tree, env=dict(os.environ, PYTHONPATH=args.tree, SGLANG_OMNI_STRICT_PORT="1"),
                        stdout=score_log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S, check=False,
                    )
            print(f"scored {arm}", flush=True)
    print("pair done", flush=True)


if __name__ == "__main__":
    main()
