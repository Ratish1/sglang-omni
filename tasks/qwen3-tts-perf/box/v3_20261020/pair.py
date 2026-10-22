"""Same-time pair of two trees (or one tree served two ways), then scoring, with each arm's SM Issue.

usage: python3 pair.py --label v3_c16_a --tree /workspace/m_bcd0d8c68 --b-tree /workspace/v3_bb673b57a
       --concurrency 16 --a-card 0 --b-card 1 --port 11000 [--no-stream] [--seed 1234] [--gpu-metrics]

Each arm is placed on its card with the stage gpu flags (no device env), both boot and bench at once (SeedTTS EN, full
corpus, streaming unless --no-stream, warmup 1), then are stopped and scored one after the other, each on its own card:
WER from an ASR server placed with its stage gpu flag and given its rendezvous port (MASTER_PORT, the scorer port plus
50) outside the 29500 block every container on the host claims from without seeing the others, then speaker
similarity. With --gpu-metrics each server runs under nsys with --trace=none (hardware counter sampling on its own
card, no CUDA injection), and the gh100 metrics' window means over the arm's own benchmark go to metrics.txt.
Results under /data/v3_20261020/pairs/<label>/{a,b}.
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
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
ROOT = Path(os.environ.get("PAIR_ROOT", "/data/v3_20261020/pairs"))
KIT = Path("/workspace/t_marks/tasks/omni_step_profiling_20260918/scripts")
HEALTH_TIMEOUT_S = 1800.0
STOP_TIMEOUT_S = 300.0
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


def serve_groups(port: int) -> set[int]:
    """Process groups of live serve processes on port: under nsys the server runs in a group of its own."""
    listing = subprocess.run(["ps", "-eo", "pgid=,stat=,args="], capture_output=True, text=True).stdout
    groups = set()
    for line in listing.splitlines():
        fields = line.split(None, 2)
        if (
            len(fields) == 3
            and not fields[1].startswith("Z")
            and "sglang_omni.cli serve" in fields[2]
            and f"--port {port} " in f"{fields[2]} "
        ):
            groups.add(int(fields[0]))
        else:
            pass
    return groups


def stop_group(server: subprocess.Popen, port: int) -> None:
    """TERM to the leader's group: a bare server exits; nsys ends its target and writes its report. Then any serve
    process still on the port, in its own group under nsys, is stopped the same way."""
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        server.wait(timeout=STOP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
    for group in serve_groups(port):
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while time.monotonic() < deadline and group in serve_groups(port):
            time.sleep(1)
        if group in serve_groups(port):
            os.killpg(group, signal.SIGKILL)
        else:
            pass


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
    parser.add_argument("--b-tree", default="")
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--a-card", type=int, required=True)
    parser.add_argument("--b-card", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--b-args", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--gpu-metrics", action="store_true")
    parser.add_argument("--decode-digest", action="store_true")
    parser.add_argument("--decode-census", action="store_true")
    parser.add_argument("--meta", default="")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args()
    arms = {"a": (args.a_card, "", args.port), "b": (args.b_card, args.b_args, args.port + 200)}
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
        server_env = dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1")
        if args.decode_digest or args.decode_census:
            (outs[arm] / "digest").mkdir(exist_ok=True)
            server_env["PYTHONPATH"] = f"{KIT / 'decode_digest'}:{tree}"
            server_env["OMNI_DECODE_DIGEST"] = str(outs[arm] / "digest")
            server_env["OMNI_DECODE_CENSUS"] = "1" if args.decode_census else "0"
        else:
            pass
        imported = subprocess.run(
            [sys.executable, "-c", "import sglang_omni; print(sglang_omni.__file__)"],
            cwd=tree, env=server_env, capture_output=True, text=True,
        ).stdout
        (outs[arm] / "tree.txt").write_text(
            f"{tree} card {card} serve args: {placement} {extra} metrics: {args.gpu_metrics}\n{identity}imports {imported}",
            encoding="utf-8",
        )
        command = [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, "--host", "127.0.0.1",
                   "--port", str(port), *shlex.split(placement), *shlex.split(extra)]
        if args.gpu_metrics:
            command = ["nsys", "profile", "-o", str(outs[arm] / "metrics"), "--force-overwrite=true", "--trace=none",
                       "--sample=none", "--cpuctxsw=none", f"--gpu-metrics-devices={card}", "--gpu-metrics-set=gh100",
                       *command]
        else:
            pass
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
        stream_args = [] if args.no_stream else ["--stream"]
        for arm, (_, _, port) in arms.items():
            command = [sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL,
                       "--use-existing-server", "--host", "127.0.0.1", "--port", str(port), "--generate-only",
                       *stream_args, "--lang", "en", "--warmup", "1", "--max-concurrency", str(args.concurrency),
                       "--output-dir", str(outs[arm] / "bench")]
            if args.seed is not None:
                command += ["--seed", str(args.seed)]
            else:
                pass
            if args.meta:
                command += ["--meta", args.meta]
            else:
                pass
            with open(outs[arm] / "bench.log", "w", encoding="utf-8") as bench_log:
                benches[arm] = subprocess.Popen(
                    command, cwd=args.tree, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=args.tree),
                    stdout=bench_log, stderr=subprocess.STDOUT,
                )
        for arm, bench in benches.items():
            print(f"{arm} bench rc {bench.wait(timeout=BENCH_TIMEOUT_S)}", flush=True)
        for arm in arms:
            (outs[arm] / "gpu_after_bench.txt").write_text(gpu_snapshot(), encoding="utf-8")
    finally:
        for arm, server in servers.items():
            stop_group(server, arms[arm][2])
    if args.gpu_metrics:
        for arm in arms:
            report = outs[arm] / "metrics.nsys-rep"
            subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "-o",
                            str(outs[arm] / "metrics.sqlite"), str(report)], capture_output=True, check=False)
            with open(outs[arm] / "metrics.txt", "w", encoding="utf-8") as metrics:
                subprocess.run([sys.executable, str(KIT / "gpu_metrics_window.py"), str(outs[arm] / "metrics.sqlite"),
                                "--bench-log", str(outs[arm] / "bench.log")], cwd=KIT, stdout=metrics,
                               stderr=subprocess.STDOUT, check=False)
            (outs[arm] / "metrics.sqlite").unlink(missing_ok=True)
        print("metrics read", flush=True)
    else:
        pass
    if args.seed is not None:
        a_hashes, b_hashes = audio_hashes(outs["a"]), audio_hashes(outs["b"])
        same = sum(a_hashes[name] == b_hashes.get(name) for name in a_hashes)
        print(f"byte identical {same} of {len(a_hashes)}", flush=True)
    else:
        pass
    if args.skip_score:
        pass
    else:
        for index, arm in enumerate(arms):
            score(outs[arm], arms[arm][0], args.port + 400 + 400 * index, args.tree)
        print("scored", flush=True)
    print("pair done", flush=True)


def score(out: Path, card: int, port: int, tree: str) -> None:
    """WER from an ASR server on the arm's own card, then speaker similarity on that card: the scorer's own servers
    default to card 0, which another container may own. The ASR pipeline is one stage, so one rendezvous port."""
    env = dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1")
    asr_env = dict(env, MASTER_PORT=str(port + 50))
    scorer = [sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL, "--lang", "en",
              "--skip-gpu-cleanup", "--output-dir", str(out / "bench")]
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
                [*scorer, "--transcribe-only", "--use-existing-server", "--host", "127.0.0.1", "--port", str(port)],
                cwd=tree, env=env, stdout=score_log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S, check=False,
            )
    finally:
        stop_group(asr, port)
    with open(out / "score--similarity-only.log", "w", encoding="utf-8") as score_log:
        subprocess.run(
            [*scorer, "--similarity-only", "--device", f"cuda:{card}"],
            cwd=tree, env=env, stdout=score_log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S, check=False,
        )


if __name__ == "__main__":
    main()
