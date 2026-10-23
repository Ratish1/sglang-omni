"""Qwen3-Omni speech A/B for a vocoder-only change, two trees at the same time on two cards of the H100 box: one
colocated bf16 server per card (examples/configs/qwen3_omni_colocated_h100_bf16.yaml, every GPU stage on that card),
each code2wav request's talker codes and served waveform dumped by code2wav_dump/sitecustomize.py. Passes of
benchmark_omni_seedtts on SeedTTS EN with voice clone over all 1,088 prompts, every arm driven by tree a's client:
seeded c1 streamed and not (the talker draws the same codes in both arms, so their audio differs only by the
vocoder), c16 streamed and not, c64 streamed. Then WER from a Qwen3-ASR server and SIM, each arm on its own card.

Layout, which seeded_quality.py reads: PAIR_ROOT/<label>_<pass>/<arm>/bench is the benchmark's output (audio/<id>.wav),
PAIR_ROOT/<label>_<pass>/<arm>/codes the dumps, PAIR_ROOT/<label>/log.txt the progress and summary.json the rows.

usage: PAIR_ROOT=/data/omni3097/pairs python3 omni_pair.py --label o1 --a-tree OLD --b-tree NEW --a-card 2
           --b-card 3 --port 20000
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/omni3097/pairs"))
KIT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
CONFIG = "examples/configs/qwen3_omni_colocated_h100_bf16.yaml"
GPU_STAGES = ("image_encoder", "audio_encoder", "thinker", "talker_ar", "code2wav")
PASSES = {
    "s_c1_seeded": ["--stream", "--max-concurrency", "1", "--seed", "1234"],
    "ns_c1_seeded": ["--max-concurrency", "1", "--seed", "1234"],
    "s_c16": ["--stream", "--max-concurrency", "16"],
    "ns_c16": ["--max-concurrency", "16"],
    "s_c64": ["--stream", "--max-concurrency", "64"],
}
STARTUP_S = 1800
SPEED_KEYS = (
    "completed_requests",
    "failed_requests",
    "throughput_qps",
    "rtf_mean",
    "latency_mean_s",
    "latency_p99_s",
    "audio_ttfp_median_s",
    "audio_ttfp_p95_s",
    "audio_ttfp_p99_s",
)


def log(label: str, text: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {text}"
    with open(ROOT / label / "log.txt", "a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def card_memory_mib(card: int) -> int:
    used = subprocess.run(
        ["nvidia-smi", "-i", str(card), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.split()
    return int(used[0]) if used and used[0].isdigit() else 0


def start(command: list[str], tree: str, env: dict[str, str], log_path: Path, port: int) -> subprocess.Popen:
    handle = open(log_path, "w")
    process = subprocess.Popen(
        command, cwd=tree, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
    )
    deadline = time.monotonic() + STARTUP_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited, see {log_path}")
        else:
            pass
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5, trust_env=False).status_code == 200:
                return process
            else:
                pass
        except httpx.HTTPError:
            pass
        time.sleep(5)
    stop(process, None)
    raise RuntimeError(f"server not healthy in {STARTUP_S} s, see {log_path}")


def stop(process: subprocess.Popen, card: int | None) -> None:
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    for _ in range(60):
        if card is None or card_memory_mib(card) < 1024:
            return
        else:
            time.sleep(2)


def benchmark(client_tree: str, args: list[str], log_path: Path) -> int:
    with open(log_path, "w") as handle:
        return subprocess.run(
            [sys.executable, "-m", "benchmarks.eval.benchmark_omni_seedtts", *args],
            cwd=client_tree,
            env=dict(os.environ, PYTHONPATH=client_tree),
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode


def run_arm(label: str, arm: str, tree: str, card: int, port: int, client_tree: str, passes: list[str]) -> None:
    arm_dir = ROOT / label / arm
    live_codes = arm_dir / "codes_live"
    live_codes.mkdir(parents=True, exist_ok=True)
    placement = [item for stage in GPU_STAGES for item in (f"--{stage}.gpu", str(card))]
    env = dict(
        os.environ,
        PYTHONPATH=f"{KIT / 'code2wav_dump'}:{tree}",
        CODE2WAV_DUMP_DIR=str(live_codes),
        SGLANG_OMNI_STRICT_PORT="1",
        SGLANG_OMNI_STARTUP_TIMEOUT=str(STARTUP_S),
    )
    identity = subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo DIRTY",
        shell=True, cwd=tree, capture_output=True, text=True,
    ).stdout.split()
    began = time.monotonic()
    server = start(
        [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, "--config", CONFIG, "--colocate",
         *placement, "--host", "127.0.0.1", "--port", str(port)],
        tree, env, arm_dir / "serve.log", port,
    )
    log(label, f"{arm} {' '.join(identity)} card {card} healthy after {time.monotonic() - began:.0f} s")
    try:
        for name in passes:
            pass_dir = ROOT / f"{label}_{name}" / arm
            pass_dir.mkdir(parents=True, exist_ok=True)
            rc = benchmark(
                client_tree,
                ["--host", "127.0.0.1", "--port", str(port), "--model", "qwen3-omni", "--lang", "en",
                 "--voice-clone", "--warmup", "1", "--generate-only", "--disable-tqdm",
                 "--output-dir", str(pass_dir / "bench"), *PASSES[name]],
                pass_dir / "bench.log",
            )
            codes = pass_dir / "codes"
            codes.mkdir(exist_ok=True)
            for dump in live_codes.glob("*.pt"):
                dump.rename(codes / dump.name)
            log(label, f"{arm} {name} rc {rc} dumps {len(list(codes.glob('*.pt')))}")
    finally:
        stop(server, card)
        log(label, f"{arm} stopped")


def score(label: str, arm: str, card: int, port: int, client_tree: str, passes: list[str]) -> None:
    env = dict(os.environ, PYTHONPATH=client_tree, SGLANG_OMNI_STRICT_PORT="1", MASTER_PORT=str(port + 50))
    asr = start(
        [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", ASR_MODEL, "--host", "127.0.0.1",
         "--port", str(port), "--asr.gpu", str(card)],
        client_tree, env, ROOT / label / arm / "asr_serve.log", port,
    )
    try:
        for name in passes:
            pass_dir = ROOT / f"{label}_{name}" / arm
            rc = benchmark(
                client_tree,
                ["--transcribe-only", "--host", "127.0.0.1", "--port", str(port), "--lang", "en",
                 "--output-dir", str(pass_dir / "bench")],
                pass_dir / "wer.log",
            )
            log(label, f"{arm} {name} wer rc {rc}")
    finally:
        stop(asr, card)
    for name in passes:
        pass_dir = ROOT / f"{label}_{name}" / arm
        rc = benchmark(
            client_tree,
            ["--similarity-only", "--lang", "en", "--device", f"cuda:{card}", "--output-dir", str(pass_dir / "bench")],
            pass_dir / "sim.log",
        )
        log(label, f"{arm} {name} sim rc {rc}")


def summary(label: str, passes: list[str]) -> list[dict[str, object]]:
    rows = []
    for name in passes:
        for arm in ("a", "b"):
            bench = ROOT / f"{label}_{name}" / arm / "bench"
            row: dict[str, object] = {"pass": name, "arm": arm}
            speed = bench / "speed_results.json"
            if speed.exists():
                values = json.loads(speed.read_text())["summary"]
                row.update({key: values.get(key) for key in SPEED_KEYS})
            else:
                pass
            wer = bench / "wer_results.json"
            if wer.exists():
                row["wer_corpus"] = json.loads(wer.read_text())["summary"].get("wer_corpus")
            else:
                pass
            sim = bench / "similarity_results.json"
            if sim.exists():
                row["sim"] = json.loads(sim.read_text())["summary"].get("speaker_similarity_mean")
            else:
                pass
            rows.append(row)
            log(label, "summary " + json.dumps(row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--a-tree", required=True)
    parser.add_argument("--b-tree", required=True)
    parser.add_argument("--a-card", type=int, required=True)
    parser.add_argument("--b-card", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--passes", default=",".join(PASSES))
    parser.add_argument("--score", default="s_c1_seeded,ns_c1_seeded,s_c16,ns_c16,s_c64")
    args = parser.parse_args()
    passes = args.passes.split(",")
    scored = [name for name in args.score.split(",") if name in passes]
    (ROOT / args.label).mkdir(parents=True, exist_ok=True)
    arms = {"a": (args.a_tree, args.a_card, args.port), "b": (args.b_tree, args.b_card, args.port + 300)}
    log(args.label, f"start arms {arms} passes {passes} scored {scored}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(run_arm, args.label, arm, tree, card, port, args.a_tree, passes)
            for arm, (tree, card, port) in arms.items()
        ]
        for job in jobs:
            job.result()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [
            pool.submit(score, args.label, arm, card, port + 600, args.a_tree, scored)
            for arm, (tree, card, port) in arms.items()
        ]
        for job in jobs:
            job.result()
    rows = summary(args.label, passes)
    (ROOT / args.label / "summary.json").write_text(json.dumps(rows, indent=1))
    log(args.label, "pair done")


if __name__ == "__main__":
    main()
