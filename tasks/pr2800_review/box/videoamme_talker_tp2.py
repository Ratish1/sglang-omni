"""Video-AMME talker in the CI's TP2 shape on the CI sample, one arm at a time on both cards.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/videoamme_talker_tp2.py --arms upk,phs
    --out /data/pr2800/<run>
Each arm serves the CI's FP8 thinker TP2 speech server (examples/run_qwen3_omni_speech_server.py: thinker on cards 0
and 1, talker and code2wav on card 1) and gets the CI's request shape: the first 10 videoamme-ci-50 samples, video and
audio in, text and audio out, greedy, 256 tokens, c16. After the arms, one Qwen3-ASR server on card 0 transcribes each
arm's audio and scores it against the arm's text as the CI does. The client and the scoring run in each tree's own
code (this file in its --client and --score modes). Writes <out>/<arm>/videoamme.json, <out>/<arm>/wer.json and
<out>/summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

MODEL = "marksverdhei/Qwen3-Omni-30B-A3B-FP8"
REPO_ID = "zhaochenyang20/Video_AMME_ci"
MAX_SAMPLES = 10
MAX_TOKENS = 256
CONCURRENCY = 16
ASR_CONCURRENCY = 4
PORT = 28000
SERVE_ARGS = [
    "--model-name",
    "qwen3-omni",
    "--thinker-max-seq-len",
    "32768",
    "--thinker-tp-size",
    "2",
    "--gpu-thinker-tp",
    "0,1",
    "--gpu-talker",
    "1",
    "--gpu-code2wav",
    "1",
    "--thinker-mem-fraction-static",
    "0.40",
    "--talker-mem-fraction-static",
    "0.21",
]


def log(out: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(out / "driver.log", "a") as handle:
        handle.write(line + "\n")


def client(port: int, out: Path) -> None:
    from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
    from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig

    config = VideoEvalConfig(
        model="qwen3-omni",
        host="127.0.0.1",
        port=port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=CONCURRENCY,
        output_dir=str(out),
        repo_id=REPO_ID,
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        asr_concurrency=ASR_CONCURRENCY,
        disable_tqdm=True,
        timeout_s=500,
    )
    results = asyncio.run(run_videoamme_eval(config, compute_wer=False))
    (out / "videoamme.json").write_text(json.dumps(results, indent=1))


def score(port: int, out: Path) -> None:
    from benchmarks.tasks.asr import compute_text_audio_consistency_from_records

    per_sample = json.loads((out / "videoamme.json").read_text())["per_sample"]
    wer = compute_text_audio_consistency_from_records(
        per_sample,
        "en",
        "cuda:0",
        audio_dir=str(out / "audio"),
        asr_router_port=port,
        asr_concurrency=ASR_CONCURRENCY,
    )
    (out / "wer.json").write_text(json.dumps(wer, indent=1))


def serve(arm_tree: Path, log_path: Path) -> subprocess.Popen:
    env = {
        **pair_check.card_env(0, arm_tree),
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SGLANG_OMNI_STRICT_PORT": "1",
        "SGLANG_OMNI_STARTUP_TIMEOUT": "1800",
    }
    process = subprocess.Popen(
        [
            "numactl",
            f"--physcpubind={pair_check.CARD_CPUS[0]},{pair_check.CARD_CPUS[1]}",
            f"--membind={pair_check.NODE}",
            sys.executable,
            "-u",
            "examples/run_qwen3_omni_speech_server.py",
            "--model-path",
            MODEL,
            *SERVE_ARGS,
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        cwd=arm_tree,
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_path.with_suffix(".pgid").write_text(str(process.pid))
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited, see {log_path}")
        else:
            pass
        try:
            if (
                httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code
                == 200
            ):
                return process
            else:
                pass
        except httpx.HTTPError:
            pass
        time.sleep(5)
    stop(process)
    raise RuntimeError(f"server not healthy in 30 min, see {log_path}")


def stop(process: subprocess.Popen) -> None:
    pair_check.stop(0, process)
    pair_check.stop(1, process)


def run_arm(out: Path, arm: str) -> int:
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    log(out, f"{arm} serving")
    server = serve(arm_tree, arm_out / "server.log")
    try:
        log(out, f"{arm} healthy, client")
        rc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--client",
                "--port",
                str(PORT),
                "--out",
                str(arm_out),
            ],
            cwd=arm_tree,
            env={**os.environ, "PYTHONPATH": str(arm_tree)},
            stdout=open(arm_out / "client.log", "w"),
            stderr=subprocess.STDOUT,
        ).returncode
    finally:
        stop(server)
    server_log = (arm_out / "server.log").read_text(errors="replace")
    log(
        out,
        f"{arm} client rc {rc}, tp ranks {'tp_rank=0/2' in server_log and 'tp_rank=1/2' in server_log}",
    )
    return rc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", help="arm,arm")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.client:
        client(args.port, args.out)
        return
    elif args.score:
        score(args.port, args.out)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    codes = [run_arm(args.out, arm) for arm in arms]
    asr_tree = pair_check.tree(pair_check.ASR_TREE)
    asr_port = pair_check.ASR_PORT
    asr = pair_check.serve(
        0, asr_tree, pair_check.ASR_MODEL, asr_port, [], args.out / "asr_server.log"
    )
    summary = {}
    try:
        for arm, code in zip(arms, codes):
            if code != 0:
                continue
            else:
                pass
            rc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--score",
                    "--port",
                    str(asr_port),
                    "--out",
                    str(args.out / arm),
                ],
                cwd=asr_tree,
                env=pair_check.card_env(0, asr_tree),
                stdout=open(args.out / arm / "score.log", "w"),
                stderr=subprocess.STDOUT,
            ).returncode
            log(args.out, f"{arm} score rc {rc}")
            videoamme = json.loads((args.out / arm / "videoamme.json").read_text())
            wer = (
                json.loads((args.out / arm / "wer.json").read_text())["summary"]
                if rc == 0
                else None
            )
            summary[arm] = dict(
                summary=videoamme["summary"], speed=videoamme["speed"], wer=wer
            )
            log(
                args.out,
                f"{arm} accuracy {videoamme['summary'].get('accuracy')} failed "
                f"{videoamme['summary'].get('failed')} wer {wer}",
            )
    finally:
        pair_check.stop(0, asr)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(args.out, "done")


if __name__ == "__main__":
    main()
