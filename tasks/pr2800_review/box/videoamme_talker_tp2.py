"""Video-AMME talker in the CI's TP2 shape on the CI sample, one arm at a time on a card pair.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/videoamme_talker_tp2.py --chains "upl,c99@0+1"
    --score-card 0 --out /data/pr2800/<run>
Each arm serves the CI's FP8 thinker TP2 speech server (examples/run_qwen3_omni_speech_server.py: thinker on the pair,
talker and code2wav on its second card) and gets the CI's request shape: the first 10 videoamme-ci-50 samples, video
and audio in, text and audio out, greedy, 256 tokens, c16. Chains on different pairs run at the same time. After the
arms, one Qwen3-ASR server on the score card transcribes each arm's audio and scores it against the arm's text as the
CI does. The client and the scoring run in each tree's own
code (this file in its --client and --score modes). --model serves another checkpoint in the same shape, and
--concurrency sends at another concurrency. --passes N runs N client passes per boot, each written to
<out>/<arm>/p<index>/ and scored on its own. Writes <out>/<arm>/videoamme.json, <out>/<arm>/wer.json and
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
from concurrent.futures import ThreadPoolExecutor
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


def client(
    port: int, out: Path, seed: int | None, concurrency: int, sample_ids: list[str] | None
) -> None:
    from benchmarks.dataset.videomme import load_videoamme_samples
    from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval
    from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig

    config = VideoEvalConfig(
        model="qwen3-omni",
        host="127.0.0.1",
        port=port,
        max_samples=MAX_SAMPLES,
        max_tokens=MAX_TOKENS,
        max_concurrency=concurrency,
        output_dir=str(out),
        repo_id=REPO_ID,
        video_fps=2,
        video_max_frames=128,
        video_max_pixels=401408,
        enable_audio=True,
        asr_concurrency=ASR_CONCURRENCY,
        disable_tqdm=True,
        timeout_s=500,
        seed=seed,
    )
    samples = load_videoamme_samples(MAX_SAMPLES, repo_id=REPO_ID)
    if sample_ids:
        samples = [sample for sample in samples if sample.sample_id in sample_ids]
    else:
        pass
    results = asyncio.run(run_videoamme_eval(config, samples=samples, compute_wer=False))
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


def serve(
    arm_tree: Path,
    log_path: Path,
    cards: tuple[int, int],
    port: int,
    code2wav_dump_dir: Path | None,
    model: str,
) -> subprocess.Popen:
    env = {
        **pair_check.card_env(cards[0], arm_tree),
        "CUDA_VISIBLE_DEVICES": f"{cards[0]},{cards[1]}",
        "SGLANG_OMNI_STRICT_PORT": "1",
        "SGLANG_OMNI_STARTUP_TIMEOUT": "1800",
    }
    if code2wav_dump_dir is not None:
        env["PYTHONPATH"] = f"{BOX / 'code2wav_dump'}:{arm_tree}"
        env["CODE2WAV_DUMP_DIR"] = str(code2wav_dump_dir)
    else:
        pass
    process = subprocess.Popen(
        [
            "numactl",
            f"--physcpubind={pair_check.CARD_CPUS[cards[0]]},{pair_check.CARD_CPUS[cards[1]]}",
            f"--membind={pair_check.NODE}",
            sys.executable,
            "-u",
            "examples/run_qwen3_omni_speech_server.py",
            "--model-path",
            model,
            *SERVE_ARGS,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
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
                httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code
                == 200
            ):
                return process
            else:
                pass
        except httpx.HTTPError:
            pass
        time.sleep(5)
    stop(process, cards)
    raise RuntimeError(f"server not healthy in 30 min, see {log_path}")


def stop(process: subprocess.Popen, cards: tuple[int, int]) -> None:
    pair_check.stop(cards[0], process)
    pair_check.stop(cards[1], process)


def run_arm(
    out: Path,
    arm: str,
    cards: tuple[int, int],
    seed: int | None,
    is_code2wav_dump: bool,
    model: str,
    concurrency: int,
    passes: int,
) -> list[tuple[str, int]]:
    arm_out = out / arm
    arm_out.mkdir(parents=True, exist_ok=True)
    pass_names = [arm] if passes == 1 else [f"{arm}/p{index}" for index in range(passes)]
    arm_tree = pair_check.tree(pair_check.ARMS[arm][0])
    port = PORT + 200 * cards[0]
    log(out, f"{arm} cards {cards} serving")
    server = serve(
        arm_tree,
        arm_out / "server.log",
        cards,
        port,
        arm_out / "code2wav_dump" if is_code2wav_dump else None,
        model,
    )
    codes = []
    try:
        for pass_name in pass_names:
            pass_out = out / pass_name
            pass_out.mkdir(parents=True, exist_ok=True)
            log(out, f"{pass_name} client")
            rc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--client",
                    "--port",
                    str(port),
                    "--out",
                    str(pass_out),
                    "--concurrency",
                    str(concurrency),
                    *([] if seed is None else ["--seed", str(seed)]),
                ],
                cwd=arm_tree,
                env={**os.environ, "PYTHONPATH": str(arm_tree)},
                stdout=open(pass_out / "client.log", "w"),
                stderr=subprocess.STDOUT,
            ).returncode
            codes.append((pass_name, rc))
    finally:
        stop(server, cards)
    server_log = (arm_out / "server.log").read_text(errors="replace")
    log(
        out,
        f"{arm} client rcs {[rc for _, rc in codes]}, tp ranks "
        f"{'tp_rank=0/2' in server_log and 'tp_rank=1/2' in server_log}",
    )
    return codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chains",
        help="chains separated by ';', each 'arm,arm@a+b': its arms one after another on cards a and b, the chains "
        "at the same time",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("--score-card", type=int, default=0)
    parser.add_argument("--seed", type=int, help="every request's sampling seed, as a seeded client sends it")
    parser.add_argument(
        "--code2wav-dump",
        action="store_true",
        help="save every request's talker codes and served audio (code2wav_dump/sitecustomize.py)",
    )
    parser.add_argument("--model", default=MODEL, help="checkpoint every arm serves")
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--sample-ids", help="client mode: send only these comma-separated samples")
    parser.add_argument(
        "--passes", type=int, default=1, help="client passes per server boot, each in <arm>/p<index>"
    )
    args = parser.parse_args()
    if args.client:
        client(
            args.port,
            args.out,
            args.seed,
            args.concurrency,
            args.sample_ids.split(",") if args.sample_ids else None,
        )
        return
    elif args.score:
        score(args.port, args.out)
        return
    else:
        pass
    args.out.mkdir(parents=True, exist_ok=True)
    chains = []
    for chain in args.chains.split(";"):
        chain_arms, pair = chain.split("@")
        first_card, second_card = (int(card) for card in pair.split("+"))
        chains.append((chain_arms.split(","), (first_card, second_card)))

    def run_chain(chain_arms: list[str], cards: tuple[int, int]) -> list[tuple[str, int]]:
        return [
            pass_code
            for arm in chain_arms
            for pass_code in run_arm(
                args.out,
                arm,
                cards,
                args.seed,
                args.code2wav_dump,
                args.model,
                args.concurrency,
                args.passes,
            )
        ]

    with ThreadPoolExecutor(max_workers=len(chains)) as pool:
        chain_codes = list(pool.map(lambda chain: run_chain(*chain), chains))
    pass_codes = [pass_code for codes_of_chain in chain_codes for pass_code in codes_of_chain]
    asr_tree = pair_check.tree(pair_check.ASR_TREE)
    asr_port = pair_check.ASR_PORT + 1000 + 200 * args.score_card
    asr = pair_check.serve(
        args.score_card, asr_tree, pair_check.ASR_MODEL, asr_port, [], args.out / "asr_server.log"
    )
    summary = {}
    try:
        for pass_name, code in pass_codes:
            if code != 0:
                continue
            else:
                pass
            pass_out = args.out / pass_name
            rc = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--score",
                    "--port",
                    str(asr_port),
                    "--out",
                    str(pass_out),
                ],
                cwd=asr_tree,
                env=pair_check.card_env(args.score_card, asr_tree),
                stdout=open(pass_out / "score.log", "w"),
                stderr=subprocess.STDOUT,
            ).returncode
            log(args.out, f"{pass_name} score rc {rc}")
            videoamme = json.loads((pass_out / "videoamme.json").read_text())
            wer = (
                json.loads((pass_out / "wer.json").read_text())["summary"]
                if rc == 0
                else None
            )
            summary[pass_name] = dict(
                summary=videoamme["summary"], speed=videoamme["speed"], wer=wer
            )
            log(
                args.out,
                f"{pass_name} accuracy {videoamme['summary'].get('accuracy')} failed "
                f"{videoamme['summary'].get('failed')} rtf {videoamme['speed'].get('rtf_mean')} wer below 50 "
                f"{None if wer is None else wer['wer_below_50_corpus']} above 50 "
                f"{None if wer is None else wer['n_above_50_pct_wer']}",
            )
    finally:
        pair_check.stop(args.score_card, asr)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(args.out, "done")


if __name__ == "__main__":
    main()
