"""Corpus WER of a bar arm's measured audio: every measured request's raw PCM (as Nari's client saves it, format from
its record) written as a wav under OUT/bench/audio with the prompt it sent as the target, a generated.json in the
SeedTTS benchmark's layout, then the benchmark's transcribe-only pass against a Qwen3-ASR server on the card (pair.py's
scoring without the similarity pass: CustomVoice has no reference).

usage: python3 bar_wer.py BAR_ARM_DIR --out OUT --tree TREE [--rate 1] [--seed 0] [--card 0] [--port 35000]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3_20261020"))
from pair import ASR_MODEL, BENCH_TIMEOUT_S, stop_group, wait_healthy  # noqa: E402

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--rate", default="1")
    parser.add_argument("--seed", default="0")
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--port", type=int, default=35000)
    args = parser.parse_args()
    run = args.arm / f"seed{args.seed}_rps{args.rate}"
    audio_dir = args.out / "bench" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for line in (run / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("phase") != "measurement" or not record.get("audio_success"):
            continue
        else:
            pass
        pcm_format = record["pcm_format"]
        samples = np.frombuffer((run / record["raw_audio_path"]).read_bytes(), dtype=np.int16)
        sample_id = f"{record['prompt_id']}_{record['phase_index']}"
        wav_path = audio_dir / f"{sample_id}.wav"
        soundfile.write(str(wav_path), samples, int(pcm_format["sample_rate_hz"]), subtype="PCM_16")
        generated.append({
            "sample_id": sample_id, "target_text": record["request_body"]["input"], "wav_path": str(wav_path),
            "is_success": True, "latency_s": 0.0,
            "audio_duration_s": round(samples.size / int(pcm_format["sample_rate_hz"]), 4),
        })
    (args.out / "bench" / "generated.json").write_text(json.dumps(generated, indent=1), encoding="utf-8")
    print(f"{args.arm.name} rate {args.rate}: {len(generated)} measured clips", flush=True)
    env = dict(os.environ, PYTHONPATH=args.tree, SGLANG_OMNI_STRICT_PORT="1")
    with open(args.out / "asr_serve.log", "w", encoding="utf-8") as serve_log:
        asr = subprocess.Popen(
            [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", ASR_MODEL, "--host", "127.0.0.1",
             "--port", str(args.port), "--asr.gpu", str(args.card)],
            cwd=args.tree, env=dict(env, MASTER_PORT=str(args.port + 50)), stdout=serve_log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        wait_healthy(args.port, asr)
        with open(args.out / "score--transcribe-only.log", "w", encoding="utf-8") as score_log:
            subprocess.run(
                [sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL, "--lang", "en",
                 "--skip-gpu-cleanup", "--output-dir", str(args.out / "bench"), "--transcribe-only",
                 "--use-existing-server", "--host", "127.0.0.1", "--port", str(args.port), "--asr-concurrency", "64"],
                cwd=args.tree, env=env, stdout=score_log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S,
                check=False,
            )
    finally:
        stop_group(asr, args.port)
    summary = json.loads((args.out / "bench" / "wer_results.json").read_text(encoding="utf-8"))["summary"]
    print(
        f"{args.arm.name} rate {args.rate}: WER {100 * summary['wer_corpus']:.2f} % over {summary['evaluated']} "
        f"(excluding samples over 50 %: {100 * summary['wer_below_50_corpus']:.2f} %, {summary['n_above_50_pct_wer']} over)",
        flush=True,
    )


if __name__ == "__main__":
    main()
