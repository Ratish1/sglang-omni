"""One Qwen3-Omni benchmark arm against a running server, or its quality scoring after.

Arms are the CI arms (same eval functions, same prompts imported from the CI modules,
the talker arms with compute_wer=False) on the full corpus: no max_samples, warmup 1,
the given concurrency, timeout 500 s where the eval takes one (seed-tts has no timeout
field and keeps its 300 s).

gen:   generation against the omni server; results land in OUT/<arm>/
score: WER of a talker arm or of seed-tts against a Qwen3-ASR server on --asr-port; run
       after the omni server is stopped.
sim:   seed-tts speaker similarity on --device; run after the ASR server is stopped too.

usage: python run_bench.py gen --arm mmmu_talker --port 8000 --concurrency 16 --out DIR
       python run_bench.py score --arm mmmu_talker --asr-port 8100 --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

ARMS = (
    "seedtts_en",
    "mmmu",
    "mmmu_talker",
    "mmsu",
    "mmsu_talker",
    "videomme",
    "videomme_talker",
    "videoamme",
    "videoamme_talker",
    "speech_identity",
)
# speech_identity: the request seed also seeds the talker, whose default seed hashes the
# server's random request id; any fixed value works as long as both arms use it
SPEECH_IDENTITY_SEED = 1234
SPEECH_IDENTITY_TEXT_PROMPTS = (
    "Name three primary colors.",
    "Say good morning to a friend in one sentence.",
    "Count from one to five.",
)
MODEL = "qwen3-omni"
TIMEOUT_S = 500
SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
VIDEO_ARGS = dict(video_fps=2, video_max_frames=128, video_max_pixels=401408)


def mmsu_args(
    port: int, out: str, concurrency: int, talker: bool, max_samples: int | None
) -> argparse.Namespace:
    from tests.test_model.test_qwen3_omni_mmsu_talker_ci import MMSU_TTS_PROMPT

    return argparse.Namespace(
        base_url=None,
        host="127.0.0.1",
        port=port,
        model=MODEL,
        modalities="text+audio" if talker else "text",
        output_dir=out,
        max_samples=max_samples,
        task_names=None,
        categories=None,
        prompt=MMSU_TTS_PROMPT if talker else None,
        max_tokens=256 if talker else 32,
        temperature=0.0,
        warmup=1,
        max_concurrency=concurrency,
        request_rate=float("inf"),
        save_audio=talker,
        disable_tqdm=True,
        seed=None,
        repo_id=None,
        lang="en",
        asr_device="cuda:0",
        asr_concurrency=32,
        timeout_s=TIMEOUT_S,
        fingerprint=False,
    )


async def seeded_speech_requests(port: int, out: str, samples_per_input: int) -> dict:
    import base64
    import hashlib

    import aiohttp

    from benchmarks.dataset.mmmu import load_mmmu_samples
    from benchmarks.dataset.mmsu import load_mmsu_samples

    requests = [
        (f"text-{index}", {"messages": [{"role": "user", "content": prompt}]})
        for index, prompt in enumerate(SPEECH_IDENTITY_TEXT_PROMPTS)
    ]
    for sample in load_mmsu_samples(max_samples=samples_per_input):
        content = "Describe this audio in one sentence."
        requests.append(
            (
                f"audio-{sample.sample_id}",
                {
                    "messages": [{"role": "user", "content": content}],
                    "audios": [sample.audio_path],
                },
            )
        )
    for sample in load_mmmu_samples(max_samples=samples_per_input):
        content = "Describe this image in one sentence."
        requests.append(
            (
                f"image-{sample.sample_id}",
                {
                    "messages": [{"role": "user", "content": content}],
                    "images": list(sample.image_data_uris),
                },
            )
        )

    per_sample = []
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for sample_id, inputs in requests:
            payload = {
                "model": MODEL,
                **inputs,
                "modalities": ["text", "audio"],
                "audio": {"format": "wav"},
                "max_tokens": 64,
                "temperature": 0.0,
                "seed": SPEECH_IDENTITY_SEED,
                "stream": False,
            }
            async with session.post(url, json=payload) as response:
                body = await response.json()
            message = body["choices"][0]["message"]
            wav_bytes = base64.b64decode(message["audio"]["data"])
            per_sample.append(
                {
                    "id": sample_id,
                    "text": message.get("content"),
                    "audio_sha256": hashlib.sha256(wav_bytes).hexdigest(),
                    "audio_bytes": len(wav_bytes),
                }
            )
    results = {"per_sample": per_sample}
    (Path(out) / "speech_identity_results.json").write_text(
        json.dumps(results, indent=1)
    )
    return results


async def generate(
    arm: str, port: int, out: str, concurrency: int, max_samples: int | None
) -> dict:
    if arm == "speech_identity":
        return await seeded_speech_requests(port, out, max_samples or 5)
    talker = arm.endswith("_talker")
    if arm == "seedtts_en":
        from benchmarks.eval.benchmark_omni_seedtts import (
            OmniSeedttsBenchmarkConfig,
            run_omni_seedtts_benchmark,
        )

        config = OmniSeedttsBenchmarkConfig(
            model=MODEL,
            meta=SEEDTTS_META,
            host="127.0.0.1",
            port=port,
            lang="en",
            voice_clone=True,
            stream=True,
            output_dir=out,
            warmup=1,
            max_concurrency=concurrency,
            max_samples=max_samples,
            disable_tqdm=True,
        )
        return await run_omni_seedtts_benchmark(config)
    if arm.startswith("mmmu"):
        from benchmarks.eval.benchmark_omni_mmmu import MMMUEvalConfig, run_mmmu_eval
        from tests.test_model.test_qwen3_omni_mmmu_talker_ci import MMMU_TTS_PROMPT

        config = MMMUEvalConfig(
            model=MODEL,
            host="127.0.0.1",
            port=port,
            output_dir=out,
            max_concurrency=concurrency,
            warmup=1,
            disable_tqdm=True,
            timeout_s=TIMEOUT_S,
            enable_audio=talker,
            max_tokens=256 if talker else 2048,
            prompt_override=MMMU_TTS_PROMPT if talker else None,
            max_samples=max_samples,
        )
        return await run_mmmu_eval(config, compute_wer=False)
    if arm.startswith("mmsu"):
        from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu

        return await run_mmsu(
            mmsu_args(port, out, concurrency, talker, max_samples), compute_wer=False
        )
    from benchmarks.eval.benchmark_omni_videomme import VideoEvalConfig, run_video_eval

    config = VideoEvalConfig(
        model=MODEL,
        host="127.0.0.1",
        port=port,
        output_dir=out,
        max_concurrency=concurrency,
        warmup=1,
        disable_tqdm=True,
        timeout_s=TIMEOUT_S,
        enable_audio=talker,
        max_tokens=256,
        max_samples=max_samples,
        **VIDEO_ARGS,
    )
    if arm.startswith("videoamme"):
        from benchmarks.eval.benchmark_omni_videoamme import run_videoamme_eval

        return await run_videoamme_eval(config, compute_wer=False)
    samples = None
    if talker:
        from benchmarks.dataset.videomme import load_videomme_samples
        from tests.test_model.test_qwen3_omni_videomme_talker_ci import (
            SHORT_ANSWER_PROMPT,
        )

        samples = load_videomme_samples(
            repo_id=None, split="test", max_samples=max_samples
        )
        for sample in samples:
            sample.prompt = f"{sample.prompt}\n{SHORT_ANSWER_PROMPT}"
    return await run_video_eval(
        config,
        samples=samples,
        task_label="Video-MME",
        output_filename="videomme_results.json",
        audio_output_dir_default="results/videomme_audio",
        compute_wer=False,
    )


def score(arm: str, asr_port: int | None, out: str, device: str, sim: bool) -> dict:
    if arm == "seedtts_en":
        from benchmarks.eval.benchmark_omni_seedtts import (
            OmniSeedttsBenchmarkConfig,
            evaluate_generated_audio,
        )
        from benchmarks.tasks.tts import run_seedtts_similarity

        config = OmniSeedttsBenchmarkConfig(
            model=MODEL,
            meta=SEEDTTS_META,
            output_dir=out,
            lang="en",
            device=device,
            port=asr_port,
            asr_concurrency=32,
        )
        if sim:
            return {"similarity": run_seedtts_similarity(config)["summary"]}
        evaluate_generated_audio(config)
        return {
            "wer": json.loads((Path(out) / "wer_results.json").read_text())["summary"]
        }
    from benchmarks.tasks.asr import compute_text_audio_consistency_from_records

    results_file = {
        "mmmu_talker": "mmmu_results.json",
        "mmsu_talker": "mmsu_results.json",
        "videomme_talker": "videomme_results.json",
        "videoamme_talker": "videoamme_results.json",
    }[arm]
    per_sample = json.loads((Path(out) / results_file).read_text())["per_sample"]
    wer = compute_text_audio_consistency_from_records(
        per_sample,
        "en",
        device,
        audio_dir=str(Path(out) / "audio"),
        asr_router_port=asr_port,
        asr_concurrency=32,
    )
    (Path(out) / "wer_results.json").write_text(json.dumps(wer, indent=1))
    return {"wer": wer["summary"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("gen", "score", "sim"))
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument(
        "--out", required=True, help="run dir; the arm writes OUT/<arm>"
    )
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--asr-port", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-samples", type=int, help="first N samples; unset is the full corpus"
    )
    args = parser.parse_args()
    out = str(Path(args.out) / args.arm)
    Path(out).mkdir(parents=True, exist_ok=True)
    began = time.time()
    if args.mode == "gen":
        asyncio.run(
            generate(args.arm, args.port, out, args.concurrency, args.max_samples)
        )
    else:
        result = score(args.arm, args.asr_port, out, args.device, args.mode == "sim")
        name = "sim_summary.json" if args.mode == "sim" else "score_summary.json"
        (Path(out) / name).write_text(json.dumps(result, indent=1))
    print(
        json.dumps({"arm": args.arm, "mode": args.mode, "wall_s": time.time() - began})
    )


if __name__ == "__main__":
    main()
