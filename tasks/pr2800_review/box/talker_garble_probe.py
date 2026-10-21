"""Talker garbles and run-ons on SeedTTS voice-clone prompts: HF's own generate, HF with zero audio rows, our server.

Run in the container from main's tree (its benchmarks package): cd <tree> && PYTHONPATH=<tree>
    python3 /workspace/box/tasks/pr2800_review/box/talker_garble_probe.py --arm hf --device cuda:0 --out <dir>
    python3 ... --arm hf_zero_rows --device cuda:1 --out <dir>
    python3 ... --arm server --port <port> --out <dir> [--concurrency 16]
The prompts are the 45 recorded garble and run-on prompts (talker_garble_prompts.json) plus --controls random others
of the SeedTTS EN corpus (seed 0). Each gets benchmark_omni_seedtts's voice-clone request as omni's preprocessor
builds it: one user turn, the reference audio placeholder then tts.py's instruction text, HF's chat template, the
reference audio at 16 kHz. The thinker is greedy (temperature 0) so only the talker varies; the talker samples with
the checkpoint's settings (temperature 0.9, top_k 50, top_p 1.0, repetition penalty 1.05). --samples outputs a prompt:
HF draws them as one batch under torch.manual_seed(prompt index); the server takes seeds 1 to K.
hf_zero_rows replays HF with the thinker's hidden zeroed at the talker prompt's multimodal rows, as omni's prefill
projects zeros there (talker_prefill.py). Writes <out>/generated.json and <out>/audio/*.wav in the benchmark's layout,
so `benchmark_omni_seedtts --transcribe-only --output-dir <out>` scores it as it scores a pass.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import random
import time
from pathlib import Path

import aiohttp
import soundfile
import torch
import torchaudio

from benchmarks.dataset.seedtts import load_seedtts_samples

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
META = "zhaochenyang20/seed-tts-eval-arrow"
HF_TALKER_MAX_FRAMES = 1024
PROMPTS = Path(__file__).resolve().parent / "talker_garble_prompts.json"


def instruction(sample) -> str:
    return (
        f'Listen to the audio above. The speaker is reading: "{sample.ref_text}". '
        f"Now please read the following text out loud in the same voice and style: "
        f"{sample.target_text}"
    )


def select_samples(controls: int) -> list:
    samples = load_seedtts_samples(META, None, split="en")
    recorded = set(json.loads(PROMPTS.read_text())["ids"])
    chosen = [s for s in samples if s.sample_id in recorded]
    others = [s for s in samples if s.sample_id not in recorded]
    random.Random(0).shuffle(others)
    return chosen + others[:controls]


def reference_audio_16k(path: str) -> torch.Tensor:
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(axis=1))
    return torchaudio.functional.resample(audio, rate, 16000) if rate != 16000 else audio


def run_hf(samples: list, samples_per_prompt: int, device: str, zero_rows: bool, out: Path) -> list[dict]:
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map=device).eval()
    if zero_rows:
        user_parts = model._get_talker_user_parts

        def zero_multimodal_rows(start, end, multimodal_mask, thinker_hidden, thinker_embed):
            return user_parts(start, end, multimodal_mask, torch.zeros_like(thinker_hidden), thinker_embed)

        model._get_talker_user_parts = zero_multimodal_rows
    else:
        pass
    generated = []
    for index, sample in enumerate(samples):
        messages = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": instruction(sample)}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audio = reference_audio_16k(sample.ref_audio).numpy()
        inputs = processor(text=[text] * samples_per_prompt, audio=[audio] * samples_per_prompt,
                           return_tensors="pt", padding=True).to(device)
        torch.manual_seed(index)
        start = time.perf_counter()
        with torch.inference_mode():
            _, wavs = model.generate(**inputs, speaker="Ethan", thinker_do_sample=False, thinker_max_new_tokens=256,
                                     talker_max_new_tokens=HF_TALKER_MAX_FRAMES)
        seconds = time.perf_counter() - start
        wavs = wavs if isinstance(wavs, list) else [wav.reshape(-1) for wav in wavs]
        for draw, wav in enumerate(wavs):
            generated.append(save(out, sample, draw, wav.float().cpu().numpy(), seconds / len(wavs)))
        print(f"{index + 1}/{len(samples)} {sample.sample_id}: {len(wavs)} outputs in {seconds:.1f} s, audio "
              f"{[round(len(w.reshape(-1)) / 24000, 1) for w in wavs]}", flush=True)
    return generated


def save(out: Path, sample, draw: int, wav, latency_s: float) -> dict:
    path = out / "audio" / f"{sample.sample_id}__{draw}.wav"
    soundfile.write(path, wav, 24000)
    return dict(sample_id=f"{sample.sample_id}__{draw}", target_text=sample.target_text, wav_path=str(path),
                is_success=True, latency_s=round(latency_s, 4), audio_duration_s=round(len(wav) / 24000, 4))


async def run_server(samples: list, samples_per_prompt: int, port: int, concurrency: int, out: Path) -> list[dict]:
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    gate = asyncio.Semaphore(concurrency)
    generated: list[dict] = []

    async def one(session: aiohttp.ClientSession, sample, draw: int) -> None:
        payload = {"model": "qwen3-omni", "messages": [{"role": "user", "content": instruction(sample)}],
                   "modalities": ["text", "audio"], "audio": {"format": "wav"}, "max_tokens": 256,
                   "temperature": 0.0, "stream": False, "seed": draw + 1, "audios": [sample.ref_audio]}
        async with gate:
            start = time.perf_counter()
            async with session.post(url, json=payload) as response:
                body = await response.json()
            latency = time.perf_counter() - start
        wav, _ = soundfile.read(io.BytesIO(base64.b64decode(body["choices"][0]["message"]["audio"]["data"])),
                                dtype="float32")
        generated.append(save(out, sample, draw, wav, latency))

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        await asyncio.gather(*(one(session, sample, draw) for sample in samples for draw in range(samples_per_prompt)))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("hf", "hf_zero_rows", "server"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--controls", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="first N prompts only, for a smoke run")
    args = parser.parse_args()
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    samples = select_samples(args.controls)
    samples = samples[: args.limit] if args.limit else samples
    print(f"{len(samples)} prompts ({len(samples) - args.controls} recorded, {args.controls} controls), "
          f"{args.samples} outputs each, arm {args.arm}", flush=True)
    if args.arm == "server":
        generated = asyncio.run(run_server(samples, args.samples, args.port, args.concurrency, args.out))
    else:
        generated = run_hf(samples, args.samples, args.device, args.arm == "hf_zero_rows", args.out)
    (args.out / "generated.json").write_text(json.dumps(generated, indent=1))
    print(f"wrote {len(generated)} outputs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
