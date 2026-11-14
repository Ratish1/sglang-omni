"""Synthetic per-stage regimes against a served Qwen3-Omni, each in its own nsys window.

usage: python3 synth_regimes.py --port P --model-path M --out DIR [--regimes a,b] [--nsys-session S] [--media-dir D]

Prompts are random words that are single tokens of the thinker's tokenizer, drawn per request so no two requests
share a prefix past the chat template; the template's own tokens are measured once, so every text prompt has an
exact token count, checked against the server's usage. The server runs with synth_eos_probe so every thinker
request decodes exactly max_tokens. Speech regimes stream audio and record first text, first audio and each chunk.
Audio, image and video inputs are generated per request at fixed shapes. Each regime runs warm requests first,
then its measured requests at its concurrency inside one nsys window (start and stop around the measured set);
<out>/<regime>/ gets requests.json and window.txt. Traces are exported by the caller after the server stops.
"""

import argparse
import asyncio
import json
import math
import random
import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import numpy

from benchmarks.tasks.tts import _collect_chat_streaming_audio


@dataclass(frozen=True)
class Regime:
    name: str
    kind: str
    words: int
    max_tokens: int
    concurrency: int
    requests: int
    media_seconds: float = 0.0
    frames: int = 0


REGIMES = (
    Regime("prefill_8k", "text", 8192, 1, 1, 6),
    Regime("prefill_32k", "text", 32000, 1, 1, 3),
    Regime("decode_c1", "text", 1, 512, 1, 2),
    Regime("decode_c16", "text", 1, 512, 16, 16),
    Regime("decode_c64", "text", 1, 512, 64, 64),
    Regime("decode_8k_c16", "text", 8192, 256, 16, 16),
    Regime("speech_c1", "speech", 1, 128, 1, 3),
    Regime("speech_c16", "speech", 1, 128, 16, 32),
    Regime("speech_c64", "speech", 1, 128, 64, 64),
    Regime("speech_prefill_8k", "speech", 8192, 32, 1, 3),
    Regime("audio_30s", "audio", 1, 1, 1, 4, media_seconds=30.0),
    Regime("audio_1s_c16", "audio", 1, 1, 16, 32, media_seconds=1.0),
    Regime("image_c16", "image", 1, 1, 16, 32),
    Regime("video_128f", "video", 1, 1, 1, 3, frames=128),
    Regime("video_128f_c4", "video", 1, 1, 4, 8, frames=128),
)
VIDEO_FPS = 2
VIDEO_MAX_PIXELS = 401408
VIDEO_SIDE = 632
AUDIO_RATE = 16000


def single_token_words(model_path: str) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    words = []
    for token in tokenizer.get_vocab():
        if (
            token.startswith("Ġ")
            and token[1:].isalpha()
            and token[1:].islower()
            and 3 <= len(token) <= 8
        ):
            word = token[1:]
            if len(tokenizer.encode(f" {word}", add_special_tokens=False)) == 1:
                words.append(word)
            else:
                pass
        else:
            pass
    words.sort()
    joined = " " + " ".join(words[:200])
    assert (
        len(tokenizer.encode(joined, add_special_tokens=False)) == 200
    ), "words merge across spaces"
    return words


def prompt_text(words: list[str], count: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(words) for _ in range(count))


def write_audio(path: Path, seconds: float, seed: int) -> None:
    rng = numpy.random.default_rng(seed)
    t = numpy.arange(int(seconds * AUDIO_RATE)) / AUDIO_RATE
    tone = 0.2 * numpy.sin(
        2
        * math.pi
        * (180 + 40 * rng.random())
        * t
        * (1 + 0.1 * numpy.sin(2 * math.pi * 0.5 * t))
    )
    signal = numpy.clip(tone + 0.02 * rng.standard_normal(t.size), -1, 1)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(AUDIO_RATE)
        out.writeframes((signal * 32767).astype("<i2").tobytes())


def frame_image(index: int, seed: int, width: int, height: int) -> numpy.ndarray:
    y, x = numpy.mgrid[0:height, 0:width]
    phase = seed * 0.37 + index * 0.11
    red = 127 + 120 * numpy.sin(x / 37.0 + phase)
    green = 127 + 120 * numpy.sin(y / 23.0 - phase)
    blue = 127 + 120 * numpy.sin((x + y) / 51.0 + 2 * phase)
    image = numpy.stack([red, green, blue], axis=-1).astype(numpy.uint8)
    cx, cy = int((index * 7 + seed * 13) % width), int((index * 5 + seed * 11) % height)
    image[max(0, cy - 40) : cy + 40, max(0, cx - 40) : cx + 40] = (
        seed * 41 % 255,
        255 - index % 255,
        90,
    )
    return image


def write_image(path: Path, seed: int) -> None:
    from PIL import Image

    Image.fromarray(frame_image(0, seed, 1280, 720)).save(path)


def write_video(path: Path, frames: int, seed: int) -> None:
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=VIDEO_FPS)
        stream.width = VIDEO_SIDE
        stream.height = VIDEO_SIDE
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(
                frame_image(index, seed, VIDEO_SIDE, VIDEO_SIDE), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def build_payload(
    regime: Regime, words: list[str], overhead: int, seed: int, media_dir: Path
) -> dict:
    # note (ratish): a long prompt totals exactly regime.words tokens with the template; a short one is one word.
    count = regime.words - overhead if regime.words > overhead else 1
    payload = {
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": prompt_text(words, count, seed)}],
        "max_tokens": regime.max_tokens,
        "temperature": 0.0,
        "modalities": ["text"],
        "stream": False,
    }
    if regime.kind == "speech":
        payload.update(
            modalities=["text", "audio"], audio={"format": "wav"}, stream=True
        )
    elif regime.kind == "audio":
        path = media_dir / f"{regime.name}_{seed}.wav"
        write_audio(path, regime.media_seconds, seed)
        payload["audios"] = [str(path)]
    elif regime.kind == "image":
        path = media_dir / f"{regime.name}_{seed}.png"
        write_image(path, seed)
        payload["images"] = [str(path)]
    elif regime.kind == "video":
        path = media_dir / f"{regime.name}_{seed}.mp4"
        write_video(path, regime.frames, seed)
        payload.update(
            videos=[str(path)],
            video_fps=VIDEO_FPS,
            video_max_frames=regime.frames,
            video_max_pixels=VIDEO_MAX_PIXELS,
        )
    else:
        pass
    return payload


async def send(session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
    record = {"start": time.time(), "ok": False}
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as response:
            if payload["stream"]:
                chunks: list[bytes] = []
                chunk_times: list[float] = []
                first_text: list[float] = []
                usage: dict = {}
                pcm_format = None
                async for raw in response.content:
                    line = raw.decode("utf-8").strip()
                    if line:
                        pcm_format = _collect_chat_streaming_audio(
                            line, chunks, pcm_format, usage, chunk_times, first_text
                        )
                    else:
                        pass
                record["usage"] = usage
                record["first_text_s"] = first_text[0] - start if first_text else None
                record["chunk_s"] = [t - start for t in chunk_times]
                record["audio_bytes"] = sum(len(c) for c in chunks)
            else:
                body = await response.json()
                record["usage"] = body.get("usage") or {}
                record["text"] = (body["choices"][0]["message"].get("content") or "")[
                    :80
                ]
            record["status"] = response.status
            record["ok"] = response.status == 200
    except Exception as exc:
        record["error"] = repr(exc)
    record["latency_s"] = time.perf_counter() - start
    return record


async def run_set(url: str, payloads: list[dict], concurrency: int) -> list[dict]:
    limit = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def one(payload: dict) -> dict:
            async with limit:
                return await send(session, url, payload)

        return await asyncio.gather(*(one(p) for p in payloads))


def nsys(*args: str) -> None:
    subprocess.run(
        ["nsys", *args], check=False, stdout=sys.stderr, stderr=subprocess.STDOUT
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regimes", default="")
    parser.add_argument("--nsys-session", default="")
    parser.add_argument("--media-dir", type=Path, default=None)
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    media_dir = args.media_dir or args.out / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    selected = set(filter(None, args.regimes.split(",")))
    regimes = [r for r in REGIMES if not selected or r.name in selected]

    words = single_token_words(args.model_path)
    probe = {
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": prompt_text(words, 10, 1)}],
        "max_tokens": 1,
        "temperature": 0.0,
        "modalities": ["text"],
        "stream": False,
    }
    calibration = asyncio.run(run_set(url, [probe], 1))[0]
    overhead = int(calibration["usage"]["prompt_tokens"]) - 10
    print(
        f"single token words {len(words)}, chat template overhead {overhead} tokens",
        flush=True,
    )

    for index, regime in enumerate(regimes):
        out = args.out / regime.name
        out.mkdir(parents=True, exist_ok=True)
        base = 1000 * (index + 1)
        warm_count = min(regime.concurrency, 2)
        warm = [
            build_payload(regime, words, overhead, base + 900 + n, media_dir)
            for n in range(warm_count)
        ]
        measured = [
            build_payload(regime, words, overhead, base + n, media_dir)
            for n in range(regime.requests)
        ]
        warm_records = asyncio.run(run_set(url, warm, regime.concurrency))
        if args.nsys_session:
            nsys(
                "start",
                f"--session={args.nsys_session}",
                "--sample=none",
                "--cpuctxsw=none",
                "--gpuctxsw=true",
                f"--output={out / 'trace'}",
                "--force-overwrite=true",
            )
        else:
            pass
        window_start = time.time()
        records = asyncio.run(run_set(url, measured, regime.concurrency))
        window_end = time.time()
        if args.nsys_session:
            nsys("stop", f"--session={args.nsys_session}")
        else:
            pass
        (out / "window.txt").write_text(
            f"{window_start} {window_end}\n", encoding="utf-8"
        )
        (out / "requests.json").write_text(
            json.dumps(
                {
                    "regime": regime.__dict__,
                    "overhead": overhead,
                    "warm": warm_records,
                    "measured": records,
                }
            ),
            encoding="utf-8",
        )
        ok = [r for r in records if r["ok"]]
        prompts = sorted({r["usage"].get("prompt_tokens") for r in ok})
        completions = sorted({r["usage"].get("completion_tokens") for r in ok})
        latency = sorted(r["latency_s"] for r in ok)
        print(
            f"{regime.name:20s} ok {len(ok)}/{len(records)} warm ok {sum(r['ok'] for r in warm_records)}/{len(warm)} "
            f"prompt tokens {prompts[:4]} completion tokens {completions[:4]} "
            f"latency p50 {latency[len(latency) // 2] if latency else float('nan'):.3f} s "
            f"window {window_end - window_start:.1f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
