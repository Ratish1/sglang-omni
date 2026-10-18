"""Exercise short, exact-window, tail and concurrent audio responses through HTTP."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf


async def request_audio(
    session: aiohttp.ClientSession, frame_cap: int, stream: bool
) -> dict[str, int | bool]:
    payload = {
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": "Say: The green train leaves the station early in the morning.",
            }
        ],
        "modalities": ["text", "audio"],
        "audio": {"format": "pcm16" if stream else "wav"},
        "max_tokens": 80,
        "talker_max_new_tokens": frame_cap,
        "stream": stream,
    }
    async with session.post(
        "http://127.0.0.1:8919/v1/chat/completions", json=payload
    ) as response:
        response.raise_for_status()
        if stream:
            chunks = []
            finished = False
            async for line in response.content:
                if not line.startswith(b"data: "):
                    continue
                else:
                    pass
                value = line[6:].strip()
                if value == b"[DONE]":
                    finished = True
                    break
                else:
                    pass
                event = json.loads(value)
                assert "error" not in event, event
                for choice in event.get("choices", []):
                    audio = choice.get("delta", {}).get("audio")
                    if audio is not None and audio.get("data"):
                        chunks.append(base64.b64decode(audio["data"], validate=True))
                    else:
                        pass
            assert finished and chunks
            assert all(len(chunk) > 0 and len(chunk) % 2 == 0 for chunk in chunks)
            samples = sum(len(chunk) // 2 for chunk in chunks)
            chunk_count = len(chunks)
        else:
            body = await response.json()
            encoded = body["choices"][0]["message"]["audio"]["data"]
            waveform, sample_rate = sf.read(
                io.BytesIO(base64.b64decode(encoded)), dtype="float32"
            )
            assert (
                sample_rate == 24000
                and waveform.size > 0
                and np.isfinite(waveform).all()
            )
            samples, chunk_count = int(waveform.size), 1
        result = {
            "stream": stream,
            "frame_cap": frame_cap,
            "audio_samples": samples,
            "audio_chunks": chunk_count,
        }
        print(json.dumps(result), flush=True)
        return result


async def main() -> None:
    results = []
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        for stream in (True, False):
            for frame_cap in (1, 9, 10, 11, 20, 21):
                results.append(await request_audio(session, frame_cap, stream))
        results.extend(
            await asyncio.gather(
                *[
                    request_audio(session, 90 + index % 3, bool(index % 2))
                    for index in range(16)
                ]
            )
        )
        async with session.get("http://127.0.0.1:8919/health") as response:
            assert response.status == 200
    Path("/data/review2909/serving.json").write_text(json.dumps(results, indent=2))
    print(f"completed {len(results)} HTTP audio requests", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
