"""Exercise a trained three-stage server without claiming corpus quality."""

import asyncio
import io
import json
from pathlib import Path

import httpx
import numpy as np
import soundfile


async def main() -> None:
    rows = []
    reference = "/data/review3133r2oct26/reference.wav"
    async with httpx.AsyncClient(timeout=180) as client:
        media = await client.get(
            "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
            follow_redirects=True,
        )
        media.raise_for_status()
        Path(reference).write_bytes(media.content)
    async with httpx.AsyncClient(base_url="http://127.0.0.1:13333", timeout=180) as client:
        ready = await client.get("/health")
        ready.raise_for_status()
        before = await client.get("/model_info")
        before.raise_for_status()

        async def speech(name, text, stream=False, seeded=True):
            payload = {
                "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
                "input": text,
                "ref_audio": reference,
                "ref_text": "We asked over twenty different people, and they all said it was his.",
                "max_new_tokens": 150,
                "stream": stream,
                "response_format": "pcm" if stream else "wav",
            }
            if seeded:
                payload["seed"] = 3133
            else:
                pass
            if stream:
                parts = []
                async with client.stream("POST", "/v1/audio/speech", json=payload) as reply:
                    reply.raise_for_status()
                    async for part in reply.aiter_bytes():
                        parts.append(part)
                raw = b"".join(parts)
                assert len(raw) % 2 == 0
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
                sample_rate = 24000
                chunks = len(parts)
            else:
                reply = await client.post("/v1/audio/speech", json=payload)
                reply.raise_for_status()
                audio, sample_rate = soundfile.read(io.BytesIO(reply.content))
                chunks = 1
            assert sample_rate == 24000 and audio.size > 2400
            assert np.isfinite(audio).all() and np.max(np.abs(audio)) > 0.001
            rows.append({"case": name, "samples": int(audio.size), "chunks": chunks,
                         "peak": float(np.max(np.abs(audio)))})

        text = "The blue train arrives at noon."
        await speech("cold_seeded", text)
        await speech("cached_seeded_stream", text, stream=True)
        await asyncio.gather(*[
            speech(f"mixed_{index}", text if index % 2 else f"Please read number {index} clearly.",
                   stream=index % 2 == 0, seeded=index < 2)
            for index in range(4)
        ])
        async with client.stream("POST", "/v1/audio/speech", json={
            "input": "This is a longer cancellation check. " * 20,
            "ref_audio": reference, "stream": True, "response_format": "pcm",
            "max_new_tokens": 300,
        }) as reply:
            reply.raise_for_status()
            async for part in reply.aiter_bytes():
                assert part
                break
        await speech("after_cancel", "The next request still completes.", stream=True)
        after = await client.get("/model_info")
        after.raise_for_status()
    result = {"requests": rows, "before": before.json(), "after": after.json()}
    Path("/data/review3133r2oct26/serving.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
