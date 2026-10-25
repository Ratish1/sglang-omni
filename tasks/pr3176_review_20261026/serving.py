"""Review-only default three-stage lifecycle check; no corpus-quality claims."""

import asyncio
import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf


async def main():
    root = Path("/data/review3176oct26")
    short = "/data/review3132oct25/reference.wav"
    audio, rate = sf.read(short, dtype="float32")
    tiled = np.concatenate((audio, np.zeros(6000, dtype=np.float32)))
    long = str(root / "long.wav")
    second = str(root / "second_long.wav")
    sf.write(long, np.resize(tiled, 2112001), rate)
    sf.write(second, np.resize(tiled, 983041), rate)
    transcript = "We asked over twenty different people, and they all said it was his."
    rows = []
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:13176", timeout=180
    ) as client:
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            try:
                ready = await client.get("/health", timeout=2)
                if ready.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("server did not become ready")
        before = (await client.get("/model_info")).json()

        async def speech(name, reference, stream, icl=True):
            payload = {
                "input": "The blue train arrives at noon.",
                "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "ref_audio": reference,
                "ref_text": transcript
                * (22 if reference == long else 11 if reference == second else 1),
                "x_vector_only_mode": not icl,
                "max_new_tokens": 100,
                "seed": 3176,
                "stream": stream,
                "response_format": "pcm" if stream else "wav",
            }
            start = time.perf_counter()
            if stream:
                parts = []
                async with client.stream(
                    "POST", "/v1/audio/speech", json=payload
                ) as reply:
                    reply.raise_for_status()
                    async for part in reply.aiter_bytes():
                        parts.append(part)
                raw = b"".join(parts)
                assert len(raw) % 2 == 0
                values = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
                sample_rate = 24000
                chunks = len(parts)
            else:
                reply = await client.post("/v1/audio/speech", json=payload)
                reply.raise_for_status()
                values, sample_rate = sf.read(io.BytesIO(reply.content))
                chunks = 1
            assert sample_rate == 24000 and values.size > 2400
            assert np.isfinite(values).all() and float(np.abs(values).max()) > 0.001
            rows.append(
                {
                    "case": name,
                    "samples": int(values.size),
                    "chunks": chunks,
                    "peak": float(np.abs(values).max()),
                    "elapsed_s": time.perf_counter() - start,
                }
            )
            print(json.dumps(rows[-1]), flush=True)

        await speech("short_cold", short, False)
        await speech("short_cached_stream", short, True)
        await asyncio.gather(
            speech("long_icl_stream", long, True),
            speech("long_xvector", long, False, icl=False),
            speech("second_long_icl", second, False),
            speech("short_during_long", short, True),
        )
        await speech("long_cached_stream", long, True)
        async with client.stream(
            "POST",
            "/v1/audio/speech",
            json={
                "input": "This request is cancelled after the first audio. " * 15,
                "ref_audio": short,
                "x_vector_only_mode": True,
                "stream": True,
                "response_format": "pcm",
                "max_new_tokens": 300,
            },
        ) as reply:
            reply.raise_for_status()
            async for part in reply.aiter_bytes():
                assert part
                break
        await speech("after_cancel", short, True)
        after = (await client.get("/model_info")).json()
    (root / "serving.json").write_text(
        json.dumps({"requests": rows, "before": before, "after": after}, indent=2)
    )
    print("SERVING CHECKS COMPLETE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
