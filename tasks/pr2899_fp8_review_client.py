"""Check FP8-cache serving and prefill replay through the speech API."""

import asyncio
import base64
import json
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:18499"
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REFERENCE_URL = (
    "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/"
    "en/prompt-wavs/common_voice_en_10119832.wav"
)
OUTPUT_DIRECTORY = Path("/data/review2899-r3")


async def main() -> None:
    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        reference_response = await client.get(REFERENCE_URL)
        reference_response.raise_for_status()
        reference = "data:audio/wav;base64," + base64.b64encode(
            reference_response.content
        ).decode("ascii")
        for attempt in range(240):
            try:
                response = await client.get(f"{BASE_URL}/health", timeout=2.0)
                ready = response.status_code == 200
            except httpx.TransportError:
                ready = False
            if ready:
                break
            else:
                await asyncio.sleep(5)
        else:
            raise RuntimeError("FP8-cache server did not become ready")

        async def synthesize(index: int, *, stream: bool, text: str) -> dict[str, int]:
            async with client.stream(
                "POST",
                f"{BASE_URL}/v1/audio/speech",
                json={
                    "model": MODEL,
                    "input": text,
                    "references": [
                        {
                            "audio_path": reference,
                            "text": "We asked over twenty different people, and they all said it was his.",
                        }
                    ],
                    "stream": stream,
                    "response_format": "pcm",
                    "max_new_tokens": 48,
                },
            ) as response:
                response.raise_for_status()
                byte_count = 0
                chunk_count = 0
                async for chunk in response.aiter_bytes():
                    byte_count += len(chunk)
                    chunk_count += 1
                assert byte_count > 0 and byte_count % 2 == 0
                return {"index": index, "bytes": byte_count, "chunks": chunk_count}

        for scenario, concurrency, stream, text in (
            ("buffered", 1, False, "Please read this sentence with a steady voice."),
            ("cached", 1, True, "Please read this sentence with a steady voice."),
            ("burst", 16, True, "Please read this new sentence with a clear voice."),
            ("long", 1, True, "I am here. We go now. " * 170),
        ):
            before = await client.get(f"{BASE_URL}/model_info")
            before.raise_for_status()
            responses = await asyncio.gather(
                *(
                    synthesize(index, stream=stream, text=text)
                    for index in range(concurrency)
                )
            )
            after = await client.get(f"{BASE_URL}/model_info")
            after.raise_for_status()
            record = {
                "responses": responses,
                "before": before.json(),
                "after": after.json(),
            }
            (OUTPUT_DIRECTORY / f"fp8-{scenario}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            print(
                f"{scenario}: {len(responses)} nonempty HTTP 200 responses", flush=True
            )


if __name__ == "__main__":
    asyncio.run(main())
