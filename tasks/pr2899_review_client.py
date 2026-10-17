"""Exercise default graph dispatch through the real Qwen3-TTS serving pipeline."""

import asyncio
import json
import sys
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:18299"
OUTPUT_DIRECTORY = Path("/data/review2899")


async def synthesize(
    client: httpx.AsyncClient,
    index: int,
    stream: bool,
    instructions: str | None = None,
) -> dict[str, int | bool]:
    async with client.stream(
        "POST",
        f"{BASE_URL}/v1/audio/speech",
        json={
            "input": f"Request {index}: please read this short sentence clearly.",
            "voice": "Ryan",
            "language": "English",
            "response_format": "pcm",
            "stream": stream,
            "max_new_tokens": 48,
            "instructions": instructions,
        },
    ) as response:
        response.raise_for_status()
        byte_count = 0
        transport_chunks = 0
        async for chunk in response.aiter_bytes():
            byte_count += len(chunk)
            transport_chunks += 1
        assert byte_count > 0 and byte_count % 2 == 0
        return {
            "index": index,
            "stream": stream,
            "status": response.status_code,
            "bytes": byte_count,
            "transport_chunks": transport_chunks,
        }


async def main(is_long_prefill: bool) -> None:
    async with httpx.AsyncClient(timeout=1200.0) as client:
        for attempt in range(120):
            try:
                health = await client.get(f"{BASE_URL}/health", timeout=2.0)
                ready = health.status_code == 200
            except httpx.TransportError:
                ready = False
            if ready:
                break
            else:
                await asyncio.sleep(5)
        else:
            raise RuntimeError("Server did not become ready")

        scenarios = (
            ("single_buffered", 1, False),
            ("single_streamed", 1, True),
            ("c16_new", 16, True),
            ("c16_repeat", 16, True),
            ("c64_new", 64, True),
            ("c64_repeat", 64, True),
        )
        if is_long_prefill:
            scenarios = (("long_instruction_prefill", 1, True),)
            instructions = "Speak with a calm and steady voice. " * 180
        else:
            instructions = None
        for label, concurrency, stream in scenarios:
            before = await client.get(f"{BASE_URL}/model_info")
            before.raise_for_status()
            responses = await asyncio.gather(
                *(
                    synthesize(client, index, stream, instructions)
                    for index in range(concurrency)
                )
            )
            after = await client.get(f"{BASE_URL}/model_info")
            after.raise_for_status()
            record = {
                "scenario": label,
                "responses": responses,
                "before": before.json(),
                "after": after.json(),
            }
            (OUTPUT_DIRECTORY / f"{label}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            print(f"{label}: {len(responses)} non-empty HTTP 200 responses", flush=True)


if __name__ == "__main__":
    asyncio.run(main(is_long_prefill=sys.argv[1:] == ["long"]))
