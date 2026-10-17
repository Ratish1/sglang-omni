"""Check CosyVoice streaming mechanics and retain a short runtime trace."""

import asyncio
import base64
import json
import sys
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:18407"
REFERENCE_URL = (
    "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/"
    "en/prompt-wavs/common_voice_en_10119832.wav"
)
OUTPUT_DIRECTORY = Path("/data/review2907")


async def synthesize(
    client: httpx.AsyncClient, reference: str, index: int, stream: bool
) -> dict[str, int | bool]:
    async with client.stream(
        "POST",
        f"{BASE_URL}/v1/audio/speech",
        json={
            "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
            "input": f"Please read sentence number {index} with a clear and steady voice.",
            "ref_audio": reference,
            "ref_text": "We asked over twenty different people, and they all said it was his.",
            "stream": stream,
            "response_format": "pcm",
            "max_new_tokens": 96,
        },
    ) as response:
        response.raise_for_status()
        byte_count = 0
        chunk_count = 0
        async for chunk in response.aiter_bytes():
            byte_count += len(chunk)
            chunk_count += 1
        assert byte_count > 0 and byte_count % 2 == 0
        return {
            "index": index,
            "stream": stream,
            "status": response.status_code,
            "bytes": byte_count,
            "transport_chunks": chunk_count,
        }


async def main(label: str) -> None:
    async with httpx.AsyncClient(timeout=1200.0, follow_redirects=True) as client:
        reference_response = await client.get(REFERENCE_URL)
        reference_response.raise_for_status()
        reference = "data:audio/wav;base64," + base64.b64encode(
            reference_response.content
        ).decode("ascii")
        for attempt in range(240):
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

        for scenario, concurrency, stream in (
            ("buffered", 1, False),
            ("streaming", 1, True),
            ("c16", 16, True),
            ("c16_repeat", 16, True),
            ("c64", 64, True),
        ):
            before = await client.get(f"{BASE_URL}/model_info")
            before.raise_for_status()
            if scenario == "c16":
                profile = await client.post(
                    f"{BASE_URL}/start_profile",
                    json={
                        "run_id": f"review2907-{label}",
                        "trace_path_template": str(
                            OUTPUT_DIRECTORY / f"{label}-{{stage}}-{{rank}}.json"
                        ),
                        "event_dir": str(OUTPUT_DIRECTORY / f"{label}-events"),
                    },
                )
                profile.raise_for_status()
            else:
                pass
            responses = await asyncio.gather(
                *(
                    synthesize(client, reference, index, stream)
                    for index in range(concurrency)
                )
            )
            if scenario == "c16":
                profile = await client.post(
                    f"{BASE_URL}/stop_profile", json={"run_id": f"review2907-{label}"}
                )
                profile.raise_for_status()
            else:
                pass
            after = await client.get(f"{BASE_URL}/model_info")
            after.raise_for_status()
            record = {
                "scenario": scenario,
                "responses": responses,
                "before": before.json(),
                "after": after.json(),
            }
            (OUTPUT_DIRECTORY / f"{label}-{scenario}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            print(
                f"{scenario}: {len(responses)} nonempty HTTP 200 responses", flush=True
            )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
