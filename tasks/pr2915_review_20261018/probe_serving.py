"""Check text and speech requests through the loaded Omni pipeline."""

import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path

import httpx


async def check_requests(arm: str) -> None:
    records: list[dict[str, str | int | float]] = []
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:18215", timeout=180
    ) as client:
        response = await client.get("/v1/models")
        response.raise_for_status()
        model = response.json()["data"][0]["id"]

        async def request(
            name: str, sampling: dict[str, int | float], speech: bool
        ) -> None:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Say exactly: The blue train arrives at noon.",
                        }
                    ],
                    "modalities": ["text", "audio"] if speech else ["text"],
                    "max_tokens": 40,
                    "talker_max_new_tokens": 100,
                    **sampling,
                },
            )
            response.raise_for_status()
            reply = response.json()
            message = reply["choices"][0]["message"]
            record: dict[str, str | int | float] = {
                "case": name,
                "status": response.status_code,
                "text": message.get("content") or "",
                "finish_reason": reply["choices"][0]["finish_reason"],
            }
            if speech:
                audio = base64.b64decode(message["audio"]["data"])
                assert len(audio) > 44
                record["audio_bytes"] = len(audio)
                record["audio_sha256"] = hashlib.sha256(audio).hexdigest()
            else:
                assert record["text"]
            records.append(record)

        await request("greedy_text", {"temperature": 0.0}, False)
        await request(
            "seeded_filtered_text",
            {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "seed": 2915},
            False,
        )
        await request("temperature_only_text", {"temperature": 0.7}, False)
        await asyncio.gather(
            *[
                request(
                    f"concurrent_filtered_text_{index}",
                    {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
                    False,
                )
                for index in range(4)
            ]
        )
        await request(
            "filtered_min_p_text",
            {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "min_p": 0.05},
            False,
        )
        await request(
            "seeded_filtered_speech",
            {"temperature": 0.7, "top_k": 20, "top_p": 0.8, "seed": 2915},
            True,
        )
        await request(
            "unseeded_filtered_speech",
            {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
            True,
        )
        audio_chunks = 0
        text_chunks = 0
        finished = False
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Say exactly: The blue train arrives at noon.",
                    }
                ],
                "modalities": ["text", "audio"],
                "stream": True,
                "max_tokens": 40,
                "talker_max_new_tokens": 100,
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.8,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line == "data: [DONE]":
                    finished = True
                elif line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    assert "error" not in chunk, chunk
                    for choice in chunk.get("choices", []):
                        delta = choice["delta"]
                        audio_chunks += bool(delta.get("audio", {}).get("data"))
                        text_chunks += bool(delta.get("content"))
                else:
                    pass
        assert finished and audio_chunks > 0 and text_chunks > 0
        records.append(
            {
                "case": "streamed_filtered_speech",
                "audio_chunks": audio_chunks,
                "text_chunks": text_chunks,
            }
        )
    records.sort(key=lambda record: record["case"])
    Path(f"/data/review2915/{arm}_serving.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    print(json.dumps(records, indent=2))


parser = argparse.ArgumentParser()
parser.add_argument("arm")
arguments = parser.parse_args()
asyncio.run(check_requests(arguments.arm))
