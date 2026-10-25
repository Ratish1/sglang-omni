"""Exercise image/video outputs and the downstream speech/cancellation lifecycle."""

import asyncio
import base64
import io
import json
from pathlib import Path

import httpx
import numpy as np
import soundfile


async def main() -> None:
    rows = []
    image = str(Path("tests/data/cars.jpg").resolve())
    video = str(Path("tests/data/draw.mp4").resolve())
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:13775", timeout=300
    ) as client:
        ready = await client.get("/health")
        ready.raise_for_status()

        async def chat(name, *, images=None, videos=None, speech=False):
            payload = {
                "model": "marksverdhei/Qwen3-Omni-30B-A3B-FP8",
                "messages": [
                    {
                        "role": "user",
                        "content": "Describe what you see in one short sentence.",
                    }
                ],
                "modalities": ["text", "audio"] if speech else ["text"],
                "max_tokens": 128,
                "temperature": 0,
                "seed": 3175,
            }
            if images:
                payload["images"] = images
            if videos:
                payload.update(
                    videos=videos,
                    video_max_frames=4,
                    video_fps=1,
                    video_max_pixels=100352,
                    use_audio_in_video=False,
                )
            if speech:
                payload.update(audio={"format": "wav"}, talker_max_new_tokens=128)
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
            assert message["content"].strip(), name
            row = {"case": name, "text": message["content"]}
            if speech:
                waveform, rate = soundfile.read(
                    io.BytesIO(base64.b64decode(message["audio"]["data"]))
                )
                assert rate == 24000 and waveform.size > 2400
                assert np.isfinite(waveform).all() and np.max(np.abs(waveform)) > 0.001
                row.update(samples=int(waveform.size), sample_rate=rate)
            rows.append(row)

        await chat("image", images=[image])
        await chat("cached_image", images=[image])
        await chat("mixed_image_video", images=[image], videos=[video])
        await asyncio.gather(
            chat("concurrent_images", images=[image, image]),
            chat("concurrent_video", videos=[video]),
            chat("concurrent_text"),
        )
        await chat("image_to_speech", images=[image], speech=True)
        cancelled = False
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Describe the image in great detail."}
                ],
                "images": [image],
                "modalities": ["text"],
                "max_tokens": 512,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    event = json.loads(line[6:])
                    choices = event.get("choices", [])
                    if choices and choices[0].get("delta", {}).get("content"):
                        cancelled = True
                        break
        assert cancelled
        await chat("after_cancel", images=[image])
        info = await client.get("/model_info")
        info.raise_for_status()
    result = {
        "requests": rows,
        "cancelled_after_content": cancelled,
        "after": info.json(),
    }
    Path("/data/review3175oct25/serving.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
