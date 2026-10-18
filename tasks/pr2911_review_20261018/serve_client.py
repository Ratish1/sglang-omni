"""Exercise speech output and a real admin retraction through HTTP."""

import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path

import httpx


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retract", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{arguments.port}"
    reference_url = (
        "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/"
        "en/prompt-wavs/common_voice_en_10119832.wav"
    )
    async with httpx.AsyncClient(timeout=240.0, follow_redirects=True) as client:
        for attempt in range(240):
            try:
                response = await client.get(f"{base_url}/health", timeout=2.0)
                ready = response.status_code == 200
            except httpx.TransportError:
                ready = False
            if ready:
                break
            else:
                await asyncio.sleep(5)
        else:
            raise RuntimeError("Server did not become ready")
        reference_response = await client.get(reference_url)
        reference_response.raise_for_status()
        reference = "data:audio/wav;base64," + base64.b64encode(
            reference_response.content
        ).decode("ascii")

        async def synthesize(
            label: str, *, stream: bool, retract: bool = False
        ) -> None:
            payload = {
                "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                "input": (
                    "We will read this sentence clearly. Then we will read it once more. "
                    * (12 if retract else 1)
                ),
                "references": [
                    {
                        "audio_path": reference,
                        "text": "We asked over twenty different people, and they all said it was his.",
                    }
                ],
                "instructions": "Be calm. " * (80 if retract else 9),
                "stream": stream,
                "response_format": "pcm",
                "seed": 1234,
                "max_new_tokens": 256 if retract else 48,
            }
            before = await client.get(f"{base_url}/model_info")
            before.raise_for_status()
            pause_result = None
            continue_result = None
            audio = bytearray()
            async with client.stream(
                "POST", f"{base_url}/v1/audio/speech", json=payload
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    audio.extend(chunk)
                    if retract and pause_result is None and len(audio) >= 184320:
                        pause = await client.post(
                            f"{base_url}/pause_generation",
                            json={"stages": ["tts_engine"], "mode": "retract"},
                        )
                        pause.raise_for_status()
                        pause_result = pause.json()
                        resume = await client.post(
                            f"{base_url}/continue_generation",
                            json={"stages": ["tts_engine"]},
                        )
                        resume.raise_for_status()
                        continue_result = resume.json()
                    else:
                        pass
                headers = dict(response.headers)
            after = await client.get(f"{base_url}/model_info")
            after.raise_for_status()
            assert audio and len(audio) % 2 == 0
            record = {
                "bytes": len(audio),
                "seconds": len(audio) / 48000,
                "sha256": hashlib.sha256(audio).hexdigest(),
                "headers": headers,
                "pause": pause_result,
                "resume": continue_result,
                "before": before.json(),
                "after": after.json(),
            }
            (arguments.output / f"{label}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )
            print(
                f"{label}: HTTP 200, {len(audio)} PCM bytes; pause={pause_result}",
                flush=True,
            )

        await synthesize("buffered", stream=False)
        await synthesize("streaming-cached", stream=True)
        if arguments.retract:
            await synthesize("retract", stream=True, retract=True)
        else:
            pass


if __name__ == "__main__":
    asyncio.run(main())
