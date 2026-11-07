"""Replay saved requests byte-for-byte at a chosen concurrency."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import uuid
from pathlib import Path

import aiohttp
import soundfile


async def replay(
    base_url: str,
    request_directory: Path,
    request_ids: list[uuid.UUID],
    concurrency: int,
    output_directory: Path,
) -> None:
    output_directory.mkdir(exist_ok=False)
    limit = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:

        async def send(request_id: uuid.UUID) -> dict[str, str | float]:
            request_bytes = (
                request_directory / f"{request_id}.request.json"
            ).read_bytes()
            async with limit:
                async with session.post(
                    f"{base_url}/v1/chat/completions",
                    data=request_bytes,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    response_bytes = await response.read()
                    (output_directory / f"{request_id}.response.json").write_bytes(
                        response_bytes
                    )
                    response.raise_for_status()
                message = json.loads(response_bytes)["choices"][0]["message"]
                waveform = base64.b64decode(message["audio"]["data"], validate=True)
                (output_directory / f"{request_id}.wav").write_bytes(waveform)
                audio = soundfile.info(io.BytesIO(waveform))
                result = {
                    "request_id": str(request_id),
                    "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                    "waveform_sha256": hashlib.sha256(waveform).hexdigest(),
                    "audio_duration_seconds": audio.duration,
                    "text": message["content"],
                }
                print(json.dumps(result), flush=True)
                return result

        outcomes = await asyncio.gather(
            *(send(request_id) for request_id in request_ids), return_exceptions=True
        )
    results: list[dict[str, str | float]] = []
    failures: list[BaseException] = []
    for request_id, outcome in zip(request_ids, outcomes):
        if isinstance(outcome, BaseException):
            results.append(
                {
                    "request_id": str(request_id),
                    "error": str(outcome),
                    "error_type": type(outcome).__name__,
                }
            )
            failures.append(outcome)
        else:
            results.append(outcome)
    (output_directory / "summary.json").write_text(json.dumps(results, indent=2))
    if failures:
        raise BaseExceptionGroup("saved request replay failed", failures)
    else:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--request-directory", type=Path, required=True)
    parser.add_argument("--request-id", type=uuid.UUID, action="append", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(
        replay(
            arguments.base_url,
            arguments.request_directory,
            arguments.request_id,
            arguments.concurrency,
            arguments.output_directory,
        )
    )


if __name__ == "__main__":
    main()
