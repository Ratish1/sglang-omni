import argparse
import asyncio
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile

from benchmarks.eval.personaplex_parity import CASES, text_prompt_for
from sglang_omni.models.personaplex.prompts import DEFAULT_TEXT_PROMPT


async def request_reply(
    session: aiohttp.ClientSession,
    endpoint: str,
    assets: Path,
    case_name: str,
    seed: int,
    greedy: bool,
) -> tuple[float, bytes, str]:
    case = CASES[case_name]
    payload = {
        "model": "nvidia/personaplex-7b-v1",
        "messages": [{"role": "user", "content": ""}],
        "audios": [str(assets / case.input_wav)],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "seed": seed,
        "stage_params": {
            "preprocessing": {
                "voice": case.voice,
                "text_prompt": text_prompt_for(case, assets) or DEFAULT_TEXT_PROMPT,
            },
            "lm": {"seed": seed},
        },
    }
    if greedy:
        payload["temperature"] = 0.0
        payload["stage_params"]["lm"]["audio_temperature"] = 0.0
    else:
        pass
    started = time.perf_counter()
    async with session.post(endpoint, json=payload) as response:
        response.raise_for_status()
        reply = await response.json()
    elapsed = time.perf_counter() - started
    message = reply["choices"][0]["message"]
    return (
        elapsed,
        base64.b64decode(message["audio"]["data"], validate=True),
        message["content"],
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-revision", required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    timeout = aiohttp.ClientTimeout(total=600)
    measurements: list[dict[str, str | int | float]] = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for name in CASES:
            await request_reply(
                session, arguments.endpoint, arguments.assets, name, 42, False
            )
        for mode, repetitions in (("sampled", 3), ("greedy", 2)):
            for iteration in range(repetitions):
                for name, case in CASES.items():
                    seed = 42424242 + iteration if mode == "sampled" else 42424242
                    elapsed, encoded_wav, text = await request_reply(
                        session,
                        arguments.endpoint,
                        arguments.assets,
                        name,
                        seed,
                        mode == "greedy",
                    )
                    waveform, rate = soundfile.read(
                        io.BytesIO(encoded_wav), dtype="float32", always_2d=True
                    )
                    expected_samples = soundfile.info(
                        arguments.assets / case.input_wav
                    ).frames
                    assert rate == 24000
                    assert waveform.shape == (expected_samples, 1)
                    assert np.isfinite(waveform).all()
                    prefix = f"{mode}-{iteration}-{name}"
                    (arguments.output / f"{prefix}.wav").write_bytes(encoded_wav)
                    (arguments.output / f"{prefix}.txt").write_text(text)
                    measurements.append(
                        {
                            "arm": arguments.arm,
                            "source_revision": arguments.source_revision,
                            "case": name,
                            "mode": mode,
                            "iteration": iteration,
                            "seed": seed,
                            "latency_seconds": elapsed,
                            "output_seconds": expected_samples / rate,
                            "rtf": elapsed * rate / expected_samples,
                            "samples": expected_samples,
                            "pcm_float_sha256": hashlib.sha256(
                                waveform.tobytes()
                            ).hexdigest(),
                            "text": text,
                        }
                    )
                    (arguments.output / "results.json").write_text(
                        json.dumps(measurements, indent=2) + "\n"
                    )
                    print(json.dumps(measurements[-1]), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
