"""Exercise the stock speech endpoint with streaming and buffered requests together."""

import base64
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path

import httpx
import numpy as np

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
REFERENCE = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"


def request_audio(index: int, reference_audio: str) -> dict[str, int | float | bool]:
    text = "The library opens at nine in the morning. Please return the book on Friday."
    text *= 1 + index % 3
    streaming = index % 2 == 0
    payload = dict(
        model=MODEL,
        input=text,
        ref_audio=reference_audio,
        ref_text="We asked over twenty different people, and they all said it was his.",
        stream=streaming,
        response_format="pcm",
    )
    started = time.perf_counter()
    first = None
    audio = bytearray()
    with httpx.Client(timeout=300) as client:
        with client.stream(
            "POST", "http://127.0.0.1:8291/v1/audio/speech", json=payload
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if chunk:
                    if first is None:
                        first = time.perf_counter() - started
                    else:
                        pass
                    audio.extend(chunk)
                else:
                    pass
    assert len(audio) > 4800 and len(audio) % 2 == 0
    pcm = np.frombuffer(audio, dtype="<i2")
    assert np.max(np.abs(pcm.astype(np.int32))) > 0
    return dict(
        index=index,
        stream=streaming,
        pcm_bytes=len(audio),
        first_audio_s=first,
        completion_s=time.perf_counter() - started,
        audio_s=len(pcm) / 24000,
    )


def main() -> None:
    with urllib.request.urlopen(REFERENCE, timeout=60) as response:
        reference_audio = (
            "data:audio/wav;base64," + base64.b64encode(response.read()).decode()
        )
    results = [request_audio(0, reference_audio), request_audio(1, reference_audio)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        results += list(
            executor.map(
                lambda index: request_audio(index, reference_audio), range(2, 18)
            )
        )
    Path("/data/review2910/serve-client.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
