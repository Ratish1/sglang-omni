"""Exercise the default Qwen3-TTS arena through the HTTP pipeline on an H100."""

import base64
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

OUTPUT_DIRECTORY = Path("/data/pr4077-ed170038")
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
URL = "http://127.0.0.1:24079"
INPUT = "The weather cleared just before we reached the station. We took the long way home, stopping at the bakery to buy fresh bread."
REFERENCE_TRANSCRIPT = "It was the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."


def wait_healthy(server: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 850
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with {server.returncode}")
        else:
            pass
        try:
            if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
                return
            else:
                pass
        except httpx.RequestError:
            pass
        time.sleep(1)
    raise TimeoutError("server never became healthy")


def request_speech(
    seed: int, reference_audio: str, *, is_streaming: bool, disconnect: bool = False
) -> dict[str, int | bool | float]:
    start_time = time.monotonic()
    first_byte_time = None
    bytes_received = 0
    with httpx.stream(
        "POST",
        f"{URL}/v1/audio/speech",
        json={
            "model": MODEL,
            "input": INPUT,
            "language": "English",
            "references": [
                {"audio_path": reference_audio, "text": REFERENCE_TRANSCRIPT}
            ],
            "response_format": "pcm",
            "stream": is_streaming,
            "seed": seed,
            "max_new_tokens": 64,
        },
        timeout=45,
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.read()!r}")
        else:
            pass
        for audio in response.iter_bytes():
            if first_byte_time is None:
                first_byte_time = time.monotonic()
            else:
                pass
            bytes_received += len(audio)
            if disconnect:
                break
            else:
                pass
    assert first_byte_time is not None
    assert bytes_received > 0 and bytes_received % 2 == 0
    return {
        "seed": seed,
        "is_streaming": is_streaming,
        "disconnect": disconnect,
        "samples": bytes_received // 2,
        "first_byte_s": first_byte_time - start_time,
        "total_s": time.monotonic() - start_time,
    }


def main() -> None:
    reference_audio = (
        "data:audio/wav;base64,"
        + base64.b64encode(
            Path("docs/_static/audio/ref_voice.wav").read_bytes()
        ).decode()
    )
    with (OUTPUT_DIRECTORY / "server.log").open("wb") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                "--model-path",
                MODEL,
                "--config",
                "examples/configs/qwen3_tts_1_7b.yaml",
                "--host",
                "127.0.0.1",
                "--port",
                "24079",
                "--tts_engine.gpu",
                "0",
                "--vocoder.gpu",
                "0",
                "--vocoder.factory.incremental_codec_compile",
                "false",
                "--vocoder.factory.suppress_bootstrap_silence",
                "false",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_healthy(server)
            print("healthy", flush=True)
            with ThreadPoolExecutor(max_workers=17) as workers:
                requests = [
                    workers.submit(
                        request_speech,
                        seed,
                        reference_audio,
                        is_streaming=seed != 16,
                        disconnect=seed == 0,
                    )
                    for seed in range(17)
                ]
                burst = [future.result() for future in requests]
            followup = request_speech(19, reference_audio, is_streaming=True)
            response = httpx.get(f"{URL}/model_info", timeout=15)
            response.raise_for_status()
            result = {
                "burst": burst,
                "followup": followup,
                "model_info": response.json(),
            }
            (OUTPUT_DIRECTORY / "serving.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            else:
                pass


if __name__ == "__main__":
    main()
