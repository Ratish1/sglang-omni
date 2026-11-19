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

OUTPUT_DIRECTORY = Path(os.environ["PROBE_OUT"])
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
URL = "http://127.0.0.1:24080"
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
    seed: int, reference_audio: str, *, is_streaming: bool, disconnect: bool = False, timeout_s: float = 45
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
        timeout=timeout_s,
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
    mode = sys.argv[1]
    if mode == "default":
        config_path = "examples/configs/qwen3_tts_1_7b.yaml"
        num_streams = 16
    else:
        vocoder_overrides = {
            "sync": {"async_decode": False},
            "rollback": {"enable_stateful_codec_decoder": False},
            "fractional": {"codec_state_slots": 0.5},
        }[mode]
        config = {
            "config_cls": "Qwen3TTSPipelineConfig",
            "model_path": MODEL,
            "stages": {
                "tts_engine": {
                    "engine": {
                        "max_queued_requests": None,
                        "disable_cuda_graph": True,
                        "cuda_graph_backend_prefill": "disabled",
                    },
                },
                "vocoder": {"factory": vocoder_overrides},
            },
        }
        config_path = str(OUTPUT_DIRECTORY / f"config-{mode}.yaml")
        Path(config_path).write_text(json.dumps(config))
        num_streams = 3
    reference_audio = (
        "data:audio/wav;base64,"
        + base64.b64encode(
            Path("docs/_static/audio/ref_voice.wav").read_bytes()
        ).decode()
    )
    with (OUTPUT_DIRECTORY / f"server-{mode}.log").open("wb") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                "--model-path",
                MODEL,
                "--config",
                config_path,
                "--host",
                "127.0.0.1",
                "--port",
                "24080",
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
            if mode == "fractional":
                streamed = request_speech(1, reference_audio, is_streaming=True)
                info = httpx.get(f"{URL}/model_info", timeout=10)
                info.raise_for_status()
                summary = next(
                    row["data"]["codec_state"]
                    for row in info.json()["results"]
                    if row["stage"] == "vocoder"
                )
                assert summary["slots"] == 0 and summary["cuda_graphs"]["window"]["enabled"]
                try:
                    whole = request_speech(2, reference_audio, is_streaming=False, timeout_s=10)
                except httpx.ReadTimeout:
                    whole = {"timed_out": True}
                (OUTPUT_DIRECTORY / "serving-fractional.json").write_text(
                    json.dumps({"streamed": streamed, "whole": whole, "codec": summary}, indent=2)
                )
                return
            else:
                pass
            with ThreadPoolExecutor(max_workers=17) as workers:
                requests = [
                    workers.submit(
                        request_speech,
                        seed,
                        reference_audio,
                        is_streaming=seed != num_streams,
                        disconnect=seed == 0,
                    )
                    for seed in range(num_streams + 1)
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
            codec_state = next(
                row["data"]["codec_state"]
                for row in result["model_info"]["results"]
                if row["stage"] == "vocoder"
            )
            assert codec_state["enabled"] == (mode == "default")
            if mode == "default":
                assert codec_state["slots"] == 128
                assert codec_state["active_slots"] == 0
                assert codec_state["left_context_fallbacks"] == 0
            else:
                pass
            (OUTPUT_DIRECTORY / f"serving-{mode}.json").write_text(
                json.dumps(result, indent=2)
            )
            print(mode, "completed", len(burst), "burst requests and a follow-up", flush=True)
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
