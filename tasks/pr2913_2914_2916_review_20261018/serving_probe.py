"""Check real-checkpoint warmup, unequal prompt lengths and both audio transports."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import signal
import socket
import subprocess
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

from sglang_omni.serve.server_warmup import warmup_tone_wav_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--prefill-backend", choices=("full", "breakable"), required=True
    )
    options = parser.parse_args()
    options.run_directory.mkdir(parents=True, exist_ok=True)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    config_path = options.run_directory / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "config_cls": "Qwen3TTSPipelineConfig",
                "model_path": options.model,
                "stages": {
                    "tts_engine": {
                        "engine": {
                            "max_running_requests": 4,
                            "max_total_tokens": 16384,
                            "max_prefill_tokens": 1024,
                            "chunked_prefill_size": 512,
                            "cuda_graph_backend_prefill": options.prefill_backend,
                            "prefill_attention_backend": (
                                "fa4" if options.prefill_backend == "full" else "fa3"
                            ),
                        }
                    }
                },
            }
        )
    )
    base_url = f"http://127.0.0.1:{port}"
    reference = "data:audio/wav;base64," + base64.b64encode(
        warmup_tone_wav_bytes()
    ).decode("ascii")

    def request_audio(index: int) -> dict[str, int | str]:
        is_streaming = index % 2 == 1
        text = "The server reads this short sentence clearly. " * (index + 1)
        response = requests.post(
            f"{base_url}/v1/audio/speech",
            json={
                "model": options.model,
                "input": text,
                "ref_audio": reference,
                "ref_text": "A short tone.",
                "stream": is_streaming,
                "response_format": "pcm" if is_streaming else "wav",
                "seed": 2914,
                "max_new_tokens": 32,
            },
            timeout=300,
        )
        response.raise_for_status()
        if is_streaming:
            pcm = response.content
        else:
            with wave.open(io.BytesIO(response.content)) as recording:
                assert recording.getframerate() == 24000
                assert recording.getnchannels() == 1
                assert recording.getsampwidth() == 2
                pcm = recording.readframes(recording.getnframes())
        assert len(pcm) > 0 and len(pcm) % 2 == 0
        return {
            "request": index,
            "pcm_bytes": len(pcm),
            "sha256": hashlib.sha256(pcm).hexdigest(),
        }

    with (options.run_directory / "server.log").open("w") as server_log:
        process = subprocess.Popen(
            [
                "sgl-omni",
                "serve",
                "--model-path",
                options.model,
                "--config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with {process.returncode}")
                else:
                    pass
                try:
                    health = requests.get(f"{base_url}/health", timeout=3)
                except requests.RequestException:
                    health = None
                if health is not None and health.status_code == 200:
                    break
                elif time.monotonic() >= deadline:
                    raise TimeoutError("server did not become ready")
                else:
                    time.sleep(1)
            outcomes = [request_audio(0)]
            with ThreadPoolExecutor(max_workers=3) as executor:
                outcomes.extend(executor.map(request_audio, range(1, 4)))
            model_info = requests.post(
                f"{base_url}/model_info",
                json={"stages": ["tts_engine"], "timeout_s": 30},
                timeout=60,
            )
            model_info.raise_for_status()
            report = {
                "source_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "model": options.model,
                "prefill_backend": options.prefill_backend,
                "requests": outcomes,
                "model_info": model_info.json(),
            }
            (options.run_directory / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(report, indent=2))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            else:
                pass


if __name__ == "__main__":
    main()
else:
    pass
