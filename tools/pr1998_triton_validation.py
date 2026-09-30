"""Compare PR #1998 startup and speech with identical attention settings."""

import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import time
import traceback
import wave
from pathlib import Path

import requests
import torch
import yaml

from benchmarks.benchmarker.utils import managed_omni_server

RUN_DIRECTORY = Path("/data/pr1998-triton-20260930")
REPOSITORY = Path("/workspace/sglang-omni")
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE = "20329946256a7e4ee3a7de22cfe215893c9c7147"
HEAD = "d0452883625d609c49d352d2c6ae9c6cc24c6acd"
ARMS = (
    ("base-triton", BASE, "triton", None, "breakable"),
    ("head-triton", HEAD, "triton", None, "full"),
    ("head-triton-breakable", HEAD, "triton", "breakable", "breakable"),
    ("head-fa3", HEAD, "fa3", None, "full"),
)

RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
os.environ["SGLANG_OMNI_STRICT_PORT"] = "1"
provenance = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "packages": {
        package: importlib.metadata.version(package)
        for package in ("sglang", "transformers", "flashinfer-python", "triton")
    },
    "gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"],
        text=True,
    ),
    "model": MODEL,
}
assert provenance["packages"]["sglang"] == "0.5.20", provenance
(RUN_DIRECTORY / "environment.json").write_text(json.dumps(provenance, indent=2))

for arm_index, arm in enumerate(ARMS):
    name, revision, attention, graph_override, expected_backend = arm
    directory = RUN_DIRECTORY / name
    directory.mkdir()
    subprocess.run(
        ["git", "checkout", "--detach", revision], cwd=REPOSITORY, check=True
    )
    engine = {
        "attention_backend": attention,
        "max_running_requests": 8,
        "cuda_graph_max_bs": 8,
        "cuda_graph_bs": [1, 2, 4, 8],
        "torch_compile_max_bs": 8,
    }
    if graph_override is not None:
        engine["cuda_graph_backend_prefill"] = graph_override
    else:
        pass
    configuration = {
        "config_cls": "Qwen3TTSPipelineConfig",
        "model_path": MODEL,
        "enable_deterministic_inference": True,
        "stages": {"tts_engine": {"engine": engine}},
    }
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(configuration, sort_keys=False))
    port = 19098 + arm_index
    url = f"http://127.0.0.1:{port}"
    result = {"arm": name, "revision": revision, "healthy": False, "status": "running"}
    started = time.monotonic()
    print(f"Starting {name} at {revision}", flush=True)
    try:
        with managed_omni_server(
            model_path=MODEL,
            port=port,
            host="127.0.0.1",
            log_file=directory / "server.log",
            server_config=str(config_path),
            timeout=1200,
            wait_for_gpu_release=False,
        ):
            result["healthy"] = True
            result["startup_seconds"] = time.monotonic() - started
            for streaming in (False, True):
                response = requests.post(
                    f"{url}/v1/audio/speech",
                    json={
                        "model": MODEL,
                        "input": "This request checks the prefill graph and speech output.",
                        "voice": "Ryan",
                        "language": "English",
                        "response_format": "pcm" if streaming else "wav",
                        "stream": streaming,
                        "seed": 123456,
                        "max_new_tokens": 128,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                mode = "streaming" if streaming else "buffered"
                extension = "pcm" if streaming else "wav"
                (directory / f"{mode}.{extension}").write_bytes(response.content)
                if streaming:
                    pcm = response.content
                    assert (
                        "audio/pcm" in response.headers["content-type"]
                    ), response.headers
                else:
                    with wave.open(io.BytesIO(response.content), "rb") as audio:
                        audio_format = (
                            audio.getnchannels(),
                            audio.getframerate(),
                            audio.getsampwidth(),
                        )
                        assert audio_format == (1, 24000, 2), audio_format
                        pcm = audio.readframes(audio.getnframes())
                assert len(pcm) > 0 and len(pcm) % 2 == 0
                result[mode] = {
                    "http_status": response.status_code,
                    "pcm_bytes": len(pcm),
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "headers": dict(response.headers),
                }
            response = requests.post(
                f"{url}/model_info",
                json={"stages": ["tts_engine"], "timeout_s": 30},
                timeout=60,
            )
            response.raise_for_status()
            model_info = response.json()
            (directory / "model-info.json").write_text(json.dumps(model_info, indent=2))
            stages = [
                stage
                for stage in model_info["stages"]
                if stage["stage"] == "tts_engine"
            ]
            assert len(stages) == 1 and stages[0]["success"], model_info
            graph = stages[0]["data"]["prefill_cuda_graph"]
            assert graph["backend"] == expected_backend, graph
            assert graph["replay_count"] > 0, graph
            result["graph"] = graph
            result["status"] = "passed"
    except (
        RuntimeError,
        TimeoutError,
        requests.RequestException,
        AssertionError,
    ) as error:
        result["status"] = "failed"
        result["error"] = str(error)
        (directory / "error.txt").write_text(traceback.format_exc())
    finally:
        result["total_seconds"] = time.monotonic() - started
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        print(
            f"Completed {name}: healthy={result['healthy']} status={result['status']}",
            flush=True,
        )

print("All four arms finished. Inspect each result and server log.", flush=True)
