"""Run checkpoint serving contracts after the assigned device becomes free."""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import yaml
from websockets.sync.client import connect

from sglang_omni.utils.json import JsonValue

DESTINATION = Path("/data/pr3005-614af78a")
PORT = 30329
BASE_CHECKPOINT = (
    "FireRedTeam/FireRedTTS3-Base-HF@e4133ef7fc8edb61883da42c584deb72f28c8c5a"
)
INSTRUCT_CHECKPOINT = (
    "FireRedTeam/FireRedTTS3-Instruct-HF@1fa25f92a13679e6f1a52a867197a6d3b4ebfef7"
)
REFERENCE = "docs/_static/audio/ref_voice.wav"
TRANSCRIPT = "It was the night before my birthday. Hooray! It’s almost here! It may not be a holiday, but it’s the best day of the year."


def request_audio(label: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    started = time.perf_counter()
    with httpx.Client(timeout=180) as client:
        response = client.post(f"http://127.0.0.1:{PORT}/v1/audio/speech", json=payload)
    observation = {
        "label": label,
        "status": response.status_code,
        "seconds": time.perf_counter() - started,
    }
    if response.status_code == 200:
        if payload.get("response_format") == "pcm":
            observation["samples"] = len(response.content) // 2
        else:
            with wave.open(io.BytesIO(response.content)) as audio:
                observation["samples"] = audio.getnframes()
                observation["sample_rate"] = audio.getframerate()
        observation["headers"] = {
            key: value
            for key, value in response.headers.items()
            if key.startswith("x-")
        }
    else:
        observation["error"] = response.text
    print(json.dumps(observation), flush=True)
    assert response.status_code == 200, observation
    assert observation["samples"] > 0, observation
    return observation


def websocket_samples(payload: dict[str, JsonValue], is_streaming: bool) -> int:
    session = {key: value for key, value in payload.items() if key != "input"}
    session["stream_audio"] = is_streaming
    session["response_format"] = "pcm"
    byte_count = 0
    with connect(
        f"ws://127.0.0.1:{PORT}/v1/audio/speech/stream", open_timeout=20
    ) as socket:
        socket.send(json.dumps({"type": "session.config", "session": session}))
        configured = json.loads(socket.recv(timeout=180))
        assert configured["type"] == "session.configured", configured
        socket.send(json.dumps({"type": "input.text", "text": "Edit."}))
        socket.send(json.dumps({"type": "input.done"}))
        while True:
            message = socket.recv(timeout=180)
            if isinstance(message, bytes):
                byte_count += len(message)
            else:
                event = json.loads(message)
                assert event["type"] != "error", event
                if event["type"] == "audio.done":
                    assert event["error"] is False, event
                elif event["type"] == "session.done":
                    break
                else:
                    pass
    return byte_count // 2


def wait_ready(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        assert process.poll() is None, f"server exited {process.returncode}"
        try:
            response = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=5)
            if response.status_code == 200:
                return
            else:
                pass
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError("FireRedTTS3 readiness deadline expired")


def check_variant(variant: str, checkpoint: str) -> list[dict[str, JsonValue]]:
    configuration = {
        "config_cls": "FireRedTTS3PipelineConfig",
        "model_path": checkpoint,
        "stages": {
            "talker": {
                "engine": {"max_running_requests": 2, "mem_fraction_static": 0.2},
                "factory": {
                    "flow_cuda_graph_capture_sizes": [1, 2],
                    "flow_mode": "fp32" if variant == "instruct" else "compile_bf16",
                },
            }
        },
    }
    configuration_path = DESTINATION / f"{variant}.yaml"
    configuration_path.write_text(yaml.safe_dump(configuration))
    observations = []
    with (DESTINATION / f"server-{variant}.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                "--model-path",
                checkpoint,
                "--config",
                str(configuration_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(PORT),
                "--allowed-local-media-path",
                "docs/_static/audio",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_ready(process)
            rejected_requests = (
                [
                    (
                        "base-auto",
                        {
                            "input": "Hello",
                            "language": "Auto",
                            "ref_audio": REFERENCE,
                            "ref_text": TRANSCRIPT,
                        },
                        "language",
                    ),
                    (
                        "base-long",
                        {
                            "input": "a" * 401,
                            "ref_audio": REFERENCE,
                            "ref_text": TRANSCRIPT,
                        },
                        "input",
                    ),
                ]
                if variant == "base"
                else [
                    (
                        "instruct-long",
                        {
                            "input": "a" * 401,
                            "task_type": "VoiceDesign",
                            "instructions": "calm",
                        },
                        "input",
                    ),
                    (
                        "acoustic-unknown-attribute",
                        {
                            "input": "",
                            "task_type": "AcousticEdit",
                            "ref_audio": REFERENCE,
                            "stage_params": {
                                "preprocessing": {"acoustic_attribute": "Tone"}
                            },
                        },
                        "stage_params.preprocessing.acoustic_attribute",
                    ),
                    (
                        "acoustic-nonnumeric",
                        {
                            "input": "",
                            "task_type": "AcousticEdit",
                            "ref_audio": REFERENCE,
                            "stage_params": {
                                "preprocessing": {"acoustic_value": "fast"}
                            },
                        },
                        "stage_params.preprocessing.acoustic_value",
                    ),
                    (
                        "acoustic-nonfinite",
                        {
                            "input": "",
                            "task_type": "AcousticEdit",
                            "ref_audio": REFERENCE,
                            "stage_params": {
                                "preprocessing": {"acoustic_value": "inf"}
                            },
                        },
                        "stage_params.preprocessing.acoustic_value",
                    ),
                ]
            )
            for label, rejected_body, parameter in rejected_requests:
                rejection = httpx.post(
                    f"http://127.0.0.1:{PORT}/v1/audio/speech",
                    json=rejected_body,
                    timeout=20,
                )
                assert rejection.status_code == 400, (label, rejection.text)
                assert rejection.json()["error"]["param"] == parameter, (
                    label,
                    rejection.text,
                )
                observations.append(
                    {
                        "label": label,
                        "status": rejection.status_code,
                        "param": parameter,
                    }
                )
                print(json.dumps(observations[-1]), flush=True)
            cloning = {
                "input": "Hello, welcome to SGLang Omni.",
                "language": "English",
                "ref_audio": REFERENCE,
                "ref_text": TRANSCRIPT,
                "seed": 42,
            }
            for is_streaming in (False, True):
                observations.append(
                    request_audio(
                        f"{variant}-cloning-{is_streaming}",
                        {
                            **cloning,
                            "stream": is_streaming,
                            "response_format": "pcm" if is_streaming else "wav",
                        },
                    )
                )
            for patch_limit in (1, 8):
                observation = request_audio(
                    f"{variant}-cap-{patch_limit}",
                    {
                        **cloning,
                        "stage_params": {"talker": {"max_patches": patch_limit}},
                    },
                )
                assert observation["samples"] == patch_limit * 3840, observation
                assert (
                    observation["headers"]["x-finish-reason"] == "length"
                ), observation
                observations.append(observation)
            if variant == "base":
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [
                        pool.submit(
                            request_audio, f"parallel-{seed}", {**cloning, "seed": seed}
                        )
                        for seed in (43, 44)
                    ]
                    observations.extend(future.result() for future in futures)
                with httpx.stream(
                    "POST",
                    f"http://127.0.0.1:{PORT}/v1/audio/speech",
                    json={**cloning, "stream": True, "response_format": "pcm"},
                    timeout=180,
                ) as response:
                    assert response.status_code == 200
                    assert next(response.iter_bytes())
                observations.append(request_audio("after-disconnect", cloning))
            else:
                for task, fields in (
                    (
                        "VoiceDesign",
                        {
                            "input": "今天天气很好。",
                            "instructions": "一个年轻女性的温柔嗓音，语速稍慢。",
                        },
                    ),
                    (
                        "SemanticEdit",
                        {
                            "input": "",
                            "ref_audio": REFERENCE,
                            "instructions": "Replace 'birthday' with 'anniversary'.",
                        },
                    ),
                ):
                    observations.append(
                        request_audio(task, {"task_type": task, "seed": 42, **fields})
                    )
                editing = {
                    "input": "",
                    "task_type": "AcousticEdit",
                    "ref_audio": REFERENCE,
                    "speed": 0.8,
                    "seed": 42,
                }
                edited = request_audio("AcousticEdit-http", editing)
                observations.append(edited)
                for is_streaming in (False, True):
                    sample_count = websocket_samples(editing, is_streaming)
                    observation = {
                        "label": f"AcousticEdit-ws-{is_streaming}",
                        "samples": sample_count,
                    }
                    print(json.dumps(observation), flush=True)
                    assert sample_count == edited["samples"], observation
                    observations.append(observation)
                compressed_path = DESTINATION / "reference.m4a"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        REFERENCE,
                        "-t",
                        "1",
                        "-c:a",
                        "aac",
                        str(compressed_path),
                    ],
                    check=True,
                )
                audio_uri = "data:audio/mp4;base64," + base64.b64encode(
                    compressed_path.read_bytes()
                ).decode("ascii")
                observations.append(
                    request_audio(
                        "AcousticEdit-m4a", {**editing, "ref_audio": audio_uri}
                    )
                )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=40)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=20)
            else:
                pass
    return observations


if __name__ == "__main__":
    DESTINATION.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for model_variant, model_checkpoint in (
        ("base", BASE_CHECKPOINT),
        ("instruct", INSTRUCT_CHECKPOINT),
    ):
        outcomes.extend(check_variant(model_variant, model_checkpoint))
        (DESTINATION / "serving.json").write_text(json.dumps(outcomes, indent=2))
