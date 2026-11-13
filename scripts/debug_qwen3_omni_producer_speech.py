"""Run guarded producer fixtures through fresh native speech lifecycles."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import time
import wave
from pathlib import Path

import psutil
import requests
import torch

from benchmarks.benchmarker.utils import stop_server, wait_for_service

OWNED_PROCESS_EXIT_TIMEOUT_SECONDS = 10
OWNED_PROCESS_POLL_SECONDS = 1


def stop_owned_server(process: subprocess.Popen[bytes], output: Path) -> None:
    owned = (
        [
            psutil.Process(process.pid),
            *psutil.Process(process.pid).children(recursive=True),
        ]
        if process.poll() is None
        else []
    )
    (output / "stop-ownership-start.json").write_text(
        json.dumps({"owned_pids": [child.pid for child in owned]}, indent=2)
    )
    stop_server(process)
    deadline = time.monotonic() + OWNED_PROCESS_EXIT_TIMEOUT_SECONDS
    while True:
        statuses: dict[str, str] = {}
        for child in owned:
            try:
                if child.is_running():
                    statuses[str(child.pid)] = child.status()
                else:
                    pass
            except psutil.NoSuchProcess:
                pass
        live = [
            pid for pid, status in statuses.items() if status != psutil.STATUS_ZOMBIE
        ]
        if not live or time.monotonic() >= deadline:
            break
        else:
            time.sleep(OWNED_PROCESS_POLL_SECONDS)
    (output / "stop-ownership.json").write_text(
        json.dumps(
            {
                "owned_pids": [child.pid for child in owned],
                "remaining_statuses": statuses,
                "live_survivors": live,
            },
            indent=2,
        )
    )
    assert not live, live


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--fixture-directories", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--fixture-mode", choices=("identity", "checkpoint"), default="checkpoint"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--request-timeout-seconds", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    template = json.loads(
        (arguments.baseline_directory / "server-command.json").read_text()
    )
    baseline_result = json.loads(
        (arguments.baseline_directory / "request-result.json").read_text()
    )
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            template["source"],
            "--",
            "sglang_omni",
            "examples/configs",
        ],
        check=True,
    )
    request_bytes = arguments.request_file.read_bytes()
    assert (
        hashlib.sha256(request_bytes).hexdigest()
        == baseline_result["request_bytes_sha256"]
    )
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for fixture_directory in arguments.fixture_directories:
        fixture_file = fixture_directory / f"{arguments.fixture_mode}.pt"
        fixture = torch.load(fixture_file, map_location="cpu", weights_only=True)
        assert fixture["mode"] == arguments.fixture_mode
        name = (
            "identity"
            if arguments.fixture_mode == "identity"
            else fixture["producer_precision"]
        )
        request_id = fixture["request_id"]
        output = arguments.output_directory / name
        output.mkdir(exist_ok=False)
        environment = os.environ.copy()
        diagnostic_environment = template["diagnostic_environment"].copy()
        diagnostic_environment.update(
            {
                "SGLANG_OMNI_DIAGNOSTIC_DELIVERY_DIRECTORY": str(output / "delivery"),
                "SGLANG_OMNI_DIAGNOSTIC_CONDITIONING_DIRECTORY": str(
                    output / "conditioning"
                ),
                "SGLANG_OMNI_DIAGNOSTIC_PRODUCER_CONTROL": str(fixture_file),
            }
        )
        environment.update(diagnostic_environment)
        with (output / "server.log").open("wb") as log:
            process = subprocess.Popen(
                template["command"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        metadata = {
            **template,
            "source": source,
            "model_source": template["source"],
            "pid": process.pid,
            "diagnostic_environment": diagnostic_environment,
        }
        (output / "server-command.json").write_text(json.dumps(metadata, indent=2))
        try:
            wait_for_service(
                arguments.base_url,
                server_process=process,
                server_log_file=output / "server.log",
                trust_env=False,
            )
            with requests.Session() as client:
                client.trust_env = False
                response = client.post(
                    f"{arguments.base_url}/v1/chat/completions",
                    data=request_bytes,
                    headers={"Content-Type": "application/json"},
                    timeout=arguments.request_timeout_seconds,
                )
            response.raise_for_status()
            payload = response.json()
            choice = payload["choices"][0]
            assert choice["message"]["content"] == baseline_result["text"]
            assert choice["finish_reason"] == baseline_result["finish_reason"] == "stop"
            assert payload["usage"] == baseline_result["usage"]
            waveform = base64.b64decode(
                choice["message"]["audio"]["data"], validate=True
            )
            (output / "speech.wav").write_bytes(waveform)
            with wave.open(io.BytesIO(waveform)) as decoded:
                duration_seconds = decoded.getnframes() / decoded.getframerate()
            conditioning = output / "conditioning" / request_id
            baseline = arguments.baseline_directory / "conditioning" / request_id
            prompt = torch.load(conditioning / "prompt.pt", weights_only=True)
            original_prompt = torch.load(baseline / "prompt.pt", weights_only=True)
            positions = fixture["prompt_positions"]
            assert torch.equal(
                prompt["prompt_hidden"][positions], fixture["replacement_hidden"]
            )
            unchanged = torch.ones(len(prompt["prompt_hidden"]), dtype=torch.bool)
            unchanged[positions] = False
            assert torch.equal(
                prompt["prompt_hidden"][unchanged],
                original_prompt["prompt_hidden"][unchanged],
            )
            for field in (
                "prompt_input_ids",
                "initial_assistant_token_ids",
                "speaker_id",
                "has_prompt_hidden",
            ):
                assert torch.equal(prompt[field], original_prompt[field]), field
            for filename in ("0000-before_prefill.pt", "finished.pt"):
                actual = torch.load(conditioning / filename, weights_only=True)
                original = torch.load(baseline / filename, weights_only=True)
                fields = (
                    (
                        "input_ids",
                        "positions",
                        "extend_prefix_length",
                        "mrope_positions",
                        "output_ids",
                        "history_length",
                    )
                    if filename == "0000-before_prefill.pt"
                    else (
                        "assistant_token_ids",
                        "enqueued_text_rows",
                        "tts_pad_embed",
                        "tts_eos_embed",
                        "thinker_done",
                        "pending_text",
                        "finished_reason",
                    )
                )
                for field in fields:
                    first, second = original[field], actual[field]
                    if isinstance(first, torch.Tensor):
                        assert torch.equal(first, second), field
                    elif (
                        isinstance(first, list)
                        and first
                        and isinstance(first[0], torch.Tensor)
                    ):
                        assert len(first) == len(second) and all(
                            torch.equal(a, b)
                            for a, b in zip(first, second, strict=True)
                        ), field
                    else:
                        assert first == second, field
            assert json.loads(
                (conditioning / "sampling.json").read_text()
            ) == json.loads((baseline / "sampling.json").read_text())
            assert json.loads(
                (output / "delivery/native-thinker-state.json").read_text()
            ) == json.loads(
                (
                    arguments.baseline_directory / "delivery/native-thinker-state.json"
                ).read_text()
            )
            frames = [
                torch.load(path, weights_only=True)
                for path in sorted(conditioning.glob("*-output.pt"))
            ]
            assert frames and all(bool(frame["past_exact"]) for frame in frames)
            report = {
                "control": name,
                "status": response.status_code,
                "request_bytes_sha256": baseline_result["request_bytes_sha256"],
                "text": choice["message"]["content"],
                "finish_reason": choice["finish_reason"],
                "usage": payload["usage"],
                "waveform_sha256": hashlib.sha256(waveform).hexdigest(),
                "audio_duration_s": duration_seconds,
                "codec_frames": len(frames),
                "native_weights_and_flags_exact": True,
                "sampling_and_future_text_exact": True,
                "only_selected_prompt_hidden_rows_changed": True,
                "all_past_checks_exact": True,
                "waveform_retention": "Temporary until paired Qwen3-ASR scoring completes.",
            }
            (output / "request-result.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            stop_owned_server(process, output)


if __name__ == "__main__":
    main()
else:
    pass
