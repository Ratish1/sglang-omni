"""Compare native rotary arithmetic on the original speech request populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import psutil

from benchmarks.benchmarker.utils import stop_server, wait_for_service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-template", type=Path, required=True)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    template = json.loads(arguments.server_template.read_text())
    server_command = [*template["command"], "--log-level", "debug"]
    request_hashes: dict[str, str] = {}
    for workload in ("tts", "mmsu"):
        request_files = list(
            (arguments.input_directory / workload).glob("*.request.json")
        )
        assert len(request_files) == (1088 if workload == "tts" else 40)
        for request_file in request_files:
            request_bytes = request_file.read_bytes()
            payload = json.loads(request_bytes)
            assert payload["max_tokens"] == 256
            assert payload["stream"] is False
            assert payload["modalities"] == ["text", "audio"]
            assert "seed" not in payload
            assert payload["temperature"] == (0.7 if workload == "tts" else 0.0)
            assert len(payload["audios"]) == 1
            assert Path(payload["audios"][0]).is_file(), payload["audios"]
            request_hashes[str(request_file)] = hashlib.sha256(
                request_bytes
            ).hexdigest()
    (arguments.output_directory / "request-bytes.json").write_text(
        json.dumps(request_hashes, indent=2)
    )
    selected_ids = arguments.output_directory / "selected-request-ids.json"
    selected_ids.write_text("[]\n")
    native_state = None
    for control in ("original", "no-contraction"):
        output = arguments.output_directory / control
        output.mkdir()
        diagnostic_environment = {
            "SGLANG_OMNI_DIAGNOSTIC_DELIVERY_DIRECTORY": str(output / "delivery"),
            "SGLANG_OMNI_CORPUS_REQUEST_ID_FILE": str(selected_ids),
            "SGLANG_OMNI_DIAGNOSTIC_CAPTURE_TOKEN_ROWS": "112",
            "SGLANG_OMNI_DIAGNOSTIC_ROPE_ALL_FORWARDS": "1",
            "SGLANG_OMNI_DIAGNOSTIC_ROPE_NO_CONTRACTION": (
                "1" if control == "no-contraction" else "0"
            ),
        }
        environment = os.environ.copy()
        assert not any(
            name.startswith("SGLANG_OMNI_DIAGNOSTIC_") for name in environment
        )
        environment.update(diagnostic_environment)
        with (output / "server.log").open("wb") as log:
            process = subprocess.Popen(
                server_command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (output / "server-command.json").write_text(
            json.dumps(
                {
                    "source": source,
                    "pid": process.pid,
                    "command": server_command,
                    "diagnostic_environment": diagnostic_environment,
                    "scope": "All shared native Thinker rotary forwards, including warmup and decode.",
                    "timing_scope": "Bounded operator observers enabled in both arms, not production performance qualification.",
                },
                indent=2,
            )
        )
        try:
            wait_for_service(
                arguments.base_url,
                server_process=process,
                server_log_file=output / "server.log",
                trust_env=False,
            )
            actual_state = json.loads(
                (output / "delivery/native-thinker-state.json").read_text()
            )
            if native_state is None:
                native_state = actual_state
            else:
                assert actual_state == native_state
            for workload in ("tts", "mmsu"):
                inputs = arguments.input_directory / workload
                command = [
                    "python3",
                    "-u",
                    "-m",
                    "scripts.debug_qwen3_omni_replay_corpus",
                    "--mode",
                    "replay",
                    "--workload",
                    workload,
                    "--input-directory",
                    str(inputs),
                    "--reference-results",
                    str(
                        inputs
                        / (
                            "speed_results.json"
                            if workload == "tts"
                            else "results.json"
                        )
                    ),
                    "--base-url",
                    arguments.base_url,
                    "--concurrency",
                    str(arguments.concurrency),
                    "--output-directory",
                    str(output / workload),
                ]
                (output / f"{workload}-command.json").write_text(
                    json.dumps(command, indent=2)
                )
                with (output / f"{workload}.log").open("wb") as log:
                    subprocess.run(
                        command, check=True, stdout=log, stderr=subprocess.STDOUT
                    )
                result = json.loads((output / workload / "results.json").read_text())
                assert result["summary"]["failed_requests"] == 0
                assert len(result["per_request"]) == (1088 if workload == "tts" else 40)
                assert all(
                    record["finish_reason"] == "stop"
                    for record in result["per_request"]
                )
                print(
                    json.dumps(
                        {
                            "control": control,
                            "workload": workload,
                            "summary": result["summary"],
                        }
                    ),
                    flush=True,
                )
        finally:
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
                pid
                for pid, status in statuses.items()
                if status != psutil.STATUS_ZOMBIE
            ]
            (output / "stop-ownership.json").write_text(
                json.dumps(
                    {"remaining_statuses": statuses, "live_survivors": live}, indent=2
                )
            )
            assert not live, live


if __name__ == "__main__":
    main()
else:
    pass
