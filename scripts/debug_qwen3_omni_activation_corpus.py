"""Compare native activation arithmetic on the original speech request populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from transformers import Qwen3OmniMoeConfig

from benchmarks.benchmarker.utils import wait_for_service
from scripts.debug_qwen3_omni_producer_speech import stop_owned_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-template", type=Path, required=True)
    parser.add_argument("--server-working-directory", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--request-id", action="append")
    parser.add_argument("--capture-request-id", action="append")
    parser.add_argument("--capture-token-rows", type=int, default=112)
    parser.add_argument("--capture-talker-operators", action="store_true")
    parser.add_argument("--capture-all-conditioning", action="store_true")
    parser.add_argument("--contribution-source-directory", type=Path)
    parser.add_argument(
        "--reference-predictor-mode",
        choices=("identity", "checkpoint", "checkpoint-codes", "checkpoint-feedback"),
    )
    parser.add_argument(
        "--control",
        action="append",
        choices=(
            "original",
            "rounded-activation",
            "rounded-routing",
            "rounded-contributions",
        ),
    )
    arguments = parser.parse_args()
    if "rounded-contributions" in (arguments.control or ()):
        assert arguments.contribution_source_directory is not None
    elif arguments.contribution_source_directory is not None:
        raise ValueError(
            "Contribution source requires the rounded-contributions control"
        )
    else:
        pass
    receiver_only = (
        arguments.capture_all_conditioning or arguments.capture_request_id is not None
    )
    if arguments.reference_predictor_mode is not None and arguments.request_id is None:
        raise ValueError("Reference predictor controls require selected request IDs")
    elif arguments.capture_all_conditioning and (
        arguments.request_id is not None
        or arguments.capture_request_id is not None
        or arguments.capture_talker_operators
        or arguments.reference_predictor_mode is not None
        or arguments.control != ["original"]
    ):
        raise ValueError(
            "Whole-population receiver capture requires only the original control"
        )
    elif arguments.capture_request_id is not None and (
        arguments.control
        not in (
            ["original"],
            ["rounded-routing"],
            ["rounded-contributions"],
            ["original", "rounded-routing"],
        )
        or arguments.reference_predictor_mode is not None
    ):
        raise ValueError("Selected receiver capture requires native Talker controls")
    else:
        pass
    arguments.output_directory.mkdir(exist_ok=False)
    driver_source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if arguments.server_working_directory is None:
        source = driver_source
    else:
        source = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=arguments.server_working_directory,
            text=True,
        ).strip()
    template = json.loads(arguments.server_template.read_text())
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    accepted_layer = configuration.talker_config.accept_hidden_layer
    server_command = [*template["command"], "--log-level", "debug"]
    request_hashes: dict[str, str] = {}
    workload_request_ids: dict[str, list[str]] = {}
    for workload in ("tts", "mmsu"):
        request_files = list(
            (arguments.input_directory / workload).glob("*.request.json")
        )
        assert len(request_files) == (1088 if workload == "tts" else 40)
        workload_request_ids[workload] = []
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
            if (
                arguments.request_id is None
                or payload["request_id"] in arguments.request_id
            ):
                workload_request_ids[workload].append(payload["request_id"])
            else:
                pass
            request_hashes[str(request_file)] = hashlib.sha256(
                request_bytes
            ).hexdigest()
    (arguments.output_directory / "request-bytes.json").write_text(
        json.dumps(request_hashes, indent=2)
    )
    selected_ids = arguments.output_directory / "selected-request-ids.json"
    traffic_request_ids = {
        request_id
        for request_ids in workload_request_ids.values()
        for request_id in request_ids
    }
    capture_request_ids = (
        sorted(traffic_request_ids)
        if arguments.capture_all_conditioning
        else arguments.capture_request_id or arguments.request_id or []
    )
    assert set(capture_request_ids) <= traffic_request_ids
    selected_ids.write_text(json.dumps(capture_request_ids))
    if arguments.request_id is not None:
        assert {
            request_id
            for request_ids in workload_request_ids.values()
            for request_id in request_ids
        } == set(arguments.request_id)
    else:
        pass
    native_state = None
    for control in arguments.control or ("original", "rounded-activation"):
        output = arguments.output_directory / control
        output.mkdir()
        diagnostic_environment = {
            "SGLANG_OMNI_DIAGNOSTIC_DELIVERY_DIRECTORY": str(output / "delivery"),
            "SGLANG_OMNI_CORPUS_REQUEST_ID_FILE": str(selected_ids),
            "SGLANG_OMNI_DIAGNOSTIC_CAPTURE_TOKEN_ROWS": str(
                arguments.capture_token_rows
            ),
            "SGLANG_OMNI_DIAGNOSTIC_ROUNDED_THINKER_ACTIVATION": (
                "1" if control == "rounded-activation" else "0"
            ),
        }
        if receiver_only:
            diagnostic_environment = {
                "SGLANG_OMNI_CORPUS_REQUEST_ID_FILE": str(selected_ids),
            }
        else:
            pass
        if control in ("rounded-routing", "rounded-contributions"):
            diagnostic_environment[
                "SGLANG_OMNI_DIAGNOSTIC_CHECKPOINT_ROUTING_WEIGHTS"
            ] = "1"
        else:
            pass
        if control == "rounded-contributions":
            diagnostic_environment["SGLANG_OMNI_DIAGNOSTIC_CONTRIBUTION_SOURCE"] = str(
                arguments.contribution_source_directory
            )
        else:
            pass
        if capture_request_ids:
            diagnostic_environment["SGLANG_OMNI_DIAGNOSTIC_CONDITIONING_DIRECTORY"] = (
                str(output / "conditioning")
            )
        else:
            pass
        if arguments.capture_talker_operators:
            assert capture_request_ids
            diagnostic_environment.update(
                {
                    "SGLANG_OMNI_DIAGNOSTIC_LAYER_OPERANDS": "1",
                    "SGLANG_OMNI_DIAGNOSTIC_MOE_BATCHES": "1",
                }
            )
        else:
            pass
        if arguments.reference_predictor_mode is not None:
            diagnostic_environment.update(
                {
                    "SGLANG_OMNI_DIAGNOSTIC_REFERENCE_PREDICTOR_MODE": arguments.reference_predictor_mode,
                    "SGLANG_OMNI_DIAGNOSTIC_REFERENCE_MODEL_PATH": arguments.model_path,
                    "SGLANG_OMNI_DIAGNOSTIC_REFERENCE_MODEL_REVISION": arguments.model_revision,
                }
            )
        else:
            pass
        environment = os.environ.copy()
        assert not any(
            name.startswith("SGLANG_OMNI_DIAGNOSTIC_") for name in environment
        )
        environment.update(diagnostic_environment)
        with (output / "server.log").open("wb") as log:
            process = subprocess.Popen(
                server_command,
                cwd=arguments.server_working_directory,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (output / "server-command.json").write_text(
            json.dumps(
                {
                    "source": source,
                    "driver_source": driver_source,
                    "server_working_directory": str(
                        arguments.server_working_directory or Path.cwd()
                    ),
                    "control": control,
                    "pid": process.pid,
                    "command": server_command,
                    "diagnostic_environment": diagnostic_environment,
                    "scope": (
                        "Native Talker control on original traffic, without Thinker operand observers."
                        if receiver_only
                        else "Native rounded SiLU in the first checkpoint-derived accept_hidden_layer Thinker blocks, including warmup and decode."
                    ),
                    "model_revision": arguments.model_revision,
                    "reference_predictor_mode": arguments.reference_predictor_mode,
                    "captured_request_ids": capture_request_ids,
                    "accepted_layer": accepted_layer,
                    "timing_scope": "Diagnostic capture requested, not production performance qualification.",
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
            if receiver_only:
                pass
            else:
                actual_state = json.loads(
                    (output / "delivery/native-thinker-state.json").read_text()
                )
                if native_state is None:
                    native_state = actual_state
                else:
                    assert actual_state["parameters"] == native_state["parameters"]
                    for actual_layer, original_layer in zip(
                        actual_state["layers"], native_state["layers"], strict=True
                    ):
                        expected_layer = original_layer.copy()
                        if actual_layer["layer_index"] < accepted_layer:
                            expected_layer["activation"] = "silu_rounded"
                        else:
                            pass
                        assert actual_layer == expected_layer, actual_layer[
                            "layer_index"
                        ]
            for workload in ("tts", "mmsu"):
                request_ids = workload_request_ids[workload]
                if not request_ids:
                    continue
                else:
                    pass
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
                if arguments.request_id is not None:
                    for request_id in request_ids:
                        command.extend(("--request-id", request_id))
                else:
                    pass
                (output / f"{workload}-command.json").write_text(
                    json.dumps(command, indent=2)
                )
                with (output / f"{workload}.log").open("wb") as log:
                    subprocess.run(
                        command, check=True, stdout=log, stderr=subprocess.STDOUT
                    )
                result = json.loads((output / workload / "results.json").read_text())
                assert result["summary"]["failed_requests"] == 0
                assert len(result["per_request"]) == len(request_ids)
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
            stop_owned_server(process, output)


if __name__ == "__main__":
    main()
else:
    pass
