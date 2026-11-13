"""Compare native Thinker activation contracts through complete speech generation."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

import requests
import torch
from transformers import Qwen3OmniMoeConfig

from benchmarks.benchmarker.utils import wait_for_service
from scripts.debug_qwen3_omni_producer_speech import stop_owned_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--request-timeout-seconds", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    template = json.loads(
        (arguments.baseline_directory / "server-command.json").read_text()
    )
    original = json.loads(
        (arguments.baseline_directory / "request-result.json").read_text()
    )
    original_state = json.loads(
        (
            arguments.baseline_directory / "delivery/native-thinker-state.json"
        ).read_text()
    )
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    accepted_layer = configuration.talker_config.accept_hidden_layer
    assert 0 < accepted_layer < len(original_state["layers"])
    request_bytes = arguments.request_file.read_bytes()
    assert hashlib.sha256(request_bytes).hexdigest() == original["request_bytes_sha256"]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for control in ("original", "rounded-activation"):
        output = arguments.output_directory / control
        output.mkdir(exist_ok=False)
        diagnostic_environment = template["diagnostic_environment"].copy()
        diagnostic_environment.pop("SGLANG_OMNI_DIAGNOSTIC_PRODUCER_CONTROL")
        diagnostic_environment.update(
            {
                "SGLANG_OMNI_DIAGNOSTIC_DELIVERY_DIRECTORY": str(output / "delivery"),
                "SGLANG_OMNI_DIAGNOSTIC_CONDITIONING_DIRECTORY": str(
                    output / "conditioning"
                ),
                "SGLANG_OMNI_DIAGNOSTIC_ROUNDED_THINKER_ACTIVATION": (
                    "1" if control == "rounded-activation" else "0"
                ),
            }
        )
        with (output / "server.log").open("wb") as log:
            process = subprocess.Popen(
                template["command"],
                env=os.environ | diagnostic_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        (output / "server-command.json").write_text(
            json.dumps(
                {
                    **template,
                    "source": source,
                    "pid": process.pid,
                    "diagnostic_environment": diagnostic_environment,
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
            waveform = base64.b64decode(
                choice["message"]["audio"]["data"], validate=True
            )
            (output / "speech.wav").write_bytes(waveform)
            state = json.loads(
                (output / "delivery/native-thinker-state.json").read_text()
            )
            assert state["parameters"] == original_state["parameters"]
            for actual_layer, original_layer in zip(
                state["layers"], original_state["layers"], strict=True
            ):
                expected_layer = original_layer.copy()
                if (
                    control == "rounded-activation"
                    and actual_layer["layer_index"] < accepted_layer
                ):
                    expected_layer["activation"] = "silu_rounded"
                else:
                    pass
                assert actual_layer == expected_layer, actual_layer["layer_index"]
            request_directories = list((output / "conditioning").iterdir())
            assert len(request_directories) == 1
            directory = request_directories[0]
            baseline = arguments.baseline_directory / "conditioning" / directory.name
            assert json.loads((directory / "sampling.json").read_text()) == json.loads(
                (baseline / "sampling.json").read_text()
            )
            finished = torch.load(directory / "finished.pt", weights_only=True)
            waveform_sha256 = hashlib.sha256(waveform).hexdigest()
            if control == "original":
                assert waveform_sha256 == original["waveform_sha256"]
                assert choice["message"]["content"] == original["text"]
                assert payload["usage"] == original["usage"]
                for filename in ("prompt.pt", "0000-before_prefill.pt", "finished.pt"):
                    previous = torch.load(baseline / filename, weights_only=True)
                    current = torch.load(directory / filename, weights_only=True)
                    assert previous.keys() == current.keys()
                    for field, value in previous.items():
                        if isinstance(value, torch.Tensor):
                            assert torch.equal(value, current[field]), (filename, field)
                        else:
                            pass
                for path in sorted(directory.glob("[0-9][0-9][0-9][0-9]-output.pt")):
                    previous = torch.load(baseline / path.name, weights_only=True)
                    current = torch.load(path, weights_only=True)
                    for field in ("codec_ids", "logits", "codec_feedback"):
                        assert torch.equal(previous[field], current[field]), (
                            path.name,
                            field,
                        )
            else:
                pass
            report = {
                "control": control,
                "model_revision": arguments.model_revision,
                "accepted_layer": accepted_layer,
                "request_bytes_sha256": original["request_bytes_sha256"],
                "waveform_sha256": waveform_sha256,
                "text": choice["message"]["content"],
                "text_matches_original": choice["message"]["content"]
                == original["text"],
                "finish_reason": choice["finish_reason"],
                "usage": payload["usage"],
                "codec_frames": len(finished["output_ids"]),
                "native_weights_and_other_flags_exact": True,
                "sampling_exact": True,
                "scope": "The alternate changes only the first accept_hidden_layer Thinker MoE activations to SGLang's existing rounded SiLU contract, before graph capture. It applies to warmup, prefill and decode, so prefix KV and returned text are not held. Talker and Code2Wav are unchanged. This is a native-path diagnostic, not a qualified production repair.",
            }
            (output / "request-result.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
        finally:
            stop_owned_server(process, output)


if __name__ == "__main__":
    main()
else:
    pass
