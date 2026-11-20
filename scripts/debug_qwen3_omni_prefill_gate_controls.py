# SPDX-License-Identifier: Apache-2.0
"""Verify two native prefill repairs against stock eager execution."""

import base64
import hashlib
import json
from pathlib import Path

import httpx
import torch

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from scripts.debug_qwen3_omni_norm_population import (
    PREFILL_PROBE_TOKEN_BY_SAMPLE,
    compare_native_prefill,
    controlled_history,
    post_seedtts_sample,
)
from sglang_omni.models.qwen3_omni.boundary_witness import WitnessControl

PREFILL_WITNESS_REQUESTS = {
    "common_voice_en_19756059-common_voice_en_19756062": "02e74230-d55a-4fba-96d6-04ced2ee388c",
    "common_voice_en_17817002-common_voice_en_17817003": "4e3727c7-ce33-49e8-9cb4-2a672e495590",
}
CONTROL_TIMEOUT_SECONDS = 300


def capture_prefill_gate_controls(
    server_port: int, run_directory: Path, control_path: Path
) -> None:
    samples = {
        sample.sample_id: sample for sample in load_seedtts_samples(SEEDTTS_DATASET_ID)
    }
    endpoint = f"http://127.0.0.1:{server_port}/v1/chat/completions"
    with httpx.Client(timeout=CONTROL_TIMEOUT_SECONDS, trust_env=False) as client:
        for sample_id, request_id in PREFILL_WITNESS_REQUESTS.items():
            trace_directory = run_directory / sample_id
            trace_directory.mkdir()
            control: WitnessControl = {
                "request_id": request_id,
                "output_directory": str(trace_directory),
                "norm_scope": "none",
                "held_input_directory": None,
                "prefill_probe_token_index": PREFILL_PROBE_TOKEN_BY_SAMPLE[sample_id],
            }
            pending_control = control_path.with_suffix(".pending")
            pending_control.write_text(json.dumps(control))
            pending_control.replace(control_path)
            sample = samples[sample_id]
            response = (
                post_seedtts_sample(client, endpoint, sample, request_id)
                .raise_for_status()
                .json()
            )
            assert response["id"] == f"chatcmpl-{request_id}"
            (choice,) = response["choices"]
            assert choice["finish_reason"] == "stop"
            audio = base64.b64decode(choice["message"]["audio"]["data"], validate=True)
            (trace_directory / "response.json").write_text(
                json.dumps(
                    {
                        "text": choice["message"]["content"],
                        "waveform_sha256": hashlib.sha256(audio).hexdigest(),
                        "input_sha256": hashlib.sha256(
                            Path(sample.ref_audio).read_bytes()
                        ).hexdigest(),
                    }
                )
            )
            print(f"{run_directory.name}: captured {sample_id}", flush=True)


def verify_prefill_gate_controls(output: Path) -> None:
    reports = []
    for sample_id in PREFILL_WITNESS_REQUESTS:
        eager_trace = output / "2-baseline-eager" / sample_id
        eager_response = json.loads((eager_trace / "response.json").read_bytes())
        eager_frames, _, eager_complete = controlled_history(eager_trace)
        assert eager_complete
        eager_layers = torch.load(eager_trace / "prefill-layers.pt", weights_only=True)
        for arm in ("1-baseline", "3-prefill-gate"):
            graph_trace = output / arm / sample_id
            graph_response = json.loads((graph_trace / "response.json").read_bytes())
            assert graph_response["input_sha256"] == eager_response["input_sha256"]
            assert graph_response["text"] == eager_response["text"]
            boundaries = compare_native_prefill(graph_trace, eager_trace)
            assert boundaries["breakable_prefill_graph_replays"] > 0
            assert boundaries["eager_prefill_graph_replays"] == 0
            graph_frames, _, graph_complete = controlled_history(graph_trace)
            assert graph_complete
            graph_layers = torch.load(
                graph_trace / "prefill-layers.pt", weights_only=True
            )
            differing_gate_layers = [
                layer_index
                for layer_index, (graph_layer, eager_layer) in enumerate(
                    zip(graph_layers, eager_layers, strict=True)
                )
                if not torch.equal(
                    graph_layer["shared_gate"], eager_layer["shared_gate"]
                )
            ]
            codec_matches = len(graph_frames) == len(eager_frames) and all(
                torch.equal(graph_frame, eager_frame)
                for graph_frame, eager_frame in zip(graph_frames, eager_frames)
            )
            waveform_matches = (
                graph_response["waveform_sha256"] == eager_response["waveform_sha256"]
            )
            prefixes_match = all(
                boundaries[f"first-decode-prefix-cache-000/{name}"]
                for name in ("key_rows", "value_rows")
            )
            reports.append(
                {
                    "sample_id": sample_id,
                    "arm": arm,
                    "different_gate_layers": differing_gate_layers,
                    "prefixes_match": prefixes_match,
                    "complete_codec_matches": codec_matches,
                    "waveform_matches": waveform_matches,
                    "boundary_comparison": boundaries,
                    "graph_response": graph_response,
                    "eager_response": eager_response,
                }
            )
    (output / "prefill-gate-controls.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
    for report in reports:
        if report["arm"] == "1-baseline":
            assert (
                report["different_gate_layers"] and not report["prefixes_match"]
            ), report
        else:
            assert not report["different_gate_layers"], report
            assert report["prefixes_match"] and report["complete_codec_matches"], report
            assert report["waveform_matches"], report
    print("Prefill gate controls passed", flush=True)
