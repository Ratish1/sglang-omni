# SPDX-License-Identifier: Apache-2.0
"""Compare the actual audio encoder against FP32 on captured request features."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    Qwen3OmniAudioEncoder,
    build_audio_tower,
    pack_padded_audio_features,
)
from sglang_omni.models.qwen3_omni.components.common import load_thinker_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--case-directory", action="append", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", arguments.gpu)
    native = Qwen3OmniAudioEncoder(
        arguments.model_path,
        device=str(device),
        dtype="bf16",
        enable_layer_cuda_graph=True,
    ).eval()
    graphed_layers = native.audio_tower.layers
    assert native.layer_graph_runner is not None
    reference = build_audio_tower(
        arguments.model_path,
        thinker_cfg=load_thinker_config(arguments.model_path),
        torch_dtype=torch.bfloat16,
        device=str(device),
    ).eval()
    reference.set_attn_implementation("sdpa")
    reports = []
    with torch.inference_mode():
        for directory in arguments.case_directory:
            features = torch.load(
                directory / "audio-features.pt", map_location="cpu", weights_only=True
            )
            recorded = torch.load(
                directory / "audio-conditioning.pt",
                map_location="cpu",
                weights_only=True,
            )
            native.audio_tower.layers = graphed_layers
            graphed = native(**features)["audio_embeds"].clone()
            assert torch.equal(graphed.cpu(), recorded["audio_embeds"])
            native.audio_tower.layers = graphed_layers.layers
            eager = native(**features)["audio_embeds"].clone()
            lengths = features["feature_attention_mask"].sum(dim=1)
            packed_features = pack_padded_audio_features(
                features["input_features"].to(device, dtype=torch.bfloat16),
                features["feature_attention_mask"],
                lengths,
            )
            reference.bfloat16()
            checkpoint_bfloat16 = reference(
                packed_features, feature_lens=lengths.to(device)
            ).last_hidden_state
            reference.float()
            with sdpa_kernel(SDPBackend.MATH):
                checkpoint_float32 = reference(
                    packed_features.float(), feature_lens=lengths.to(device)
                ).last_hidden_state
            report = {
                "case_directory": str(directory),
                "native_capture_reproduced": True,
                "graphed_vs_eager": asdict(compare_tensors(graphed, eager)),
                "graphed_vs_float32": asdict(
                    compare_tensors(graphed, checkpoint_float32)
                ),
                "eager_vs_float32": asdict(compare_tensors(eager, checkpoint_float32)),
                "checkpoint_bfloat16_vs_float32": asdict(
                    compare_tensors(checkpoint_bfloat16, checkpoint_float32)
                ),
            }
            torch.save(
                {
                    "graphed": graphed.cpu(),
                    "eager": eager.cpu(),
                    "checkpoint_bfloat16": checkpoint_bfloat16.cpu(),
                    "checkpoint_float32": checkpoint_float32.cpu(),
                },
                arguments.output_directory / f"{directory.name}.pt",
            )
            reports.append(report)
            (arguments.output_directory / "results.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
