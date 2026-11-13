"""Replay executed residual normalization against retained native operands."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from flashinfer.norm import fused_add_rmsnorm, rmsnorm
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.qwen3_omni_checkpoint_weights import CheckpointIndex


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    checkpoint_directory = Path(
        snapshot_download(arguments.model_path, revision=arguments.model_revision)
    )
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        checkpoint_directory
    ).thinker_config.text_config
    checkpoint_index = CheckpointIndex.model_validate_json(
        (checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    captures = [
        captured
        for filename in sorted((arguments.delivery_directory / "writes").glob("*.pt"))
        for captured in [torch.load(filename, map_location="cpu", weights_only=True)]
        if captured["request_id"] == arguments.request_id
    ]
    assert len(captures) == 1
    captured = captures[0]
    operators = captured["operator_batches"]
    layers = captured["layer_batches"]
    live = slice(
        captured["batch_offset"], captured["batch_offset"] + captured["extend_length"]
    )
    activation = layers["layer_00_hidden"].to(device)
    residual = activation.clone()
    comparisons = []
    for layer_index in range(len(layers) - 1):
        prefix = f"operator_{layer_index:02d}"
        weights = {}
        for kind in ("input_layernorm", "post_attention_layernorm"):
            weight_name = f"thinker.model.layers.{layer_index}.{kind}.weight"
            with safe_open(
                checkpoint_directory / checkpoint_index.weight_map[weight_name],
                framework="pt",
                device="cpu",
            ) as checkpoint:
                weights[kind] = checkpoint.get_tensor(weight_name).to(device)
        unrounded_input = (
            activation.float()
            if layer_index == 0
            else activation.float() + residual.float()
        )
        assert torch.equal(
            unrounded_input.to(activation.dtype)[live],
            layers[f"layer_{layer_index:02d}_hidden"][live].to(device),
        ), layer_index
        if layer_index == 0:
            normalized_input = rmsnorm(
                activation, weights["input_layernorm"], configuration.rms_norm_eps
            )
        else:
            normalized_input = activation.clone()
            fused_add_rmsnorm(
                normalized_input,
                residual,
                weights["input_layernorm"],
                configuration.rms_norm_eps,
            )
        expected_input = operators[f"{prefix}_qkv_input"].to(device)
        assert torch.equal(normalized_input[live], expected_input[live]), layer_index
        assert torch.equal(
            residual[live], layers[f"layer_{layer_index:02d}_hidden"][live].to(device)
        ), layer_index
        projection = operators[f"{prefix}_attention_projection_output"].to(device)
        unrounded_post_attention = projection.float() + residual.float()
        normalized_post_attention = projection.clone()
        fused_add_rmsnorm(
            normalized_post_attention,
            residual,
            weights["post_attention_layernorm"],
            configuration.rms_norm_eps,
        )
        expected_post_attention = operators[f"{prefix}_router_input"].to(device)
        assert torch.equal(
            normalized_post_attention[live], expected_post_attention[live]
        ), layer_index
        assert torch.equal(
            residual[live], unrounded_post_attention.to(residual.dtype)[live]
        ), layer_index
        for kind, values, weight, actual in (
            (
                "input",
                unrounded_input,
                weights["input_layernorm"],
                normalized_input,
            ),
            (
                "post_attention",
                unrounded_post_attention,
                weights["post_attention_layernorm"],
                normalized_post_attention,
            ),
        ):
            values = values[live].double()
            inverse_rms = torch.rsqrt(
                values.square().mean(-1, keepdim=True) + configuration.rms_norm_eps
            )
            independently_rounded = (values * inverse_rms * weight.double()).to(
                actual.dtype
            )
            actual = actual[live]
            assert bool(torch.isfinite(actual).all())
            next_representable = torch.nextafter(actual, independently_rounded)
            comparisons.append(
                {
                    "layer_index": layer_index,
                    "normalization": kind,
                    "native_replay_exact": True,
                    "fp64_formula": asdict(
                        compare_tensors(actual, independently_rounded)
                    ),
                    "elements": actual.numel(),
                    "different_elements": int(
                        (actual != independently_rounded).sum().item()
                    ),
                    "more_than_one_bf16_step": int(
                        (
                            (actual != independently_rounded)
                            & (next_representable != independently_rounded)
                        )
                        .sum()
                        .item()
                    ),
                }
            )
        activation = operators[f"{prefix}_moe_output"].to(device)
        assert torch.equal(
            (activation.float() + residual.float()).to(activation.dtype)[live],
            layers[f"layer_{layer_index + 1:02d}_hidden"][live].to(device),
        ), layer_index
    report = {
        "request_id": arguments.request_id,
        "layers": len(layers) - 1,
        "normalizations": comparisons,
        "scope": "Original padded native operands, checkpoint weights, and original FlashInfer residual/norm entry points. Every live normalized activation and rounded residual must reproduce the captured execution exactly. The separate FP64 formula measures numerical distance, not equality required of an approximate FP32 reduction/reciprocal-square-root kernel. No speech-quality or production-fix claim follows from replay equality.",
    }
    (arguments.output_directory / "comparisons.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            {
                "layers": report["layers"],
                "exact_native_normalizations": len(comparisons),
                "fp64_different_elements": sum(
                    comparison["different_elements"] for comparison in comparisons
                ),
                "more_than_one_bf16_step": sum(
                    comparison["more_than_one_bf16_step"] for comparison in comparisons
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
