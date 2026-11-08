"""Reconstruct captured Talker residual, normalization and position boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from sglang.kernels.ops.attention.rope import apply_rope_with_cos_sin_cache_inplace
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.debug_qwen3_omni_operator_audit import compare_tensors


def normalized(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    hidden_states = hidden_states.float()
    inverse_rms = (hidden_states.square().mean(dim=-1, keepdim=True) + epsilon).rsqrt()
    return (hidden_states * inverse_rms * weight.float()).to(weight.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    assert not arguments.output_file.exists()
    torch.set_float32_matmul_precision("highest")
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    epsilon = config.text_config.rms_norm_eps
    head_size = config.text_config.head_dim
    query_width = config.text_config.num_attention_heads * head_size
    key_width = config.text_config.num_key_value_heads * head_size
    inverse_frequencies = 1.0 / (
        config.text_config.rope_parameters["rope_theta"]
        ** (
            torch.arange(0, head_size, 2, device=reference.device, dtype=torch.float32)
            / head_size
        )
    )
    reports = []
    with torch.inference_mode():
        for capture_file in sorted(arguments.capture_directory.glob("request-*.pt")):
            capture = torch.load(capture_file, map_location="cpu", weights_only=True)
            request_id = capture["request_id"]
            if request_id not in arguments.request_id:
                continue
            else:
                pass
            conditioning = torch.load(
                arguments.capture_directory
                / "talker_state"
                / f"{request_id}.conditioning.pt",
                map_location=reference.device,
                weights_only=True,
            )
            for frame, snapshot in enumerate(capture["snapshots"]):
                hidden_states = (
                    snapshot["prefill_input_embeddings"][-1:].to(reference.device)
                    if frame == 0
                    else conditioning["native_decode_input_embeddings"][
                        frame - 1 : frame
                    ]
                )
                residual = None
                for layer_index, layer in enumerate(reference.model.layers):
                    attention = {
                        name.rsplit(".", 1)[-1].removeprefix("diagnostic_"): tensor.to(
                            reference.device
                        )
                        for name, tensor in snapshot.items()
                        if f".layers.{layer_index}.self_attn.attn.diagnostic_" in name
                    }
                    moe = {
                        name.rsplit(".", 1)[-1].removeprefix(
                            "diagnostic_moe_"
                        ): tensor.to(reference.device)
                        for name, tensor in snapshot.items()
                        if f".layers.{layer_index}.mlp.diagnostic_moe_" in name
                    }
                    if residual is None:
                        residual = hidden_states.clone()
                        attention_sum = hidden_states.float()
                        expected_input = rmsnorm(
                            hidden_states, layer.input_layernorm.weight, epsilon
                        )
                    else:
                        attention_sum = hidden_states.float() + residual.float()
                        expected_input = hidden_states.clone()
                        fused_add_rmsnorm(
                            expected_input,
                            residual,
                            layer.input_layernorm.weight,
                            epsilon,
                        )
                    attention_normalized = normalized(
                        attention_sum, layer.input_layernorm.weight, epsilon
                    )
                    mlp_sum = (
                        residual.float()
                        + attention["attention_projection"][None].float()
                    )
                    expected_mlp_input = attention["attention_projection"][None].clone()
                    fused_add_rmsnorm(
                        expected_mlp_input,
                        residual,
                        layer.post_attention_layernorm.weight,
                        epsilon,
                    )
                    mlp_normalized = normalized(
                        mlp_sum, layer.post_attention_layernorm.weight, epsilon
                    )
                    query, keys, values = attention["qkv_projection"].split(
                        [query_width, key_width, key_width]
                    )
                    query = query.view(1, -1, head_size).clone()
                    keys = keys.view(1, -1, head_size).clone()
                    query_float = normalized(
                        query, layer.self_attn.q_norm.weight, epsilon
                    )
                    keys_float = normalized(
                        keys, layer.self_attn.k_norm.weight, epsilon
                    )
                    fused_inplace_qknorm(
                        query,
                        keys,
                        layer.self_attn.q_norm.weight,
                        layer.self_attn.k_norm.weight,
                        epsilon,
                        head_dim=head_size,
                    )
                    position = int(attention["position"])
                    frequencies = (
                        torch.arange(
                            position + 1, device=reference.device, dtype=torch.float32
                        )[:, None]
                        * inverse_frequencies[None]
                    )
                    cosine_sine_cache = torch.cat(
                        (frequencies.cos(), frequencies.sin()), dim=-1
                    )
                    cosines = cosine_sine_cache[-1, : head_size // 2].repeat(2)
                    sines = cosine_sine_cache[-1, head_size // 2 :].repeat(2)
                    float_rotated = []
                    for vectors in (query_float, keys_float):
                        first_half, second_half = vectors.float().chunk(2, dim=-1)
                        rotated = torch.cat((-second_half, first_half), dim=-1)
                        float_rotated.append(
                            (vectors.float() * cosines + rotated * sines).to(
                                vectors.dtype
                            )
                        )
                    apply_rope_with_cos_sin_cache_inplace(
                        positions=torch.tensor([position], device=reference.device),
                        q=query,
                        k=keys,
                        cos_sin_cache=cosine_sine_cache,
                        is_neox=True,
                    )
                    native_keys = snapshot[f"layer_{layer_index}_keys"][-1].to(
                        reference.device
                    )
                    native_values = snapshot[f"layer_{layer_index}_values"][-1].to(
                        reference.device
                    )
                    comparisons = {
                        "attention_norm_vs_kernel": compare_tensors(
                            attention["attention_input"][None], expected_input
                        ),
                        "attention_norm_vs_float32": compare_tensors(
                            attention["attention_input"][None], attention_normalized
                        ),
                        "moe_norm_vs_kernel": compare_tensors(
                            moe["input"][None], expected_mlp_input
                        ),
                        "moe_norm_vs_float32": compare_tensors(
                            moe["input"][None], mlp_normalized
                        ),
                        "query_position_vs_kernel": compare_tensors(
                            attention["query"], query[0]
                        ),
                        "query_position_vs_float32": compare_tensors(
                            attention["query"], float_rotated[0][0]
                        ),
                        "key_position_vs_kernel": compare_tensors(native_keys, keys[0]),
                        "key_position_vs_float32": compare_tensors(
                            native_keys, float_rotated[1][0]
                        ),
                        "value_vs_projection": compare_tensors(
                            native_values, values.view_as(native_values)
                        ),
                    }
                    reports.append(
                        {
                            "request_id": request_id,
                            "frame": frame,
                            "layer": layer_index,
                            "comparisons": comparisons,
                        }
                    )
                    hidden_states = moe["output"][None]
                final_hidden = hidden_states.clone()
                fused_add_rmsnorm(
                    final_hidden, residual, reference.model.norm.weight, epsilon
                )
                final_comparison = compare_tensors(
                    conditioning["native_backbone_hidden_states"][frame : frame + 1],
                    final_hidden,
                )
                reports[-1]["final_normalization"] = final_comparison
                arguments.output_file.write_text(json.dumps(reports, indent=2))
                print(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "frame": frame,
                            "final_normalization": final_comparison,
                        }
                    ),
                    flush=True,
                )
    assert {report["request_id"] for report in reports} == set(arguments.request_id)


if __name__ == "__main__":
    main()
