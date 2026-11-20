# SPDX-License-Identifier: Apache-2.0
"""Measure one RMSNorm cast-order swap in both held Talker implementations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.cache_utils import DynamicCache
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextRMSNorm,
)

from scripts.debug_qwen3_omni_held_layers import difference
from scripts.debug_qwen3_omni_held_talker import filtered_distribution


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--native-swapped", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    target_device = torch.device("cuda", 0)
    files = (
        ("h200_sampling", arguments.fixture, "decode-sampling-001.pt"),
        ("prefix", arguments.fixture, "first-decode-prefix-cache-000.pt"),
        ("operands", arguments.fixture, "backbone-decode-001.pt"),
        ("native_sampling", arguments.native, "decode-sampling-001.pt"),
        ("swap_sampling", arguments.native_swapped, "decode-sampling-001.pt"),
        ("native_layers", arguments.native, "first-decode-layers.pt"),
        ("swap_layers", arguments.native_swapped, "first-decode-layers.pt"),
        ("native_input", arguments.native, "held-first-decode-000.pt"),
        ("swap_input", arguments.native_swapped, "held-first-decode-000.pt"),
    )
    tensors = {
        name: torch.load(
            directory / filename, map_location=target_device, weights_only=True
        )
        for name, directory, filename in files
    }
    native_sampling = tensors["native_sampling"]
    swap_sampling = tensors["swap_sampling"]
    h200_sampling = tensors["h200_sampling"]
    assert torch.equal(native_sampling["raw_logits"], h200_sampling["raw_logits"])
    for field_name in native_sampling.keys() - {
        "raw_logits",
        "predictor_query",
        "sampled_code",
    }:
        assert torch.equal(
            native_sampling[field_name], swap_sampling[field_name]
        ), field_name
    held = tensors["native_input"]
    position = native_sampling["position"].reshape(1, 1, 1)
    prefix = tensors["prefix"]
    assert (
        int(position)
        == held["position"]
        == tensors["swap_input"]["position"]
        == prefix["position"]
    )
    embedding = held["embedding"]
    assert torch.equal(embedding, tensors["swap_input"]["embedding"])
    operands = tensors["operands"]
    assert torch.equal(
        embedding,
        operands["feedback_row"].reshape(-1) + operands["text_row"].reshape(-1),
    )
    native_layer = tensors["native_layers"][0]
    swapped_layer = tensors["swap_layers"][0]
    for field_name in ("positions", "input_contribution", "attention_residual"):
        assert torch.equal(native_layer[field_name], swapped_layer[field_name])
    assert torch.equal(embedding, native_layer["input_contribution"])
    native_normalized = native_layer["attention_normalized"]
    checkpoint_normalized = swapped_layer["attention_normalized"]
    assert torch.equal(
        native_normalized, swapped_layer["attention_normalized_before_override"]
    )
    assert not torch.equal(native_normalized, checkpoint_normalized)

    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    reference.eval()
    rotary_axes = operands["rotary_positions"]
    position_ids = (
        position.expand(4, 1, 1)
        if rotary_axes is None
        else torch.cat((position, rotary_axes.reshape(3, 1, 1)))
    )
    norm = reference.model.layers[0].input_layernorm
    reference_logits: dict[str, torch.Tensor] = {}
    reference_hidden: dict[str, tuple[torch.Tensor, ...]] = {}

    def install_native_norm(
        module: Qwen3OmniMoeThinkerTextRMSNorm,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert module is norm
        assert torch.equal(inputs[0].reshape(-1), embedding)
        assert torch.equal(output.reshape(-1), checkpoint_normalized)
        return native_normalized.view_as(output)

    assert torch.equal(
        norm(embedding.view(1, 1, -1)).reshape(-1), checkpoint_normalized
    )
    for arm_name, should_swap in (
        ("checkpoint", False),
        ("checkpoint_with_native_cast", True),
        ("checkpoint_repeat", False),
        ("checkpoint_with_native_cast_repeat", True),
    ):
        cache = DynamicCache(config=config.text_config)
        for layer_index, (key_rows, value_rows) in enumerate(
            zip(prefix["key_rows"], prefix["value_rows"], strict=True)
        ):
            cache.update(
                key_rows.transpose(0, 1).unsqueeze(0),
                value_rows.transpose(0, 1).unsqueeze(0),
                layer_index,
            )
        hook = norm.register_forward_hook(install_native_norm) if should_swap else None
        try:
            result = reference.model(
                inputs_embeds=embedding.view(1, 1, -1),
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
        finally:
            if hook is not None:
                hook.remove()
            else:
                pass
        assert cache.get_seq_length() == int(position) + 1
        reference_logits[arm_name] = (
            reference.codec_head(result.last_hidden_state).float().reshape(-1)
        )
        reference_hidden[arm_name] = result.hidden_states
    assert torch.equal(
        reference_logits["checkpoint"], reference_logits["checkpoint_repeat"]
    )
    assert torch.equal(
        reference_logits["checkpoint_with_native_cast"],
        reference_logits["checkpoint_with_native_cast_repeat"],
    )
    logits = {
        "native": native_sampling["raw_logits"],
        "native_with_checkpoint_cast": swap_sampling["raw_logits"],
        "checkpoint": reference_logits["checkpoint"],
        "checkpoint_with_native_cast": reference_logits["checkpoint_with_native_cast"],
    }
    sampled_native_code = int(native_sampling["sampled_code"])
    sampling_reports = {}
    distributions = {}
    for arm_name, arm_logits in logits.items():
        distribution, selected_code = filtered_distribution(arm_logits, native_sampling)
        distributions[arm_name] = distribution
        sampling_reports[arm_name] = {
            "selected_code": selected_code,
            "probability_of_native_code": float(distribution[0, sampled_native_code]),
        }
    assert sampling_reports["native"]["selected_code"] == sampled_native_code
    assert sampling_reports["native_with_checkpoint_cast"]["selected_code"] == int(
        swap_sampling["sampled_code"]
    )
    pairs = (
        ("native", "checkpoint"),
        ("native_with_checkpoint_cast", "checkpoint"),
        ("checkpoint_with_native_cast", "native"),
        ("native_with_checkpoint_cast", "native"),
        ("checkpoint_with_native_cast", "checkpoint"),
    )
    report = {
        "position": int(position),
        "first_norm_swap": difference(checkpoint_normalized, native_normalized),
        "sampling": sampling_reports,
        "comparisons": {
            f"{left}_vs_{right}": {
                **difference(logits[left], logits[right]),
                "filtered_total_variation": float(
                    0.5 * (distributions[left] - distributions[right]).abs().sum()
                ),
            }
            for left, right in pairs
        },
        "native_layer_inputs": [
            difference(swapped["attention_residual"], baseline["attention_residual"])
            for baseline, swapped in zip(
                tensors["native_layers"], tensors["swap_layers"], strict=True
            )
        ],
        "checkpoint_layer_inputs_and_final_norm": [
            difference(swapped, baseline)
            for baseline, swapped in zip(
                reference_hidden["checkpoint"],
                reference_hidden["checkpoint_with_native_cast"],
                strict=True,
            )
        ],
    }
    with arguments.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
else:
    pass
