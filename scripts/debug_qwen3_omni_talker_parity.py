# SPDX-License-Identifier: Apache-2.0
"""Compare captured Talker forwards with the checkpoint implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-experts", action="store_true")
    parser.add_argument("--expert-steps", type=int, nargs="+")
    parser.add_argument("--backbone-only", action="store_true")
    arguments = parser.parse_args()
    device = torch.device("cuda", arguments.gpu)
    torch.cuda.set_device(device)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    comparisons = []
    cache = None
    with torch.inference_mode():
        for step in range(arguments.steps):
            before_phase = "before_prefill" if step == 0 else "before_decode"
            after_phase = "prefill_forward" if step == 0 else "post_decode"
            before = torch.load(
                arguments.trace_directory / f"{step:04d}-{before_phase}.pt",
                weights_only=True,
            )
            after = torch.load(
                arguments.trace_directory / f"{step:04d}-{after_phase}.pt",
                weights_only=True,
            )
            embeddings = (
                before["prefill_embeds"] if step == 0 else before["feedback_input"]
            ).to(device)
            positions = before["positions"].to(device).reshape(1, 1, -1)
            outputs = reference.model(
                inputs_embeds=embeddings.unsqueeze(0),
                position_ids=positions.expand(3, 1, -1),
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            logits = reference.codec_head(outputs.last_hidden_state[:, -1]).float()
            actual_logits = after["logits"].to(device).float()
            difference = logits - actual_logits
            comparison = {
                "step": step,
                "logits_max_absolute_error": difference.abs().max().item(),
                "logits_relative_error": (difference.norm() / logits.norm()).item(),
                "reference_top_tokens": logits[0].topk(8).indices.tolist(),
                "actual_top_tokens": actual_logits[0].topk(8).indices.tolist(),
                "reference_eos_raw_probability": logits.softmax(-1)[
                    0, config.codec_eos_token_id
                ].item(),
                "actual_eos_raw_probability": actual_logits.softmax(-1)[
                    0, config.codec_eos_token_id
                ].item(),
            }
            if not arguments.backbone_only:
                sampled = torch.load(
                    arguments.trace_directory
                    / f"{step:04d}-{'post_prefill' if step == 0 else 'post_decode'}.pt",
                    weights_only=True,
                )
                codes = sampled["codec_ids"].to(device)
                native_hidden = sampled["hidden_states"].to(device).unsqueeze(1)
                first_embedding = reference.model.codec_embedding(codes[:, :1])
                predicted = reference.code_predictor.generate(
                    inputs_embeds=torch.cat((native_hidden, first_embedding), dim=1),
                    max_new_tokens=config.num_code_groups - 1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                reference_codes = predicted.sequences[0].tolist()
                comparison["reference_residual_codes"] = reference_codes
                comparison["actual_residual_codes"] = codes[0, 1:].tolist()
                for index, reference_code in enumerate(reference_codes):
                    actual_code = codes[0, index + 1].item()
                    if reference_code != actual_code:
                        scores = predicted.scores[index][0].float()
                        comparison["first_residual_difference"] = index
                        comparison["residual_logit_margin"] = (
                            scores[reference_code] - scores[actual_code]
                        ).item()
                        comparison["actual_code_reference_rank"] = (
                            (scores > scores[actual_code]).sum().item()
                        )
                        break
                    else:
                        pass
                embedding_rows = [first_embedding]
                for index, embedding in enumerate(
                    reference.code_predictor.get_input_embeddings()
                ):
                    embedding_rows.append(embedding(codes[:, index + 1 : index + 2]))
                feedback = torch.cat(embedding_rows, dim=1).sum(dim=1)
                actual_feedback = sampled["codec_feedback"].to(device)
                comparison["feedback_max_absolute_error"] = (
                    (feedback - actual_feedback).abs().max().item()
                )
            else:
                pass
            if arguments.compare_experts and (
                arguments.expert_steps is None or step in arguments.expert_steps
            ):
                expert_comparisons = []
                for layer_index, layer in enumerate(reference.model.layers):
                    prefix = f"moe_{layer_index}_"
                    live_rows = embeddings.shape[-2]
                    bucket_rows = min(
                        tensor.shape[0]
                        for name, tensor in after.items()
                        if name.startswith(prefix + "input_")
                        and tensor.shape[0] >= live_rows
                    )
                    native_input = after[f"{prefix}input_{bucket_rows}"][:live_rows].to(
                        device
                    )
                    router_logits, routing_weights, selected_experts = layer.mlp.gate(
                        native_input
                    )
                    native_experts = (
                        after[f"{prefix}experts_{bucket_rows}"][:live_rows]
                        .to(device)
                        .long()
                    )
                    native_weights = after[f"{prefix}weights_{bucket_rows}"][
                        :live_rows
                    ].to(device)
                    shared_output = layer.mlp.shared_expert(native_input)
                    shared_output *= layer.mlp.shared_expert_gate(
                        native_input
                    ).sigmoid()
                    routed_output = layer.mlp.experts(
                        native_input, selected_experts, routing_weights
                    )
                    fixed_routing_output = layer.mlp.experts(
                        native_input, native_experts, native_weights
                    )
                    expert_comparison = {
                        "layer": layer_index,
                        "expert_sets_equal": torch.equal(
                            selected_experts.sort(-1).values,
                            native_experts.sort(-1).values,
                        ),
                        "native_weight_sum_min": native_weights.sum(-1).min().item(),
                        "native_weight_sum_max": native_weights.sum(-1).max().item(),
                    }
                    for name, expected, actual in (
                        (
                            "router",
                            router_logits,
                            after[f"{prefix}router_logits_{bucket_rows}"],
                        ),
                        (
                            "shared",
                            shared_output,
                            after[f"{prefix}shared_output_{bucket_rows}"],
                        ),
                        (
                            "routed",
                            routed_output,
                            after[f"{prefix}routed_output_{bucket_rows}"],
                        ),
                        (
                            "fixed_routing",
                            fixed_routing_output,
                            after[f"{prefix}routed_output_{bucket_rows}"],
                        ),
                    ):
                        difference = (
                            expected.float() - actual[:live_rows].to(device).float()
                        )
                        expert_comparison[f"{name}_relative_error"] = (
                            difference.norm() / expected.float().norm()
                        ).item()
                        expert_comparison[f"{name}_max_error"] = (
                            difference.abs().max().item()
                        )
                    expert_comparisons.append(expert_comparison)
                comparison["experts"] = expert_comparisons
            else:
                pass
            comparisons.append(comparison)
            print(json.dumps(comparison), flush=True)
    arguments.output.write_text(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
