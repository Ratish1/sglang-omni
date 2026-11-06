# SPDX-License-Identifier: Apache-2.0
"""Measure every captured Talker expert layer against explicit precision paths."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as functional
import tvm_ffi
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType, get_cutlass_fused_moe_module
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeTalkerTextSparseMoeBlock,
)

from scripts.qwen3_omni_audit_utils import compare_tensors


def run_experts(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    profiles: list[int],
) -> torch.Tensor:
    return cutlass_fused_moe(
        input=hidden_states,
        token_selected_experts=selected_experts.int(),
        token_final_scales=routing_weights.float(),
        fc1_expert_weights=gate_up_weights,
        fc2_expert_weights=down_weights,
        output_dtype=torch.bfloat16,
        quant_scales=None,
        activation_type=ActivationType.Swiglu,
        use_fused_finalize=False,
        profile_ids=profiles,
    )[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--profiles", type=int, nargs=2, required=True)
    parser.add_argument("--control-library", type=Path, required=True)
    parser.add_argument("--rounded-library", type=Path, required=True)
    parser.add_argument(
        "--captured-kernel",
        choices=("rebuilt_control", "rounded_fused"),
        default="rebuilt_control",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
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
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    assert reference.model.config._experts_implementation == "eager"
    saved = torch.load(arguments.trace_file, weights_only=True)
    live_token_count = saved["positions"].numel()
    bucket_token_count = min(
        tensor.shape[0]
        for name, tensor in saved.items()
        if name.startswith("moe_0_input_") and tensor.shape[0] >= live_token_count
    )
    modules = {
        "rebuilt_control": tvm_ffi.load_module(str(arguments.control_library)),
        "rounded_fused": tvm_ffi.load_module(str(arguments.rounded_library)),
    }
    runner_key = (torch.bfloat16,) * 3 + (False,) * 6
    runners = {name: module.init(*runner_key) for name, module in modules.items()}
    runner_table = get_cutlass_fused_moe_module("90").MoERunner.runner_dict
    captured_reference_inputs: list[torch.Tensor] = []
    captured_reference_outputs: list[torch.Tensor] = []

    def capture_reference(
        module: Qwen3OmniMoeTalkerTextSparseMoeBlock,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        captured_reference_inputs.append(
            inputs[0].reshape(-1, module.experts.hidden_dim).detach().clone()
        )
        captured_reference_outputs.append(
            output.reshape(-1, module.experts.hidden_dim).detach().clone()
        )

    hooks = [
        layer.mlp.register_forward_hook(capture_reference)
        for layer in reference.model.layers
    ]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        positions = saved["positions"].to(device).reshape(1, 1, -1)
        prefix_output = reference.model(
            inputs_embeds=saved["prefill_embeds"].to(device).unsqueeze(0),
            position_ids=positions.expand(3, 1, -1),
            use_cache=True,
        )
        prefix_logits = reference.codec_head(
            prefix_output.last_hidden_state[:, -1]
        ).float()
    for hook in hooks:
        hook.remove()
    assert len(captured_reference_inputs) == len(reference.model.layers)
    reports = []
    with torch.inference_mode():
        for layer_index, layer in enumerate(reference.model.layers):
            prefix = f"moe_{layer_index}_"
            hidden_states = saved[f"{prefix}input_{bucket_token_count}"].to(device)
            selected_experts = saved[f"{prefix}experts_{bucket_token_count}"].to(device)
            routing_weights = saved[f"{prefix}weights_{bucket_token_count}"].to(device)
            captured_routed = saved[f"{prefix}routed_output_{bucket_token_count}"].to(
                device
            )
            captured_shared = saved[f"{prefix}shared_output_{bucket_token_count}"].to(
                device
            )
            gate_weights, up_weights = layer.mlp.experts.gate_up_proj.chunk(2, dim=1)
            gate_up_weights = torch.cat((up_weights, gate_weights), dim=1).contiguous()
            down_weights = layer.mlp.experts.down_proj.contiguous()
            reference_routed = layer.mlp.experts(
                hidden_states, selected_experts.long(), routing_weights
            )
            shared = layer.mlp.shared_expert(hidden_states)
            shared *= layer.mlp.shared_expert_gate(hidden_states).sigmoid()
            _, reference_weights, reference_experts = layer.mlp.gate(hidden_states)
            routing_sets_equal = torch.equal(
                selected_experts.sort(-1).values, reference_experts.sort(-1).values
            )
            if routing_sets_equal:
                routing_weight_error = asdict(
                    compare_tensors(
                        routing_weights.gather(1, selected_experts.argsort(-1)),
                        reference_weights.gather(1, reference_experts.argsort(-1)),
                    )
                )
            else:
                routing_weight_error = None
            canonical_routed = layer.mlp.experts(
                hidden_states, reference_experts, reference_weights
            )
            report = {
                "layer": layer_index,
                "live_tokens": live_token_count,
                "bucket_tokens": bucket_token_count,
                "routing_sets_equal": routing_sets_equal,
                "routing_weight_error": routing_weight_error,
                "routing_precision_effect": asdict(
                    compare_tensors(reference_routed, canonical_routed)
                ),
                "shared_reference_error": asdict(
                    compare_tensors(captured_shared, shared)
                ),
                "reference_input_error": asdict(
                    compare_tensors(
                        hidden_states[:live_token_count],
                        captured_reference_inputs[layer_index],
                    )
                ),
                "reference_block_output_error": asdict(
                    compare_tensors(
                        (captured_routed + captured_shared)[:live_token_count],
                        captured_reference_outputs[layer_index],
                    )
                ),
                "reference_weight_dtype": str(reference_weights.dtype),
                "native_weight_dtype": str(routing_weights.dtype),
                "routed_errors": {},
                "activation_errors": {},
            }
            for name, runner in runners.items():
                with patch.dict(runner_table, {runner_key: runner}):
                    routed = run_experts(
                        hidden_states,
                        selected_experts,
                        routing_weights,
                        gate_up_weights,
                        down_weights,
                        arguments.profiles,
                    )
                if name == arguments.captured_kernel:
                    assert torch.equal(routed, captured_routed), layer_index
                else:
                    pass
                report["routed_errors"][name] = asdict(
                    compare_tensors(routed, reference_routed)
                )
            activation_shape = (
                bucket_token_count,
                selected_experts.shape[1],
                config.text_config.moe_intermediate_size,
            )
            reference_activation = torch.empty(
                activation_shape, device=device, dtype=torch.bfloat16
            )
            projection_rounded_activation = torch.empty_like(reference_activation)
            accumulator_activation = torch.empty_like(reference_activation)
            for expert_index in selected_experts.unique().tolist():
                token_rows, selected_columns = (
                    selected_experts == expert_index
                ).nonzero(as_tuple=True)
                expert_inputs = hidden_states[token_rows]
                projections = functional.linear(
                    expert_inputs, layer.mlp.experts.gate_up_proj[expert_index]
                )
                gate, up = projections.chunk(2, dim=-1)
                reference_activation[token_rows, selected_columns] = (
                    functional.silu(gate) * up
                )
                projection_rounded_activation[token_rows, selected_columns] = (
                    functional.silu(gate.float()) * up.float()
                ).to(torch.bfloat16)
                gate, up = (
                    functional.linear(
                        expert_inputs.double(),
                        layer.mlp.experts.gate_up_proj[expert_index].double(),
                    )
                    .float()
                    .chunk(2, dim=-1)
                )
                accumulator_activation[token_rows, selected_columns] = (
                    functional.silu(gate) * up
                ).to(torch.bfloat16)
            identity_down = torch.zeros_like(down_weights)
            activation_columns = identity_down.shape[2]
            diagonal = torch.arange(activation_columns, device=device)
            identity_down[:, diagonal, diagonal] = 1
            for name, runner in runners.items():
                measured = torch.empty_like(reference_activation)
                with patch.dict(runner_table, {runner_key: runner}):
                    for selected_column in range(selected_experts.shape[1]):
                        single_scale = torch.zeros_like(routing_weights)
                        single_scale[:, selected_column] = 1
                        identity_output = run_experts(
                            hidden_states,
                            selected_experts,
                            single_scale,
                            gate_up_weights,
                            identity_down,
                            arguments.profiles,
                        )
                        assert (
                            torch.count_nonzero(identity_output[:, activation_columns:])
                            == 0
                        )
                        measured[:, selected_column] = identity_output[
                            :, :activation_columns
                        ]
                report["activation_errors"][name] = {
                    precision: asdict(compare_tensors(measured, expected))
                    for precision, expected in (
                        ("eager_bf16", reference_activation),
                        (
                            "projection_bf16_activation_fp32",
                            projection_rounded_activation,
                        ),
                        (
                            "fp64_dot_rounded_to_fp32_before_activation",
                            accumulator_activation,
                        ),
                    )
                }
            reports.append(report)
            (arguments.output_directory / f"layer-{layer_index:02d}.json").write_text(
                json.dumps(report, indent=2)
            )
            print(
                json.dumps(
                    {
                        "layer": layer_index,
                        "routed_errors": report["routed_errors"],
                        "activation_errors": report["activation_errors"],
                    }
                ),
                flush=True,
            )
    source = Path(inspect.getfile(type(reference)))
    summary = {
        "model_path": arguments.model_path,
        "trace_file": str(arguments.trace_file),
        "trace_sha256": hashlib.sha256(arguments.trace_file.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(device),
        "transformers": version("transformers"),
        "flashinfer": version("flashinfer-python"),
        "reference_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference_experts": reference.model.config._experts_implementation,
        "reference_attention": "sdpa_math",
        "profiles": arguments.profiles,
        "captured_kernel": arguments.captured_kernel,
        "libraries": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in (
                ("rebuilt_control", arguments.control_library),
                ("rounded_fused", arguments.rounded_library),
            )
        },
        "layer_count": len(reports),
        "selected_experts_per_token": selected_experts.shape[1],
        "prefix_logits_error": asdict(
            compare_tensors(saved["logits"].to(device), prefix_logits)
        ),
        "layers": reports,
        "interpretation": "Measured numerical differences, not an arbitrary allclose pass or speech-quality certification. The FP64 dot-product path is a mathematical diagnostic, not the checkpoint BF16 contract.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
