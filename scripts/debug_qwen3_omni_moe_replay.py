# SPDX-License-Identifier: Apache-2.0
"""Replay a captured routed-expert call without loading the serving pipeline."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack, nullcontext
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import torch
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig, autotune
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--tune-max-tokens", type=int, nargs="+", required=True)
    parser.add_argument("--token-counts", type=int, nargs="+")
    parser.add_argument("--profiles", type=int, nargs=2)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    device = torch.device("cuda", arguments.gpu)
    torch.cuda.set_device(device)
    checkpoint = Path(snapshot_download(arguments.model_path, local_files_only=True))
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path)
    expert_count = config.talker_config.text_config.num_experts
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "talker.model.layers.0.mlp.experts."
    expert_names = [name for name in weight_map if name.startswith(prefix)]
    with ExitStack() as stack:
        readers = {
            filename: stack.enter_context(
                safe_open(str(checkpoint / filename), framework="pt")
            )
            for filename in {weight_map[name] for name in expert_names}
        }
        weights = {
            name: readers[weight_map[name]].get_tensor(name) for name in expert_names
        }
        gate_up_weights = torch.stack(
            [
                torch.cat(
                    (
                        weights[f"{prefix}{expert}.up_proj.weight"],
                        weights[f"{prefix}{expert}.gate_proj.weight"],
                    )
                )
                for expert in range(expert_count)
            ]
        ).to(device)
        down_weights = torch.stack(
            [
                weights[f"{prefix}{expert}.down_proj.weight"]
                for expert in range(expert_count)
            ]
        ).to(device)
    saved = torch.load(arguments.trace_file, weights_only=True)
    hidden_states = saved["moe_0_input_128"].to(device)
    selected_experts = saved["moe_0_experts_128"].to(device).int()
    routing_weights = saved["moe_0_weights_128"].to(device).float()
    native_output = saved["moe_0_routed_output_128"].to(device)
    if arguments.cache is not None:
        AutoTuner.get().load_configs(str(arguments.cache))
    else:
        pass
    arguments.output_directory.mkdir(exist_ok=False)
    comparisons = []
    choices: list[dict[str, str | int]] = []
    choose_one = AutoTuner.choose_one

    def record_choice(
        tuner: AutoTuner,
        custom_op: str,
        runners: list[TunableRunner],
        tuning_config: TuningConfig,
        inputs: list[torch.Tensor | None],
        *,
        gemm_idx: int,
    ) -> tuple[TunableRunner, int]:
        runner, tactic = choose_one(
            tuner, custom_op, runners, tuning_config, inputs, gemm_idx=gemm_idx
        )
        spec = tuning_config.dynamic_tensor_specs[0]
        activation = inputs[0]
        assert activation is not None
        choices.append(
            dict(
                operation=custom_op,
                tactic=tactic,
                token_bucket=spec.map_to_tuning_buckets(activation.shape[0]),
                maximum_bucket=spec.gen_tuning_buckets[-1],
            )
        )
        return runner, tactic

    with torch.inference_mode(), patch.object(AutoTuner, "choose_one", record_choice):
        if arguments.reference:
            torch.set_float32_matmul_precision("highest")
            checkpoint_output = torch.zeros_like(hidden_states)
            float_output = torch.zeros_like(hidden_states, dtype=torch.float32)
            for expert in selected_experts.unique().tolist():
                token_rows, routing_columns = (selected_experts == expert).nonzero(
                    as_tuple=True
                )
                expert_input = hidden_states[token_rows]
                up, gate = torch.nn.functional.linear(
                    expert_input, gate_up_weights[expert]
                ).chunk(2, dim=-1)
                expert_output = torch.nn.functional.linear(
                    torch.nn.functional.silu(gate) * up, down_weights[expert]
                )
                scales = routing_weights[token_rows, routing_columns, None]
                checkpoint_output.index_add_(
                    0, token_rows, expert_output * scales.to(expert_output.dtype)
                )
                up, gate = torch.nn.functional.linear(
                    expert_input.float(), gate_up_weights[expert].float()
                ).chunk(2, dim=-1)
                expert_output = torch.nn.functional.linear(
                    torch.nn.functional.silu(gate) * up, down_weights[expert].float()
                )
                float_output.index_add_(0, token_rows, expert_output * scales)
            torch.save(
                checkpoint_output.cpu(), arguments.output_directory / "checkpoint.pt"
            )
            torch.save(float_output.cpu(), arguments.output_directory / "float32.pt")
        else:
            pass
        token_counts = arguments.token_counts or [hidden_states.shape[0]] * len(
            arguments.tune_max_tokens
        )
        for call_index, (max_tokens, token_count) in enumerate(
            zip(arguments.tune_max_tokens, token_counts, strict=True)
        ):
            choices.clear()
            profile_override = (
                {"profile_ids": arguments.profiles}
                if arguments.profiles is not None
                else {}
            )
            tuning_context = (
                autotune(
                    True,
                    cache=str(
                        arguments.output_directory / f"autotune-{max_tokens}.json"
                    ),
                )
                if arguments.autotune
                else nullcontext()
            )
            with tuning_context:
                output = cutlass_fused_moe(
                    input=hidden_states.narrow(0, 0, token_count),
                    token_selected_experts=selected_experts.narrow(0, 0, token_count),
                    token_final_scales=routing_weights.narrow(0, 0, token_count),
                    fc1_expert_weights=gate_up_weights,
                    fc2_expert_weights=down_weights,
                    output_dtype=torch.bfloat16,
                    quant_scales=None,
                    tune_max_num_tokens=max_tokens,
                    activation_type=ActivationType.Swiglu,
                    use_fused_finalize=False,
                    **profile_override,
                )[0]
            expected_output = native_output.narrow(0, 0, token_count)
            difference = output.float() - expected_output.float()
            comparison = dict(
                tune_max_tokens=max_tokens,
                token_count=token_count,
                choices=choices.copy(),
                native_equal=torch.equal(output, expected_output),
                relative_error=(
                    difference.norm() / expected_output.float().norm()
                ).item(),
                max_error=difference.abs().max().item(),
            )
            comparisons.append(comparison)
            output_path = (
                arguments.output_directory / f"{call_index:02d}-{max_tokens}.pt"
            )
            comparison["output_file"] = output_path.name
            torch.save(output.cpu(), output_path)
            print(json.dumps(comparison), flush=True)
    (arguments.output_directory / "comparisons.json").write_text(
        json.dumps(
            dict(
                flashinfer=version("flashinfer-python"),
                autotuned=arguments.autotune,
                profiles=arguments.profiles,
                cache=str(arguments.cache),
                comparisons=comparisons,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
