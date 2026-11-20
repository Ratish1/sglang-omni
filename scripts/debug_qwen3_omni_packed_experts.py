# SPDX-License-Identifier: Apache-2.0
"""Replay the actual packed first-layer expert tensors and cross selected tactics."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from flashinfer.autotuner import AutoTuner

from scripts.debug_qwen3_omni_peer_shape import read_weights
from scripts.debug_qwen3_omni_prefill_prefix import (
    compare_activation_store,
    routed_experts,
)


def difference(first: torch.Tensor, second: torch.Tensor) -> dict[str, int | float]:
    assert first.shape == second.shape
    delta = first.float() - second.float()
    return {
        "changed_elements": int(delta.count_nonzero()),
        "max_abs_difference": float(delta.abs().max()),
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--serial-trace", type=Path, required=True)
    parser.add_argument("--peer-trace", type=Path, required=True)
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    arguments = parser.parse_args()
    torch.cuda.set_device(arguments.gpu_id)
    device = torch.device("cuda", arguments.gpu_id)
    config = json.loads((arguments.checkpoint / "config.json").read_bytes())[
        "talker_config"
    ]["text_config"]
    input_width = config["hidden_size"]
    expert_width = config["moe_intermediate_size"]
    expert_count = config["num_experts"]
    first_projection = torch.empty(
        expert_count, 2 * expert_width, input_width, dtype=torch.bfloat16, device=device
    )
    second_projection = torch.empty(
        expert_count, input_width, expert_width, dtype=torch.bfloat16, device=device
    )
    for expert_index in range(expert_count):
        gate, up, down = read_weights(
            arguments.checkpoint,
            tuple(
                f"mlp.experts.{expert_index}.{projection}_proj"
                for projection in ("gate", "up", "down")
            ),
        )
        first_projection[expert_index] = torch.cat((up, gate))
        second_projection[expert_index] = down

    tuner = AutoTuner.get()
    tuner.load_configs(str(arguments.autotune_cache))
    assert not tuner.is_tuning_mode
    choose_profile = tuner.choose_one
    operands = {}
    outputs = {}
    target_indices = {}
    tactic_pairs = {}
    sequences = {}
    for name, trace_directory in (
        ("serial", arguments.serial_trace),
        ("peers", arguments.peer_trace),
    ):
        geometry = torch.load(
            trace_directory / "prefill-probe-geometry-000.pt", weights_only=True
        )
        layer = torch.load(
            trace_directory / "prefill-layers.pt",
            weights_only=True,
            map_location=device,
        )[0]
        target_index = geometry["packed_token_index"]
        raw_tokens = sum(geometry["request_token_lengths"])
        inputs = layer["packed_expert_input"]
        expert_ids = layer["packed_expert_ids"].to(torch.int32)
        expert_weights = layer["packed_expert_weights"].to(torch.float32)
        recorded_output = layer["packed_expert_output"]
        assert torch.equal(inputs[target_index], layer["mlp_normalized"])
        assert torch.equal(expert_ids[target_index], layer["routed_expert_ids"])
        assert torch.equal(expert_weights[target_index], layer["routed_expert_weights"])
        assert torch.equal(recorded_output[target_index], layer["routed_output"])
        operands[name] = (
            inputs,
            expert_ids,
            expert_weights,
            first_projection,
            second_projection,
        )
        outputs[name] = recorded_output
        target_indices[name] = target_index
        sequences[name] = geometry["request_token_lengths"]
        with patch.object(tuner, "choose_one", wraps=choose_profile) as choices:
            automatic = routed_experts(*operands[name])
        assert torch.equal(automatic[:raw_tokens], recorded_output[:raw_tokens]), (
            name,
            difference(automatic[:raw_tokens], recorded_output[:raw_tokens]),
        )
        assert len(choices.call_args_list) == 2
        tactic_pairs[name] = [
            int(choose_profile(*call.args, **call.kwargs)[1])
            for call in choices.call_args_list
        ]

    assert all(
        torch.equal(a[target_indices["serial"]], b[target_indices["peers"]])
        for a, b in zip(operands["serial"][:3], operands["peers"][:3], strict=True)
    ), sequences
    effects = {}
    for name, other in (("serial", "peers"), ("peers", "serial")):
        target_index = target_indices[name]
        other_index = target_indices[other]
        native_first, native_second = tactic_pairs[name]
        alternate_first, alternate_second = tactic_pairs[other]
        trials = {}
        for trial_name, pair in (
            ("explicit_reference", [native_first, native_second]),
            ("swap_first_projection", [alternate_first, native_second]),
            ("swap_second_projection", [native_first, alternate_second]),
            ("swap_both", [alternate_first, alternate_second]),
        ):
            result = routed_experts(*operands[name], tactics=pair)[target_index]
            trials[trial_name] = {
                "profile_ids": pair,
                "vs_native": difference(result, outputs[name][target_index]),
                "vs_other": difference(result, outputs[other][other_index]),
            }
        assert trials["explicit_reference"]["vs_native"]["changed_elements"] == 0
        effects[name] = {
            "valid_sequence_lengths": sequences[name],
            "physical_token_count": operands[name][0].size(0),
            "target_index": target_index,
            "trials": trials,
        }

    print(
        json.dumps({"packed_replays_exact": True, "effects": effects}, sort_keys=True)
    )
    peer_target = target_indices["peers"]
    peer_second = tactic_pairs["peers"][1]
    compare_activation_store(
        *operands["peers"],
        slice(peer_target, peer_target + 1),
        [[tactic_pairs[name][0], peer_second] for name in ("serial", "peers")],
    )


if __name__ == "__main__":
    main()
else:
    pass
