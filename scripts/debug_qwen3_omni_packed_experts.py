# SPDX-License-Identifier: Apache-2.0
"""Replay the actual packed first-layer expert tensors and cross selected tactics."""

import argparse
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from flashinfer.autotuner import AutoTuner, OptimizationProfile
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.test_utils import publish_build_topology

from scripts.debug_qwen3_omni_peer_shape import read_weights
from scripts.debug_qwen3_omni_prefill_prefix import (
    compare_activation_store,
    routed_experts,
)

SGLANG_FIXED_MOE_TILES = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
}


def load_cutlass_expert_weights(
    checkpoint: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    config = json.loads((checkpoint / "config.json").read_bytes())["talker_config"][
        "text_config"
    ]
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
            checkpoint,
            tuple(
                f"mlp.experts.{expert_index}.{projection}_proj"
                for projection in ("gate", "up", "down")
            ),
        )
        first_projection[expert_index] = torch.cat((up, gate))
        second_projection[expert_index] = down
    return first_projection, second_projection


def difference(first: torch.Tensor, second: torch.Tensor) -> dict[str, int | float]:
    assert first.shape == second.shape
    delta = first.float() - second.float()
    return {
        "changed_elements": int(delta.count_nonzero()),
        "changed_rows": int(delta.reshape(-1, delta.shape[-1]).ne(0).any(-1).sum()),
        "max_abs_difference": float(delta.abs().max()),
    }


def capture_replay(
    expert_operands: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    profile_ids: list[int],
    eager_output: torch.Tensor,
) -> torch.cuda.CUDAGraph:
    replay_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(replay_graph):
        graph_output = routed_experts(*expert_operands, tactics=profile_ids)
    replay_graph.replay()
    assert torch.equal(graph_output, eager_output)
    return replay_graph


def measure_replays(
    graphs: dict[str, torch.cuda.CUDAGraph],
    replays_per_observation: int,
    paired_blocks: int,
) -> dict[str, int | dict[str, list[float]] | dict[str, float]]:
    timings_microseconds: dict[str, list[float]] = {
        trial_name: [] for trial_name in graphs
    }
    names = tuple(graphs)
    for trial_name in (*names, *reversed(names)) * paired_blocks:
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays_per_observation):
            graphs[trial_name].replay()
        stop.record()
        stop.synchronize()
        timings_microseconds[trial_name].append(
            start.elapsed_time(stop) * 1000 / replays_per_observation
        )
    return {
        "replays_per_observation": replays_per_observation,
        "microseconds_per_replay": timings_microseconds,
        "median_microseconds_per_replay": {
            trial_name: median(values)
            for trial_name, values in timings_microseconds.items()
        },
    }


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--serial-trace", type=Path, required=True)
    parser.add_argument("--peer-trace", type=Path, required=True)
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--timing-replays", type=int, default=0)
    parser.add_argument("--timing-blocks", type=int, default=8)
    parser.add_argument("--find-fastest-equivalent-peer-gemm1", action="store_true")
    parser.add_argument("--sglang-fixed-tiles", action="store_true")
    arguments = parser.parse_args()
    if arguments.timing_replays < 0 or arguments.timing_blocks < 1:
        parser.error("Use a nonnegative replay count and a positive block count")
    elif arguments.find_fastest_equivalent_peer_gemm1 and not arguments.timing_replays:
        parser.error("Equivalent tactic search requires replay timing")
    elif arguments.sglang_fixed_tiles and not arguments.timing_replays:
        parser.error("Fixed-tile comparison requires replay timing")
    else:
        pass
    torch.cuda.set_device(arguments.gpu_id)
    device = torch.device("cuda", arguments.gpu_id)
    first_projection, second_projection = load_cutlass_expert_weights(
        arguments.checkpoint, device
    )
    expert_count = first_projection.size(0)

    tuner = AutoTuner.get()
    tuner.load_configs(str(arguments.autotune_cache))
    assert not tuner.is_tuning_mode
    choose_profile = tuner.choose_one
    operands = {}
    outputs = {}
    target_indices = {}
    target_ranges = {}
    tactic_pairs = {}
    sequences = {}
    valid_peer_profiles: list[int] = []
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
        target_start = sum(geometry["request_token_lengths"][: geometry["request_row"]])
        target_count = geometry["request_token_lengths"][geometry["request_row"]]
        target_ranges[name] = slice(target_start, target_start + target_count)
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
        if name == "peers" and arguments.find_fastest_equivalent_peer_gemm1:
            first_call = choices.call_args_list[0]
            _, (runner,), _, tuning_inputs = first_call.args
            runner.gemm_idx_for_tuning = 1
            optimization_profile = OptimizationProfile(
                shapes=[
                    list(tensor.shape) if tensor is not None else [0]
                    for tensor in tuning_inputs
                ],
                tensor_initializers=[None] * len(tuning_inputs),
            )
            valid_peer_profiles = runner.get_valid_tactics(
                tuning_inputs, optimization_profile
            )
            assert tactic_pairs[name][0] in valid_peer_profiles
        else:
            pass

    assert all(
        torch.equal(a[target_indices["serial"]], b[target_indices["peers"]])
        for a, b in zip(operands["serial"][:3], operands["peers"][:3], strict=True)
    ), sequences
    full_prefix_inputs_equal = {
        name: torch.equal(
            first[target_ranges["serial"]], second[target_ranges["peers"]]
        )
        for name, first, second in zip(
            ("activation", "expert_ids", "expert_weights"),
            operands["serial"][:3],
            operands["peers"][:3],
            strict=True,
        )
    }
    effects = {}
    for name, other in (("serial", "peers"), ("peers", "serial")):
        target_index = target_indices[name]
        other_index = target_indices[other]
        native_first, native_second = tactic_pairs[name]
        alternate_first, alternate_second = tactic_pairs[other]
        trials = {}
        timed_graphs: dict[str, torch.cuda.CUDAGraph] = {}
        for trial_name, pair in (
            ("explicit_reference", [native_first, native_second]),
            ("swap_first_projection", [alternate_first, native_second]),
            ("swap_second_projection", [native_first, alternate_second]),
            ("swap_both", [alternate_first, alternate_second]),
        ):
            result = routed_experts(*operands[name], tactics=pair)
            trials[trial_name] = {
                "profile_ids": pair,
                "vs_native": difference(
                    result[target_index], outputs[name][target_index]
                ),
                "vs_other": difference(
                    result[target_index], outputs[other][other_index]
                ),
                "full_prefix_vs_native": difference(
                    result[target_ranges[name]], outputs[name][target_ranges[name]]
                ),
                "full_prefix_vs_other": difference(
                    result[target_ranges[name]], outputs[other][target_ranges[other]]
                ),
            }
            if arguments.timing_replays and trial_name in (
                "explicit_reference",
                "swap_first_projection",
            ):
                timed_graphs[trial_name] = capture_replay(operands[name], pair, result)
            else:
                pass
        assert trials["explicit_reference"]["vs_native"]["changed_elements"] == 0
        effects[name] = {
            "valid_sequence_lengths": sequences[name],
            "physical_token_count": operands[name][0].size(0),
            "target_index": target_index,
            "trials": trials,
        }
        if timed_graphs:
            effects[name]["graph_replay_timing"] = measure_replays(
                timed_graphs, arguments.timing_replays, arguments.timing_blocks
            )
        else:
            pass
        if name == "peers" and valid_peer_profiles:
            assert alternate_first in valid_peer_profiles
            raw_tokens = sum(sequences[name])
            anchor_output = routed_experts(
                *operands[name], tactics=[alternate_first, native_second]
            )
            search_results = {}
            for profile_id in valid_peer_profiles:
                pair = [profile_id, native_second]
                output = routed_experts(*operands[name], tactics=pair)
                live_difference = difference(
                    output[:raw_tokens], anchor_output[:raw_tokens]
                )
                result = {"vs_anchor_live_prefix": live_difference}
                if live_difference["changed_elements"] == 0:
                    candidate = capture_replay(operands[name], pair, output)
                    result["graph_replay_timing"] = measure_replays(
                        {
                            "native": timed_graphs["explicit_reference"],
                            "candidate": candidate,
                        },
                        arguments.timing_replays,
                        arguments.timing_blocks,
                    )
                else:
                    pass
                search_results[profile_id] = result
            effects[name]["equivalent_gemm1_search"] = {
                "anchor_gemm1_profile": alternate_first,
                "native_gemm2_profile": native_second,
                "profile_results": search_results,
            }
        else:
            pass

    if arguments.sglang_fixed_tiles:
        assert all(full_prefix_inputs_equal.values())
        up_weights, gate_weights = first_projection.chunk(2, dim=1)
        triton_weights = torch.cat((gate_weights, up_weights), dim=1)
        runner_config = MoeRunnerConfig(
            num_experts=expert_count,
            num_local_experts=expert_count,
            inplace=False,
            gate_up_interleaved=False,
        )
        with TemporaryDirectory() as rendezvous_directory:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=arguments.gpu_id,
                distributed_init_method=(Path(rendezvous_directory) / "rank").as_uri(),
                backend="gloo",
            )
            publish_build_topology(tp_size=1, ep_size=1, pp_size=1)
            initialize_model_parallel(backend="gloo")
            try:
                triton_outputs: dict[str, torch.Tensor] = {}
                with override_config(SGLANG_FIXED_MOE_TILES.copy()):
                    for name, expert_operands in operands.items():
                        activation, expert_ids, routing_weights, _, _ = expert_operands
                        assert activation.size(0) > 32
                        live_rows = sum(sequences[name])
                        topk = StandardTopKOutput(routing_weights, expert_ids, None)
                        triton_output = fused_experts(
                            activation,
                            triton_weights,
                            second_projection,
                            topk,
                            runner_config,
                        )
                        assert torch.isfinite(triton_output[:live_rows]).all()
                        triton_outputs[name] = triton_output[
                            target_ranges[name]
                        ].clone()
                        triton_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(triton_graph):
                            captured_output = fused_experts(
                                activation,
                                triton_weights,
                                second_projection,
                                topk,
                                runner_config,
                            )
                        triton_graph.replay()
                        assert torch.equal(
                            captured_output[:live_rows], triton_output[:live_rows]
                        )
                        native_graph = capture_replay(
                            expert_operands, tactic_pairs[name], outputs[name]
                        )
                        effects[name]["sglang_fixed_tiles"] = {
                            "vs_sglang_serial": difference(
                                triton_outputs[name], triton_outputs["serial"]
                            ),
                            "vs_native": difference(
                                triton_outputs[name], outputs[name][target_ranges[name]]
                            ),
                            "timing": measure_replays(
                                {"native": native_graph, "sglang": triton_graph},
                                arguments.timing_replays,
                                arguments.timing_blocks,
                            ),
                        }
            finally:
                destroy_model_parallel()
                destroy_distributed_environment()
    else:
        pass

    print(
        json.dumps(
            {
                "packed_replays_exact": True,
                "full_prefix_inputs_equal": full_prefix_inputs_equal,
                "effects": effects,
            },
            sort_keys=True,
        )
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
