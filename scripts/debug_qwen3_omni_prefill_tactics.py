# SPDX-License-Identifier: Apache-2.0
"""Screen equivalent Talker GEMM1 tactics on held smaller peer packs."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from flashinfer.autotuner import AutoTuner, OptimizationProfile

from scripts.debug_qwen3_omni_packed_experts import (
    capture_replay,
    difference,
    load_cutlass_expert_weights,
    measure_replays,
)
from scripts.debug_qwen3_omni_prefill_prefix import routed_experts


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--peer-trace", type=Path, required=True)
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--physical-rows", type=int, required=True)
    parser.add_argument("--previous-bucket-rows", type=int, required=True)
    parser.add_argument("--anchor-tactic", type=int, required=True)
    parser.add_argument("--candidate-tactics", type=int, nargs="+", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--replays", type=int, default=500)
    parser.add_argument("--blocks", type=int, default=8)
    arguments = parser.parse_args()
    torch.cuda.set_device(arguments.gpu_id)
    device = torch.device("cuda", arguments.gpu_id)
    up_weights, down_weights = load_cutlass_expert_weights(arguments.checkpoint, device)
    geometry = torch.load(
        arguments.peer_trace / "prefill-probe-geometry-000.pt", weights_only=True
    )
    layer = torch.load(
        arguments.peer_trace / "prefill-layers.pt",
        weights_only=True,
        map_location=device,
    )[0]
    operands = (
        layer["packed_expert_input"],
        layer["packed_expert_ids"].to(torch.int32),
        layer["packed_expert_weights"].float(),
    )
    request_token_lengths = geometry["request_token_lengths"]
    live_rows = sum(request_token_lengths)
    physical_rows = arguments.physical_rows
    assert 0 < arguments.previous_bucket_rows < physical_rows < live_rows
    tuner = AutoTuner.get()
    tuner.load_configs(str(arguments.autotune_cache))
    choose = tuner.choose_one
    assert torch.equal(
        routed_experts(*operands, up_weights, down_weights)[:live_rows],
        layer["packed_expert_output"][:live_rows],
    )
    tested_packs = 0
    for peer_row, peer_token_count in enumerate(request_token_lengths):
        remaining_live_rows = live_rows - peer_token_count
        if (
            peer_row == geometry["request_row"]
            or not arguments.previous_bucket_rows < remaining_live_rows <= physical_rows
        ):
            continue
        else:
            pass
        peer_start = sum(request_token_lengths[:peer_row])
        peer_end = peer_start + peer_token_count
        packed_end = physical_rows + peer_token_count
        assert packed_end <= operands[0].size(0)
        expert_operands = tuple(
            torch.cat((tensor[:peer_start], tensor[peer_end:packed_end]))
            for tensor in operands
        ) + (up_weights, down_weights)
        with patch.object(tuner, "choose_one", wraps=choose) as choices:
            stock_output = routed_experts(*expert_operands)
        assert len(choices.call_args_list) == 2
        native_tactics = [
            int(choose(*call.args, **call.kwargs)[1]) for call in choices.call_args_list
        ]
        _, (runner,), _, tuning_inputs = choices.call_args_list[0].args
        runner.gemm_idx_for_tuning = 1
        valid_tactics = set(
            runner.get_valid_tactics(
                tuning_inputs,
                OptimizationProfile(
                    shapes=[
                        list(tensor.shape) if tensor is not None else [0]
                        for tensor in tuning_inputs
                    ],
                    tensor_initializers=[None] * len(tuning_inputs),
                ),
            )
        )
        assert arguments.anchor_tactic in valid_tactics
        second_tactic = native_tactics[1]
        anchor_output = routed_experts(
            *expert_operands, tactics=[arguments.anchor_tactic, second_tactic]
        )[:remaining_live_rows]
        stock_graph = capture_replay(expert_operands, native_tactics, stock_output)
        results = {}
        for first_tactic in arguments.candidate_tactics:
            if first_tactic not in valid_tactics:
                results[first_tactic] = {"valid": False}
                continue
            else:
                pass
            tactic_pair = [first_tactic, second_tactic]
            output = routed_experts(*expert_operands, tactics=tactic_pair)
            delta = difference(output[:remaining_live_rows], anchor_output)
            result = {"valid": True, "vs_anchor": delta}
            if delta["changed_elements"] == 0:
                candidate = capture_replay(expert_operands, tactic_pair, output)
                result["timing"] = measure_replays(
                    {"stock": stock_graph, "candidate": candidate},
                    arguments.replays,
                    arguments.blocks,
                )
            else:
                pass
            results[first_tactic] = result
        tested_packs += 1
        print(
            json.dumps(
                {
                    "omitted_peer_row": peer_row,
                    "live_rows": remaining_live_rows,
                    "physical_rows": physical_rows,
                    "native_tactics": native_tactics,
                    "anchor_tactic": arguments.anchor_tactic,
                    "profiles": results,
                }
            ),
            flush=True,
        )
    assert tested_packs > 0


if __name__ == "__main__":
    main()
else:
    pass
