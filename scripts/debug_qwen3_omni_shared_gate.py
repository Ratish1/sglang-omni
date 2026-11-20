# SPDX-License-Identifier: Apache-2.0
"""Check the actual captured shared gates across eager and captured row shapes."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional

from sglang_omni.models.qwen3_omni.components.shared_expert_gate import (
    project_shared_expert_gate,
)

PREFILL_SHAPES = (1, 16, 134, 144, 151, 160, 257, 1024)
GRAPH_REPLAY_COUNT = 300


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    arguments = parser.parse_args()
    weight_index = json.loads(
        (arguments.checkpoint / "model.safetensors.index.json").read_text()
    )["weight_map"]
    generator = torch.Generator(device="cuda").manual_seed(42)
    for line in (arguments.trace / "generation.jsonl").read_text().splitlines():
        witness = json.loads(line)
        probe = witness["pairs"]["prefill_probe"]
        layer_index = int(next(iter(probe["different_operands_by_layer"])))
        trace_directory = arguments.trace / witness["directory"] / "observe"
        layer = torch.load(trace_directory / "prefill-layers.pt", weights_only=True)[
            layer_index
        ]
        hidden_row = layer["mlp_normalized"].cuda()
        weight_name = f"talker.model.layers.{layer_index}.mlp.shared_expert_gate.weight"
        with safe_open(
            arguments.checkpoint / weight_index[weight_name], framework="pt"
        ) as weights:
            weight = weights.get_tensor(weight_name).cuda()
        reference_gate = (hidden_row.double() * weight.double()).sum().bfloat16()
        opposite_gate = (-hidden_row.double() * weight.double()).sum().bfloat16()
        shapes = []
        for token_count in PREFILL_SHAPES:
            activations = torch.randn(
                token_count,
                hidden_row.numel(),
                dtype=weight.dtype,
                device=weight.device,
                generator=generator,
            )
            token_index = min(probe["logical_token_index"], token_count - 1)
            activations[token_index].copy_(hidden_row)
            candidate = project_shared_expert_gate(activations, weight)
            assert torch.equal(candidate[token_index].reshape(()), reference_gate)
            legacy = functional.linear(activations, weight)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_candidate = project_shared_expert_gate(activations, weight)
            graph.replay()
            assert torch.equal(captured_candidate, candidate)
            activations[token_index].copy_(-hidden_row)
            graph.replay()
            assert torch.equal(
                captured_candidate[token_index].reshape(()), opposite_gate
            )
            activations[token_index].copy_(hidden_row)
            graph.replay()
            assert torch.equal(captured_candidate, candidate)
            timings = {}
            for mode in ("candidate", "linear"):
                measured_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(measured_graph):
                    if mode == "candidate":
                        measured_gate = project_shared_expert_gate(activations, weight)
                    else:
                        measured_gate = functional.linear(activations, weight)
                measured_graph.replay()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for replay_index in range(GRAPH_REPLAY_COUNT):
                    measured_graph.replay()
                end.record()
                end.synchronize()
                assert measured_gate.shape == (token_count, 1)
                timings[mode] = start.elapsed_time(end) * 1000 / GRAPH_REPLAY_COUNT
            shapes.append(
                {
                    "token_count": token_count,
                    "legacy_gate": float(legacy[token_index]),
                    "fixed_gate": float(candidate[token_index]),
                    "whole_tensor_equal_in_graph": True,
                    "changed_input_updates_replay": True,
                    "average_graph_replay_microseconds": timings,
                }
            )
        print(
            json.dumps(
                {
                    "sample_id": witness["sample_id"],
                    "layer_index": layer_index,
                    "fp64_rounded_gate": float(reference_gate),
                    "shapes": shapes,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
