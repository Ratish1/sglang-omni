"""Verify native routing intervention on complete captured speech lifetimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from transformers import Qwen3OmniMoeConfig


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioning-directory", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--rounded-routing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        local_files_only=True,
    ).talker_config.text_config
    records = []
    for directory in sorted(arguments.conditioning_directory.iterdir()):
        if not directory.is_dir():
            continue
        else:
            pass
        finished = torch.load(directory / "finished.pt", weights_only=True)
        snapshots = sorted(directory.glob("*-operators.pt"))
        assert [source.name for source in snapshots] == [
            f"{frame:04d}-operators.pt" for frame in range(len(finished["output_ids"]))
        ]
        layer_batches = 0
        graph_layer_batches = 0
        prefill_layer_batches = 0
        weight_values = 0
        changed_native_weights = 0
        checkpoint_weight_disagreements = 0
        for source in snapshots:
            layers = torch.load(source, weights_only=True)
            assert len(layers) == configuration.num_hidden_layers
            for layer in layers:
                logits = layer["batch_router_logits"].to(device)
                observed_indices = layer["batch_expert_indices"].to(device)
                observed_weights = layer["batch_expert_weights"].to(device)
                native_weights, native_indices = moe_fused_gate(
                    logits,
                    None,
                    configuration.num_experts_per_tok,
                    scoring_func="softmax",
                    renormalize=configuration.norm_topk_prob,
                )
                assert torch.equal(native_indices, observed_indices)
                rounded_weights = native_weights.to(logits.dtype).float()
                expected_weights = (
                    rounded_weights if arguments.rounded_routing else native_weights
                )
                assert torch.equal(expected_weights, observed_weights)
                probabilities = logits.float().softmax(dim=-1)
                checkpoint_weights = probabilities.gather(1, native_indices.long())
                if configuration.norm_topk_prob:
                    checkpoint_weights /= checkpoint_weights.sum(dim=-1, keepdim=True)
                else:
                    pass
                checkpoint_weights = checkpoint_weights.to(logits.dtype).float()
                checkpoint_weight_disagreements += int(
                    torch.count_nonzero(checkpoint_weights != rounded_weights)
                )
                changed_native_weights += int(
                    torch.count_nonzero(native_weights != rounded_weights)
                )
                weight_values += native_weights.numel()
                layer_batches += 1
                graph_layer_batches += int(layer["batch_is_graph"])
                prefill_layer_batches += int(layer["layout_is_prefill"])
        record = {
            "request_id": directory.name,
            "frames": len(snapshots),
            "layer_batches": layer_batches,
            "graph_layer_batches": graph_layer_batches,
            "prefill_layer_batches": prefill_layer_batches,
            "weight_values": weight_values,
            "changed_native_weights": changed_native_weights,
            "checkpoint_weight_disagreements": checkpoint_weight_disagreements,
            "native_indices_exact": True,
            "expected_weights_exact": True,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
    assert records
    with arguments.output.open("x") as output:
        json.dump(
            {"rounded_routing": arguments.rounded_routing, "records": records},
            output,
            indent=2,
        )


if __name__ == "__main__":
    main()
else:
    pass
