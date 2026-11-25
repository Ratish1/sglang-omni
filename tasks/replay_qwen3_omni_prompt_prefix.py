# SPDX-License-Identifier: Apache-2.0
"""Isolate first-decode prefix effects using validated native captures."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import difference, validate
from replay_qwen3_omni_native_attention import reference_attention, replay_attention
from replay_qwen3_omni_scheduler_bounds import physical_schedule


@torch.inference_mode()
def main(root: Path, sample_id: str) -> None:
    torch.set_num_threads(1)
    report = {}
    for policy in ("stock", "candidate"):
        directories = [
            root / f"{policy}-prompt-{prompt}" / "captures" / sample_id
            for prompt in ("first", "second")
        ]
        captures = [validate(directory) for directory in directories]
        steps = [
            json.loads((directory / "trajectory.json").read_text())["frames"][1]
            for directory in directories
        ]
        openings = [capture[0] for capture in captures]
        frames = [capture[1] for capture in captures]
        for field in ("input_ids", "codec_ids", "feedback"):
            assert torch.equal(openings[0][field], openings[1][field]), field
        for field in (
            "positions",
            "rotary_positions",
            "consumed_input",
            "text_embedding",
            "layer0_qkv_projection",
            "layer0_query_after_rope",
        ):
            assert torch.equal(frames[0][field], frames[1][field]), field
        split_counts = [
            int(
                physical_schedule(
                    frame["attention_scheduler_metadata"], step["graph_batch_rows"]
                )[0, step["request_row"]]
            )
            for frame, step in zip(frames, steps, strict=True)
        ]
        assert split_counts[0] == split_counts[1]
        layer_count = openings[0]["stored_keys"].shape[0]
        assert all(step["captured_layers"] == layer_count for step in steps)
        keys = [
            torch.cat((opening["stored_keys"], frame["stored_keys"]), dim=1).cuda()
            for opening, frame in zip(openings, frames, strict=True)
        ]
        values = [
            torch.cat((opening["stored_values"], frame["stored_values"]), dim=1).cuda()
            for opening, frame in zip(openings, frames, strict=True)
        ]
        layers = []
        for layer_index in range(layer_count):
            prefix = f"layer{layer_index}_"
            for arm_index, frame in enumerate(frames):
                query = (
                    frame[f"{prefix}query_after_rope"]
                    .cuda()
                    .reshape(1, -1, keys[arm_index].shape[-1])
                )
                replay = replay_attention(
                    query,
                    keys[arm_index][layer_index],
                    values[arm_index][layer_index],
                    split_counts[arm_index],
                )
                assert torch.equal(
                    replay.cpu().flatten(),
                    frame[f"{prefix}attention_context"].flatten(),
                ), (policy, arm_index, layer_index)
            layers.append(
                {
                    "layer": layer_index,
                    "prefill_keys": difference(
                        openings[0]["stored_keys"][layer_index],
                        openings[1]["stored_keys"][layer_index],
                    ),
                    "prefill_values": difference(
                        openings[0]["stored_values"][layer_index],
                        openings[1]["stored_values"][layer_index],
                    ),
                    "decode": {
                        field: difference(
                            frames[0][prefix + field], frames[1][prefix + field]
                        )
                        for field in (
                            "query_after_rope",
                            "attention_context",
                            "attention_projection",
                            "mlp_normalized",
                            "router_logits",
                            "expert_ids",
                            "expert_weights",
                            "routed_experts",
                            "shared_expert",
                            "layer_output",
                        )
                    },
                }
            )
        assert torch.equal(keys[0][0, -1], keys[1][0, -1])
        assert torch.equal(values[0][0, -1], values[1][0, -1])
        query = (
            frames[0]["layer0_query_after_rope"]
            .cuda()
            .reshape(1, -1, keys[0].shape[-1])
        )
        baseline = frames[0]["layer0_attention_context"].flatten()
        baseline_reference = reference_attention(query, keys[0][0], values[0][0])
        prefix_crosses = {}
        for key_index in (0, 1):
            for value_index in (0, 1):
                output = replay_attention(
                    query,
                    keys[key_index][0],
                    values[value_index][0],
                    split_counts[0],
                ).cpu()
                reference = reference_attention(
                    query, keys[key_index][0], values[value_index][0]
                )
                prefix_crosses[f"keys_{key_index}_values_{value_index}"] = {
                    "native_output_difference": difference(output.flatten(), baseline),
                    "fp64_mean_absolute_change": float(
                        (reference - baseline_reference).abs().mean()
                    ),
                    "fp64_maximum_absolute_change": float(
                        (reference - baseline_reference).abs().max()
                    ),
                    "native_mean_absolute_error_fp64": float(
                        (output.double() - reference).abs().mean()
                    ),
                    "native_maximum_absolute_error_fp64": float(
                        (output.double() - reference).abs().max()
                    ),
                }
        report[policy] = {
            "capture_sha256": [
                sha256((directory / "trajectory.pt").read_bytes()).hexdigest()
                for directory in directories
            ],
            "native_attention_replays_exact": 2 * layer_count,
            "layer0_query_and_current_kv_equal": True,
            "first_decode_split_count": split_counts[0],
            "layer0_prefix_crosses": prefix_crosses,
            "layers": layers,
            "first_changed_expert_layer": next(
                (
                    layer_index
                    for layer_index in range(layer_count)
                    if not torch.equal(
                        frames[0][f"layer{layer_index}_expert_ids"],
                        frames[1][f"layer{layer_index}_expert_ids"],
                    )
                ),
                None,
            ),
            "first_decode_codes": [frame["codec_ids"].tolist() for frame in frames],
        }
        print(
            json.dumps(
                {
                    "policy": policy,
                    "native_attention_replays_exact": 2 * layer_count,
                    "first_changed_expert_layer": report[policy][
                        "first_changed_expert_layer"
                    ],
                    "layer0_prefix_crosses": prefix_crosses,
                }
            ),
            flush=True,
        )
    with (root / "prompt-prefix-replay.json").open("x") as receipt:
        json.dump(
            {
                "scope": "First decode only. All recorded layers must reproduce native attention. Layer-zero crosses hold query, current key/value and split count while substituting captured prefix keys/values. They do not regenerate speech or remove peer prefill geometry as a cause of prefix differences.",
                "sample_id": sample_id,
                "policies": report,
            },
            receipt,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--sample-id", required=True)
    arguments = parser.parse_args()
    main(arguments.root, arguments.sample_id)
else:
    pass
