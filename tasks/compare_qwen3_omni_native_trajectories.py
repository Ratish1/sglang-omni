# SPDX-License-Identifier: Apache-2.0
"""Validate native capture boundaries before comparing selected speech trajectories."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import torch


def difference(
    left: torch.Tensor, right: torch.Tensor
) -> dict[str, bool | int | float | list[int]]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "same_layout": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    else:
        changed = left != right
        return {
            "same_layout": True,
            "changed_values": int(changed.count_nonzero()),
            "total_values": left.numel(),
            "max_absolute_difference": float(
                (left.float() - right.float()).abs().nan_to_num().max()
            ),
        }


def validate(directory: Path) -> list[dict[str, torch.Tensor]]:
    frames = torch.load(
        directory / "trajectory.pt", weights_only=True, map_location="cpu"
    )
    metadata = json.loads((directory / "trajectory.json").read_text())
    assert len(frames) == len(metadata["frames"]) == len(metadata["output_ids"])
    assert metadata["finish_reason"] == "stop"
    assert [int(frame["codec_ids"][0]) for frame in frames] == metadata["output_ids"]
    assert metadata["frames"][0]["phase"] == "prefill"
    assert all(row["phase"] == "decode" for row in metadata["frames"][1:])
    assert (
        frames[0]["logical_slots"].unique().numel()
        == frames[0]["logical_slots"].numel()
    )
    assert int(frames[0]["logical_slots"].min()) > 0
    for frame_index, (frame, step) in enumerate(
        zip(frames, metadata["frames"], strict=True)
    ):
        assert frame["stored_keys"].shape == frame["stored_values"].shape
        assert torch.isfinite(frame["predictor_query"]).all()
        assert torch.isfinite(frame["feedback"]).all()
        assert step["request_ids"][step["request_row"]] == metadata["request_id"]
        if frame_index == 0:
            assert frame["stored_keys"].shape[1] == step["sequence_length"]
            assert torch.equal(
                frame["logical_slots"][-len(frame["output_slots"]) :],
                frame["output_slots"],
            )
        else:
            assert torch.equal(
                frame["queued_feedback"], frames[frame_index - 1]["feedback"]
            )
            assert torch.equal(
                frame["consumed_input"],
                frame["queued_feedback"].reshape(-1)
                + frame["text_embedding"].reshape(-1),
            )
            assert (
                step["sequence_length"]
                == metadata["frames"][frame_index - 1]["sequence_length"] + 1
            )
            assert frame["stored_keys"].shape[1] == 1
            assert step["graph_batch_rows"] >= len(step["request_ids"])
            if "layer0_contribution" in frame:
                assert torch.equal(
                    frame["layer0_contribution"], frame["consumed_input"]
                )
                assert torch.equal(
                    frame["layer0_positions"].flatten(),
                    frame["rotary_positions"].flatten(),
                )
                assert torch.equal(
                    frame["attention_read_slots"], frame["attention_owned_slots"]
                )
                assert torch.equal(
                    frame["attention_read_slots"][-1:], frame["output_slots"]
                )
                assert (
                    int(frame["attention_sequence_lengths"][step["request_row"]])
                    == step["sequence_length"]
                )
                if "scheduled_sequence_lengths_cpu" in frame:
                    assert torch.equal(
                        frame["scheduled_sequence_lengths_cpu"],
                        frame["attention_sequence_lengths"][: len(step["request_ids"])],
                    )
                    if step["cuda_graph"]:
                        assert torch.equal(
                            frame["graph_sequence_lengths_cpu"],
                            frame["attention_sequence_lengths"],
                        )
                    else:
                        pass
                    if step["attention_uses_cpu_lengths"]:
                        assert step["attention_bound_tokens"] == int(
                            frame["attention_sequence_lengths"].max()
                        )
                    else:
                        pass
                else:
                    pass
                for field in ("keys", "values"):
                    expected_prefix = torch.cat(
                        [
                            previous[f"stored_{field}"][0]
                            for previous in frames[: frame_index + 1]
                        ]
                    )
                    assert torch.equal(
                        frame[f"attention_read_{field}"], expected_prefix
                    ), (frame_index, field)
                assert torch.equal(
                    frame["layer0_key_after_rope"].flatten(),
                    frame["attention_read_keys"][-1].flatten(),
                )
                assert torch.equal(
                    frame["layer0_value_before_cache"].flatten(),
                    frame["attention_read_values"][-1].flatten(),
                )
                if "captured_layers" in step:
                    assert step["captured_layers"] == frame["stored_keys"].shape[0]
                    for layer_index in range(step["captured_layers"]):
                        prefix = f"layer{layer_index}_"
                        assert torch.equal(
                            frame[f"{prefix}positions"].flatten(),
                            frame["rotary_positions"].flatten(),
                        )
                        for operand, field in (
                            ("key_after_rope", "stored_keys"),
                            ("value_before_cache", "stored_values"),
                        ):
                            assert torch.equal(
                                frame[f"{prefix}{operand}"].flatten(),
                                frame[field][layer_index].flatten(),
                            ), (frame_index, layer_index, operand)
                        if layer_index > 0:
                            previous = f"layer{layer_index - 1}_"
                            assert torch.equal(
                                frame[f"{prefix}contribution"],
                                frame[f"{previous}layer_output"],
                            ), (frame_index, layer_index, "contribution")
                            assert torch.equal(
                                frame[f"{prefix}previous_residual"],
                                frame[f"{previous}mlp_residual"],
                            ), (frame_index, layer_index, "residual")
                        else:
                            pass
                        for name, values in frame.items():
                            if name.startswith(f"{prefix}peer_"):
                                selected = name.replace(f"{prefix}peer_", prefix, 1)
                                assert values.shape[0] == step["graph_batch_rows"]
                                assert torch.equal(
                                    values[step["request_row"]], frame[selected]
                                ), (frame_index, layer_index, name)
                            else:
                                pass
                else:
                    pass
            else:
                pass
    return frames


def main(root: Path) -> None:
    first_root = root / "first" / "captures"
    second_root = root / "second" / "captures"
    sample_ids = sorted(path.name for path in first_root.iterdir() if path.is_dir())
    assert sample_ids == sorted(
        path.name for path in second_root.iterdir() if path.is_dir()
    )
    report = {}
    for sample_id in sample_ids:
        first = validate(first_root / sample_id)
        second = validate(second_root / sample_id)
        first_metadata = json.loads(
            (first_root / sample_id / "trajectory.json").read_text()
        )
        second_metadata = json.loads(
            (second_root / sample_id / "trajectory.json").read_text()
        )
        fields = (
            "prefill_embeddings",
            "input_ids",
            "positions",
            "rotary_positions",
            "predictor_query",
            "logits",
            "codec_ids",
            "feedback",
            "text_embedding",
            "consumed_input",
            "stored_keys",
            "stored_values",
            "attention_read_keys",
            "attention_read_values",
            "layer0_contribution",
            "layer0_positions",
            "layer0_attention_normalized",
            "layer0_attention_residual",
            "layer0_qkv_projection",
            "layer0_query_after_rope",
            "layer0_key_after_rope",
            "layer0_value_before_cache",
            "layer0_attention_context",
            "layer0_attention_projection",
            "layer0_mlp_normalized",
            "layer0_mlp_residual",
            "layer0_router_logits",
            "layer0_expert_ids",
            "layer0_expert_weights",
            "layer0_routed_experts",
            "layer0_shared_expert",
            "layer0_shared_gate",
            "layer0_layer_output",
        )
        first_differences = {}
        for frame_index, (left, right) in enumerate(zip(first, second)):
            for field in fields:
                if field in first_differences:
                    continue
                elif field not in left or field not in right:
                    assert (field in left) == (field in right)
                    continue
                elif not torch.equal(left[field], right[field]):
                    first_differences[field] = {
                        "frame": frame_index,
                        **difference(left[field], right[field]),
                    }
                else:
                    pass
        prefill = first[0]["prefill_embeddings"]
        other_prefill = second[0]["prefill_embeddings"]
        if prefill.shape == other_prefill.shape:
            changed_rows = (prefill != other_prefill).any(dim=-1).nonzero().flatten()
            changed_token_ids = (
                first[0]["input_ids"].index_select(0, changed_rows).unique().tolist()
            )
        else:
            changed_rows = torch.empty(0, dtype=torch.long)
            changed_token_ids = []
        layer_comparisons = []
        prefix_codes_equal = True
        prefix_keys_equal = True
        prefix_values_equal = True
        for frame_index, (left, right) in enumerate(zip(first, second)):
            left_metadata = first_metadata["frames"][frame_index]
            right_metadata = second_metadata["frames"][frame_index]
            if (
                "captured_layers" in left_metadata
                and "captured_layers" in right_metadata
            ):
                assert (
                    left_metadata["captured_layers"]
                    == right_metadata["captured_layers"]
                )
                layer_changes = {}
                for layer_index in range(left_metadata["captured_layers"]):
                    prefix = f"layer{layer_index}_"
                    fields = {
                        name
                        for name in left
                        if name.startswith(prefix)
                        and not name.startswith(f"{prefix}peer_")
                    }
                    assert fields == {
                        name
                        for name in right
                        if name.startswith(prefix)
                        and not name.startswith(f"{prefix}peer_")
                    }
                    changes = {}
                    for name in sorted(fields):
                        assert torch.isfinite(left[name]).all(), (
                            frame_index,
                            name,
                            "first",
                        )
                        assert torch.isfinite(right[name]).all(), (
                            frame_index,
                            name,
                            "second",
                        )
                        if not torch.equal(left[name], right[name]):
                            changes[name.removeprefix(prefix)] = difference(
                                left[name], right[name]
                            )
                        else:
                            pass
                    layer_changes[str(layer_index)] = changes
                layer_comparisons.append(
                    {
                        "frame": frame_index,
                        "prior_codes_equal": prefix_codes_equal,
                        "prior_keys_equal": prefix_keys_equal,
                        "prior_values_equal": prefix_values_equal,
                        "consumed_input_equal": torch.equal(
                            left["consumed_input"], right["consumed_input"]
                        ),
                        "codes_equal": torch.equal(
                            left["codec_ids"], right["codec_ids"]
                        ),
                        "layers": layer_changes,
                    }
                )
            else:
                pass
            prefix_codes_equal &= torch.equal(left["codec_ids"], right["codec_ids"])
            prefix_keys_equal &= torch.equal(left["stored_keys"], right["stored_keys"])
            prefix_values_equal &= torch.equal(
                left["stored_values"], right["stored_values"]
            )
        report[sample_id] = {
            "frames": [len(first), len(second)],
            "capture_sha256": [
                sha256(
                    (capture_root / sample_id / "trajectory.pt").read_bytes()
                ).hexdigest()
                for capture_root in (first_root, second_root)
            ],
            "prefill_geometry": [
                metadata["frames"][0] for metadata in (first_metadata, second_metadata)
            ],
            "changed_prefill_rows": changed_rows.tolist(),
            "changed_prefill_token_ids": changed_token_ids,
            "first_differences": first_differences,
            "layer_comparisons": layer_comparisons,
        }
    destination = root / "trajectory-comparison.json"
    with destination.open("x") as result:
        json.dump(report, result, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    main(arguments.root)
else:
    pass
