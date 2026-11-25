# SPDX-License-Identifier: Apache-2.0
"""Replay identical native attention operands with explicit reduction splits."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import difference, validate
from replay_qwen3_omni_scheduler_bounds import physical_schedule
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache


def replay_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    split_count: int,
) -> torch.Tensor:
    sequence_tokens, key_heads, head_width = keys.shape
    assert keys.shape == values.shape and query.shape[-1] == head_width
    assert query.shape[0] == 1 and query.shape[1] % key_heads == 0
    assert query.dtype == keys.dtype == values.dtype == torch.bfloat16
    assert split_count > 0
    return flash_attn_with_kvcache(
        q=query,
        k_cache=keys[:, None],
        v_cache=values[:, None],
        page_table=torch.arange(sequence_tokens, dtype=torch.int32, device="cuda")[
            None
        ],
        cache_seqlens=torch.tensor([sequence_tokens], dtype=torch.int32, device="cuda"),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        max_seqlen_q=1,
        softmax_scale=head_width**-0.5,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        num_splits=split_count,
        ver=3,
    )


def reference_attention(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    query_heads = query.shape[1]
    key_heads = keys.shape[1]
    head_width = keys.shape[-1]
    assert query_heads % key_heads == 0
    expanded_keys = keys.double().repeat_interleave(query_heads // key_heads, dim=1)
    expanded_values = values.double().repeat_interleave(query_heads // key_heads, dim=1)
    scores = (
        torch.einsum("qhd,khd->hqk", query.double(), expanded_keys) * head_width**-0.5
    )
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), expanded_values).cpu()


@torch.inference_mode()
def audit_layers(root: Path, fp64_reference: bool) -> None:
    torch.set_num_threads(1)
    captures = sorted(root.glob("captures/*/trajectory.pt"))
    assert captures
    report = {}
    for capture in captures:
        frames = validate(capture.parent)
        steps = json.loads(capture.with_suffix(".json").read_text())["frames"]
        prefixes = {
            field: torch.cat(
                [frames[0].get(f"decode_prefix_{field}", frames[0][f"stored_{field}"])]
                + [frame[f"stored_{field}"] for frame in frames[1:]],
                dim=1,
            )
            for field in ("keys", "values")
        }
        keys = prefixes["keys"].cuda()
        values = prefixes["values"].cuda()
        prefill_tokens = frames[0]["stored_keys"].shape[1]
        assert keys.shape == values.shape
        assert keys.shape[1] == prefill_tokens + len(frames) - 1
        checked_by_layer = [0] * keys.shape[0]
        mismatches = []
        reference_errors = []
        for frame_index, (frame, step) in enumerate(zip(frames, steps, strict=True)):
            if frame_index == 0:
                continue
            else:
                assert step["phase"] == "decode" and step["attention_num_splits"] == 0
            sequence_tokens = step["sequence_length"]
            assert sequence_tokens == prefill_tokens + frame_index
            schedule = physical_schedule(
                frame["attention_scheduler_metadata"], step["graph_batch_rows"]
            )
            split_count = int(schedule[0, step["request_row"]])
            for layer_index in range(step.get("captured_layers", 1)):
                query = frame[f"layer{layer_index}_query_after_rope"].cuda()
                query = query.reshape(1, -1, keys.shape[-1])
                actual = (
                    replay_attention(
                        query,
                        keys[layer_index, :sequence_tokens],
                        values[layer_index, :sequence_tokens],
                        split_count,
                    )
                    .cpu()
                    .flatten()
                )
                expected = frame[f"layer{layer_index}_attention_context"].flatten()
                assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                checked_by_layer[layer_index] += 1
                if not torch.equal(actual, expected):
                    mismatches.append(
                        {
                            "frame": frame_index,
                            "layer": layer_index,
                            "sequence_tokens": sequence_tokens,
                            "splits": split_count,
                            **difference(actual, expected),
                        }
                    )
                else:
                    pass
                if fp64_reference:
                    reference = reference_attention(
                        query,
                        keys[layer_index, :sequence_tokens],
                        values[layer_index, :sequence_tokens],
                    ).flatten()
                    error = actual.double() - reference
                    assert torch.isfinite(reference).all() and reference.norm() > 0
                    reference_errors.append(
                        {
                            "frame": frame_index,
                            "layer": layer_index,
                            "sequence_tokens": sequence_tokens,
                            "splits": split_count,
                            "mean_absolute_error": float(error.abs().mean()),
                            "maximum_absolute_error": float(error.abs().max()),
                            "relative_l2_error": float(error.norm() / reference.norm()),
                            "rounded_reference_changed_values": int(
                                (actual != reference.to(actual.dtype)).count_nonzero()
                            ),
                        }
                    )
                else:
                    pass
        for field, actual in (("keys", keys), ("values", values)):
            assert torch.equal(actual.cpu(), prefixes[field])
        assert all(checked_by_layer)
        report[capture.parent.name] = {
            "capture_sha256": sha256(capture.read_bytes()).hexdigest(),
            "checked_by_layer": checked_by_layer,
            "mismatches": mismatches,
            "cache_unchanged": True,
            "fp64_reference_errors": reference_errors,
        }
        print(
            json.dumps(
                {
                    name: value
                    for name, value in report[capture.parent.name].items()
                    if name != "fp64_reference_errors"
                }
            ),
            flush=True,
        )
    report_name = (
        "attention-layer-reference.json"
        if fp64_reference
        else "attention-layer-replay.json"
    )
    with (root / report_name).open("x") as receipt:
        json.dump(
            {
                "scope": "Selected request at every recorded layer boundary, using its native query, reconstructed native KV prefix and recorded split count. Equality checks effective attention output, not unrecorded physical reads or speech quality. Optional float64 errors compare with direct grouped-query scaled dot-product attention over the same complete causal prefix, not a speech-quality threshold.",
                "captures": report,
            },
            receipt,
            indent=2,
        )
    assert all(not comparison["mismatches"] for comparison in report.values())


def main(root: Path) -> None:
    torch.set_num_threads(1)
    comparison = json.loads((root / "trajectory-comparison.json").read_text())
    report = {}
    for sample_id, summary in comparison.items():
        frame_index = summary["first_differences"]["layer0_attention_context"]["frame"]
        first = validate(root / "first/captures" / sample_id)[frame_index]
        second = validate(root / "second/captures" / sample_id)[frame_index]
        for field in (
            "layer0_query_after_rope",
            "attention_read_keys",
            "attention_read_values",
        ):
            assert torch.equal(first[field], second[field]), (sample_id, field)
        keys = first["attention_read_keys"].cuda()
        values = first["attention_read_values"].cuda()
        head_width = keys.shape[-1]
        query = first["layer0_query_after_rope"].cuda().reshape(1, -1, head_width)
        reference = reference_attention(query, keys, values)
        results = {}
        for split_count in (1, 2):
            output = replay_attention(query, keys, values, split_count)
            actual = output.cpu()
            results[str(split_count)] = {
                "first_native": difference(
                    actual.flatten(), first["layer0_attention_context"].flatten()
                ),
                "second_native": difference(
                    actual.flatten(), second["layer0_attention_context"].flatten()
                ),
                "rounded_fp64": difference(actual, reference.to(actual.dtype)),
                "mean_absolute_error_fp64": float(
                    (actual.double() - reference).abs().mean()
                ),
                "maximum_absolute_error_fp64": float(
                    (actual.double() - reference).abs().max()
                ),
            }
        report[sample_id] = {"frame": frame_index, "split_replays": results}
    with (root / "attention-split-replay.json").open("x") as receipt:
        json.dump(report, receipt, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--fp64-reference", action="store_true")
    arguments = parser.parse_args()
    if arguments.all_layers:
        audit_layers(arguments.root, arguments.fp64_reference)
    elif arguments.fp64_reference:
        parser.error("--fp64-reference requires --all-layers")
    else:
        main(arguments.root)
else:
    pass
