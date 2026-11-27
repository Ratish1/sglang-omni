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
def audit_precision(root: Path) -> None:
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    captures = sorted(root.glob("captures/*/trajectory.pt"))
    assert captures
    reports = []
    for capture in captures:
        frames = validate(capture.parent)
        steps = json.loads(capture.with_suffix(".json").read_text())["frames"]
        keys = torch.cat([frame["stored_keys"] for frame in frames], dim=1).cuda()
        values = torch.cat([frame["stored_values"] for frame in frames], dim=1).cuda()
        comparisons = []
        for frame_index, (frame, step) in enumerate(zip(frames, steps, strict=True)):
            if frame_index == 0:
                continue
            else:
                assert step["phase"] == "decode"
            schedule = physical_schedule(
                frame["attention_scheduler_metadata"], step["graph_batch_rows"]
            )
            split_count = int(schedule[0, step["request_row"]])
            sequence_tokens = step["sequence_length"]
            for layer_index in range(step.get("captured_layers", 1)):
                query = frame[f"layer{layer_index}_query_after_rope"].cuda()
                query = query.reshape(1, -1, keys.shape[-1])
                layer_keys = keys[layer_index, :sequence_tokens]
                layer_values = values[layer_index, :sequence_tokens]
                native = replay_attention(query, layer_keys, layer_values, split_count)
                expected = frame[f"layer{layer_index}_attention_context"]
                assert torch.equal(native.cpu().flatten(), expected.flatten())
                heads_per_group = query.shape[1] // layer_keys.shape[1]
                expanded_keys = layer_keys.repeat_interleave(heads_per_group, dim=1)
                expanded_values = layer_values.repeat_interleave(heads_per_group, dim=1)
                scores = (
                    torch.einsum("qhd,khd->hqk", query.double(), expanded_keys.double())
                    * query.shape[-1] ** -0.5
                )
                probabilities = scores.softmax(-1)
                reference = torch.einsum(
                    "hqk,khd->qhd", probabilities, expanded_values.double()
                )
                assert torch.equal(
                    reference.cpu(),
                    reference_attention(query, layer_keys, layer_values),
                )
                float_scores = (
                    torch.einsum("qhd,khd->hqk", query.float(), expanded_keys.float())
                    * query.shape[-1] ** -0.5
                )
                float_output = torch.einsum(
                    "hqk,khd->qhd", float_scores.softmax(-1), expanded_values.float()
                )
                rounded_probabilities = probabilities.bfloat16().double()
                probability_output = torch.einsum(
                    "hqk,khd->qhd", rounded_probabilities, expanded_values.double()
                )
                exponentials = (scores - scores.max(-1, keepdim=True).values).exp()
                exponential_output = torch.einsum(
                    "hqk,khd->qhd",
                    exponentials.bfloat16().double(),
                    expanded_values.double(),
                ) / exponentials.sum(-1).transpose(0, 1).unsqueeze(-1)
                reference_norm = reference.norm()
                assert reference_norm > 0
                measurements = {}
                for name, output in (
                    ("native", native),
                    ("final_bfloat16_store_only", reference.bfloat16()),
                    ("float32_attention", float_output),
                    ("float32_attention_bfloat16_store", float_output.bfloat16()),
                    ("normalized_probability_bfloat16_store", probability_output),
                    ("unnormalized_exponential_bfloat16_store", exponential_output),
                ):
                    assert torch.isfinite(output).all()
                    error = output.double() - reference
                    measurements[name] = {
                        "relative_l2_error": float(error.norm() / reference_norm),
                        "maximum_absolute_error": float(error.abs().max()),
                        "bfloat16_values_different_from_native": int(
                            (output.bfloat16() != native).count_nonzero()
                        ),
                        "bfloat16_values_different_from_rounded_reference": int(
                            (output.bfloat16() != reference.bfloat16()).count_nonzero()
                        ),
                    }
                comparisons.append(
                    {
                        "frame": frame_index,
                        "layer": layer_index,
                        "sequence_tokens": sequence_tokens,
                        "native_splits": split_count,
                        "measurements": measurements,
                    }
                )
        for field, actual in (("stored_keys", keys), ("stored_values", values)):
            assert torch.equal(
                actual.cpu(), torch.cat([frame[field] for frame in frames], dim=1)
            )
        reports.append(
            {
                "sample_id": capture.parent.name,
                "capture_sha256": sha256(capture.read_bytes()).hexdigest(),
                "native_replay_exact": True,
                "cache_unchanged": True,
                "comparisons": comparisons,
            }
        )
        print(json.dumps({"sample": capture.parent.name, "checks": len(comparisons)}))
    with (root / "attention-precision-decomposition.json").open("x") as receipt:
        json.dump(
            {
                "scope": "Held native operands only. Final BF16 rounding is separated from FP32 attention and two isolated probability-rounding formulas. The latter use dense FP64 arithmetic outside the named cast and do not recreate FA3's tiled online softmax. No native continuation, speech improvement or performance is measured.",
                "fp32_matmul_precision": torch.backends.cuda.matmul.fp32_precision,
                "captures": reports,
            },
            receipt,
            indent=2,
        )


@torch.inference_mode()
def audit_layers(root: Path, fp64_reference: bool) -> None:
    torch.set_num_threads(1)
    captures = sorted(root.glob("captures/*/trajectory.pt"))
    assert captures
    report = {}
    for capture in captures:
        frames = validate(capture.parent)
        steps = json.loads(capture.with_suffix(".json").read_text())["frames"]
        keys = torch.cat([frame["stored_keys"] for frame in frames], dim=1).cuda()
        values = torch.cat([frame["stored_values"] for frame in frames], dim=1).cuda()
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
        for field, actual in (("stored_keys", keys), ("stored_values", values)):
            assert torch.equal(
                actual.cpu(), torch.cat([frame[field] for frame in frames], dim=1)
            )
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
    parser.add_argument("--precision-decomposition", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    arguments = parser.parse_args()
    torch.cuda.set_device(arguments.gpu)
    if arguments.precision_decomposition:
        assert not arguments.all_layers and not arguments.fp64_reference
        audit_precision(arguments.root)
    elif arguments.all_layers:
        audit_layers(arguments.root, arguments.fp64_reference)
    elif arguments.fp64_reference:
        parser.error("--fp64-reference requires --all-layers")
    else:
        main(arguments.root)
else:
    pass
