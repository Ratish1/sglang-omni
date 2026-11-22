# SPDX-License-Identifier: Apache-2.0
"""Replay identical native attention operands with explicit reduction splits."""

import argparse
import json
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import difference, validate
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache


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
        sequence_length, key_heads, head_width = keys.shape
        query = first["layer0_query_after_rope"].cuda().reshape(1, -1, head_width)
        query_heads = query.shape[1]
        page_table = torch.arange(sequence_length, dtype=torch.int32, device="cuda")[
            None
        ]
        sequence_lengths = torch.tensor(
            [sequence_length], dtype=torch.int32, device="cuda"
        )
        query_offsets = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        expanded_keys = keys.double().repeat_interleave(query_heads // key_heads, dim=1)
        expanded_values = values.double().repeat_interleave(
            query_heads // key_heads, dim=1
        )
        scores = (
            torch.einsum("qhd,khd->hqk", query.double(), expanded_keys)
            * head_width**-0.5
        )
        reference = torch.einsum(
            "hqk,khd->qhd", scores.softmax(-1), expanded_values
        ).cpu()
        results = {}
        for split_count in (1, 2):
            output = flash_attn_with_kvcache(
                q=query,
                k_cache=keys[:, None],
                v_cache=values[:, None],
                page_table=page_table,
                cache_seqlens=sequence_lengths,
                cu_seqlens_q=query_offsets,
                max_seqlen_q=1,
                softmax_scale=head_width**-0.5,
                causal=True,
                window_size=(-1, -1),
                softcap=0.0,
                num_splits=split_count,
                ver=3,
            )
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
    arguments = parser.parse_args()
    main(arguments.root)
else:
    pass
