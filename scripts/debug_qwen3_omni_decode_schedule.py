"""Compare isolated FA3 decode with scheduling from the recorded peer batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, TypedDict

import torch
from pydantic import BaseModel
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata

METADATA_ALIGNMENT = 4
METADATA_VECTORS = 4
DECODE_PADDING_TOKENS = 1


class DecodeBatch(BaseModel):
    mode: Literal["DECODE"]
    active_row: int
    callback_sequence_lengths: list[int]


class CapturedCache(TypedDict):
    keys: list[torch.Tensor]
    values: list[torch.Tensor]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-directory", type=Path, action="append", required=True)
    parser.add_argument("--page-table-tokens", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    summaries = []
    for directory in arguments.trace_directory:
        files = sorted(directory.glob("*-output.pt"))
        prefill: CapturedCache = torch.load(files[0], weights_only=True)
        layer_count = len(prefill["keys"])
        prefix_tokens, key_heads, head_channels = prefill["keys"][0].shape
        maximum_tokens = prefix_tokens + len(files) - 1
        assert maximum_tokens <= arguments.page_table_tokens
        keys = torch.empty(
            (layer_count, maximum_tokens, 1, key_heads, head_channels),
            dtype=torch.bfloat16,
            device=arguments.gpu,
        )
        values = torch.empty_like(keys)
        keys[:, :prefix_tokens].copy_(torch.stack(prefill["keys"]).unsqueeze(2))
        values[:, :prefix_tokens].copy_(torch.stack(prefill["values"]).unsqueeze(2))
        page_table = torch.zeros(
            (1, arguments.page_table_tokens),
            dtype=torch.int32,
            device=arguments.gpu,
        )
        page_table[0, :maximum_tokens] = torch.arange(
            maximum_tokens, dtype=torch.int32, device=arguments.gpu
        )
        reports = []
        output_file = arguments.output_directory / f"{directory.name}.jsonl"
        with output_file.open("w") as output:
            for frame_index, filename in enumerate(files[1:], start=1):
                captured: CapturedCache = torch.load(filename, weights_only=True)
                token_count = prefix_tokens + frame_index
                keys[:, token_count - 1 : token_count].copy_(
                    torch.stack(captured["keys"]).unsqueeze(2)
                )
                values[:, token_count - 1 : token_count].copy_(
                    torch.stack(captured["values"]).unsqueeze(2)
                )
                batch = DecodeBatch.model_validate_json(
                    (directory / f"{frame_index:04d}-batch.json").read_bytes()
                )
                assert batch.callback_sequence_lengths[batch.active_row] == token_count
                snapshots: list[dict[str, torch.Tensor]] = torch.load(
                    directory / f"{frame_index:04d}-operators.pt", weights_only=True
                )
                query_heads = snapshots[0]["rotated_query"].numel() // head_channels
                batch_size = snapshots[0]["batch_attention_values"].shape[0]
                padding_requests = batch_size - len(batch.callback_sequence_lengths)
                assert padding_requests >= 0
                if padding_requests:
                    assert bool(snapshots[0]["batch_is_graph"])
                else:
                    pass
                # note (ratish): FA3 graph buffers reset inactive sequence lengths to one token.
                scheduled_lengths = (
                    batch.callback_sequence_lengths
                    + [DECODE_PADDING_TOKENS] * padding_requests
                )
                batch_lengths = torch.tensor(
                    scheduled_lengths,
                    dtype=torch.int32,
                    device=arguments.gpu,
                )
                batch_metadata = get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=1,
                    max_seqlen_k=arguments.page_table_tokens,
                    num_heads=query_heads,
                    num_heads_k=key_heads,
                    headdim=head_channels,
                    cache_seqlens=batch_lengths,
                    cu_seqlens_q=torch.arange(
                        batch_size + 1, dtype=torch.int32, device=arguments.gpu
                    ),
                    page_size=1,
                    causal=True,
                ).cpu()
                rounded_batch_size = (
                    (batch_size + METADATA_ALIGNMENT - 1)
                    // METADATA_ALIGNMENT
                    * METADATA_ALIGNMENT
                )
                assert batch_metadata.numel() == (
                    rounded_batch_size * METADATA_VECTORS + 1
                )
                batch_order = batch_metadata[
                    2 * rounded_batch_size : 2 * rounded_batch_size + batch_size
                ]
                virtual_row = (batch_order == batch.active_row).nonzero().item()
                lengths = batch_lengths[batch.active_row : batch.active_row + 1]
                cumulative_query_tokens = torch.arange(
                    2, dtype=torch.int32, device=arguments.gpu
                )
                single_metadata = get_scheduler_metadata(
                    batch_size=1,
                    max_seqlen_q=1,
                    max_seqlen_k=arguments.page_table_tokens,
                    num_heads=query_heads,
                    num_heads_k=key_heads,
                    headdim=head_channels,
                    cache_seqlens=lengths,
                    cu_seqlens_q=cumulative_query_tokens,
                    page_size=1,
                    causal=True,
                )
                selected_splits = int(batch_metadata[virtual_row])
                assert 0 < selected_splits <= int(single_metadata[0])
                matched_metadata = torch.zeros_like(single_metadata)
                for vector in (0, 1, 3):
                    matched_metadata[vector * METADATA_ALIGNMENT] = batch_metadata[
                        vector * rounded_batch_size + virtual_row
                    ]
                for layer_index, snapshot in enumerate(snapshots):
                    query = (
                        snapshot["rotated_query"]
                        .to(arguments.gpu)
                        .view(1, query_heads, head_channels)
                    )
                    expected = (
                        snapshot["attention_values"].to(arguments.gpu).view_as(query)
                    )
                    comparisons = {}
                    for name, metadata in (
                        ("isolated", None),
                        ("recorded_schedule", matched_metadata),
                    ):
                        actual = flash_attn_with_kvcache(
                            query,
                            keys[layer_index, :token_count],
                            values[layer_index, :token_count],
                            page_table=page_table,
                            cache_seqlens=lengths,
                            cu_seqlens_q=cumulative_query_tokens,
                            max_seqlen_q=1,
                            softmax_scale=head_channels**-0.5,
                            causal=True,
                            scheduler_metadata=metadata,
                        )
                        comparisons[name] = {
                            "exact": torch.equal(actual, expected),
                            "different_values": int((actual != expected).sum()),
                            "maximum_absolute_error": float(
                                (actual.float() - expected.float()).abs().max()
                            ),
                        }
                    report = {
                        "frame": frame_index,
                        "layer": layer_index,
                        "batch_size": batch_size,
                        "selected_tokens": token_count,
                        "batch_sequence_lengths": batch.callback_sequence_lengths,
                        "graph_padding_requests": padding_requests,
                        "isolated_splits": int(single_metadata[0]),
                        "recorded_splits": selected_splits,
                        "comparisons": comparisons,
                    }
                    reports.append(report)
                    output.write(json.dumps(report) + "\n")
                output.flush()
        summaries.append(
            {
                "trace_directory": str(directory),
                "decode_layer_calls": len(reports),
                "exact_calls": {
                    name: sum(row["comparisons"][name]["exact"] for row in reports)
                    for name in ("isolated", "recorded_schedule")
                },
                "scope": "Held actual per-layer query and complete selected-request KV only. Peer lengths drive FA3 scheduling, not fabricated peer queries or caches. This does not generate speech or prove recurrent equivalence.",
            }
        )
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(summaries, indent=2)
        )
        print(json.dumps(summaries[-1]), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
