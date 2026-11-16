"""Compare isolated FA3 decode with scheduling from the recorded peer batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from scripts.qwen3_omni_talker_attention import DecodeBatch, recorded_decode_schedule


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
                schedule = recorded_decode_schedule(
                    batch=batch,
                    batch_size=batch_size,
                    token_count=token_count,
                    query_heads=query_heads,
                    key_heads=key_heads,
                    head_channels=head_channels,
                    page_table_tokens=arguments.page_table_tokens,
                    device=keys.device,
                )
                lengths = torch.tensor(
                    [token_count], dtype=torch.int32, device=arguments.gpu
                )
                cumulative_query_tokens = torch.arange(
                    2, dtype=torch.int32, device=arguments.gpu
                )
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
                        ("recorded_schedule", schedule.metadata),
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
                        "graph_padding_requests": schedule.padding_requests,
                        "isolated_splits": schedule.isolated_splits,
                        "recorded_splits": schedule.recorded_splits,
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
