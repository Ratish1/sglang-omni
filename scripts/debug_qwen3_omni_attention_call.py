"""Replay captured native first-layer decode attention with its real peer caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sgl_kernel.flash_attn import get_scheduler_metadata
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    finished = torch.load(arguments.trace_directory / "finished.pt", weights_only=True)
    exact_counts: dict[str, int] = {}
    maximum_errors: dict[str, float] = {}
    with (arguments.output_directory / "frames.jsonl").open("x") as output:
        for frame in range(1, len(finished["output_ids"])):
            observed = torch.load(
                arguments.trace_directory / f"{frame:04d}-output.pt", weights_only=True
            )
            snapshot = torch.load(
                arguments.trace_directory / f"{frame:04d}-operators.pt",
                weights_only=True,
            )[0]
            assert bool(observed["past_exact"])
            assert not bool(observed["is_extend"])
            assert int(observed["attention_page_size_tokens"]) == 1
            assert int(observed["attention_version"]) == 3
            assert torch.equal(
                observed["attention_backend_key_pointers"],
                observed["attention_recorded_key_pointers"],
            )
            assert torch.equal(
                observed["attention_backend_value_pointers"],
                observed["attention_recorded_value_pointers"],
            )
            query_heads = int(observed["attention_query_head_count"])
            head_dimensions = int(observed["attention_query_head_dimensions"])
            query = (
                snapshot["batch_rotated_query"]
                .to(device)
                .reshape(-1, query_heads, head_dimensions)
            )
            expected = snapshot["batch_attention_values"].to(device).reshape_as(query)
            page_table = observed["attention_page_table"].to(device)
            cache_lengths = observed["attention_cache_lengths"].to(device)
            cumulative_query_lengths = observed[
                "attention_cumulative_query_lengths"
            ].to(device)
            active_requests = int(observed["attention_active_request_count"])
            selected_row = int(observed["attention_query_row"])
            assert selected_row == int(snapshot["batch_selected_row"])
            assert len(query) == len(page_table) == len(cache_lengths)
            assert torch.equal(
                cumulative_query_lengths,
                torch.arange(len(query) + 1, dtype=torch.int32, device=device),
            )
            assert 0 <= selected_row < active_requests <= len(query)
            key_shape = tuple(observed["attention_key_cache_shape"].tolist())
            value_shape = tuple(observed["attention_value_cache_shape"].tolist())
            key_cache = torch.empty_strided(
                key_shape,
                tuple(observed["attention_key_cache_strides"].tolist()),
                dtype=query.dtype,
                device=device,
            )
            value_cache = torch.empty_strided(
                value_shape,
                tuple(observed["attention_value_cache_strides"].tolist()),
                dtype=query.dtype,
                device=device,
            )
            locations = observed["attention_live_locations"].to(device)
            assert torch.equal(
                torch.unique(
                    page_table[
                        torch.arange(page_table.shape[1], device=device)[None]
                        < cache_lengths[:, None]
                    ]
                ).long(),
                locations,
            )
            key_cache[:, 0].index_copy_(
                0, locations, observed["attention_live_layer0_keys"].to(device)
            )
            value_cache[:, 0].index_copy_(
                0, locations, observed["attention_live_layer0_values"].to(device)
            )
            retained_metadata = observed[
                "attention_scheduler_metadata_after_forward"
            ].to(device)
            assert retained_metadata.numel() > 0
            fresh_metadata = get_scheduler_metadata(
                batch_size=len(query),
                max_seqlen_q=int(observed["attention_max_query_tokens"]),
                max_seqlen_k=int(observed["attention_max_key_tokens"]),
                num_heads=int(observed["attention_scheduler_query_head_count"]),
                num_heads_k=int(observed["attention_scheduler_key_head_count"]),
                headdim=int(observed["attention_scheduler_head_dimensions"]),
                cache_seqlens=cache_lengths,
                qkv_dtype=query.dtype,
                cu_seqlens_q=cumulative_query_lengths,
                page_size=int(observed["attention_page_size_tokens"]),
                causal=True,
                has_softcap=bool(observed["attention_scheduler_has_softcap"]),
                num_splits=int(observed["attention_split_count"]),
            )
            metadata_equal = torch.equal(retained_metadata, fresh_metadata)
            comparisons = {}
            for name in (
                "retained-metadata",
                "fresh-metadata",
                "automatic-metadata",
                "trimmed-page-table",
                "single-full-page-table",
                "single-trimmed-page-table",
                "retained-repeat",
            ):
                is_single = name.startswith("single-")
                selected_query = (
                    query[selected_row : selected_row + 1] if is_single else query
                )
                selected_lengths = (
                    cache_lengths[selected_row : selected_row + 1]
                    if is_single
                    else cache_lengths
                )
                selected_table = (
                    page_table[selected_row : selected_row + 1]
                    if is_single
                    else page_table
                )
                if "trimmed" in name:
                    selected_table = selected_table[:, : int(selected_lengths.max())]
                else:
                    pass
                selected_cumulative_lengths = (
                    torch.arange(2, dtype=torch.int32, device=device)
                    if is_single
                    else cumulative_query_lengths
                )
                if name in ("retained-metadata", "retained-repeat"):
                    scheduler_metadata = retained_metadata.clone()
                elif name == "fresh-metadata":
                    scheduler_metadata = fresh_metadata.clone()
                else:
                    scheduler_metadata = None
                result = flash_attn_with_kvcache(
                    q=selected_query,
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=selected_table,
                    cache_seqlens=selected_lengths,
                    cu_seqlens_q=selected_cumulative_lengths,
                    max_seqlen_q=int(observed["attention_max_query_tokens"]),
                    softmax_scale=float(observed["attention_scaling"]),
                    causal=True,
                    window_size=(-1, -1),
                    softcap=float(observed["attention_logit_cap"]),
                    num_splits=int(observed["attention_decode_split_count"]),
                    scheduler_metadata=scheduler_metadata,
                    ver=int(observed["attention_version"]),
                )
                target = expected[selected_row : selected_row + 1]
                selected_result = (
                    result if is_single else result[selected_row : selected_row + 1]
                )
                assert result.dtype == expected.dtype == torch.bfloat16
                assert torch.isfinite(result).all()
                exact = torch.equal(selected_result, target)
                error = float((selected_result.float() - target.float()).abs().max())
                exact_counts[name] = exact_counts.get(name, 0) + exact
                maximum_errors[name] = max(maximum_errors.get(name, 0.0), error)
                comparisons[name] = {
                    "selected_exact": exact,
                    "selected_changed_values": int(
                        torch.count_nonzero(selected_result != target)
                    ),
                    "selected_maximum_absolute_error": error,
                    "all_live_exact": (
                        torch.equal(
                            result[:active_requests], expected[:active_requests]
                        )
                        if not is_single
                        else exact
                    ),
                }
            output.write(
                json.dumps(
                    {
                        "frame": frame,
                        "batch_size": len(query),
                        "active_requests": active_requests,
                        "selected_row": selected_row,
                        "retained_metadata": observed[
                            "attention_scheduler_metadata_after_forward"
                        ].tolist(),
                        "fresh_metadata": fresh_metadata.cpu().tolist(),
                        "metadata_equal_before_replay": metadata_equal,
                        "comparisons": comparisons,
                    }
                )
                + "\n"
            )
            output.flush()
            print(
                json.dumps({"frame": frame, "exact_counts": exact_counts}), flush=True
            )
    report = {
        "request_id": arguments.trace_directory.name,
        "decode_frames": len(finished["output_ids"]) - 1,
        "exact_counts": exact_counts,
        "maximum_absolute_errors": maximum_errors,
        "scope": "Captured first-layer full batch queries, live peer KV contents, physical page IDs, cache shapes/strides and official scheduler metadata. Other layers are not replayed by this tool.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
else:
    pass
