"""Separate attention API layout and splitting on actual recorded decode operands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from transformers import Qwen3OmniMoeConfig


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--decode-page-table-tokens", type=int, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        local_files_only=True,
    ).talker_config.text_config
    initial = torch.load(
        arguments.trace_directory / "0000-output.pt", weights_only=True
    )
    finished = torch.load(arguments.trace_directory / "finished.pt", weights_only=True)
    keys = [tensor.to(device) for tensor in initial["keys"]]
    values = [tensor.to(device) for tensor in initial["values"]]
    counts: dict[str, int] = {}
    maximum_errors: dict[str, float] = {}
    with (arguments.output_directory / "layers.jsonl").open("x") as output:
        for frame in range(1, len(finished["output_ids"])):
            observed = torch.load(
                arguments.trace_directory / f"{frame:04d}-output.pt", weights_only=True
            )
            snapshots = torch.load(
                arguments.trace_directory / f"{frame:04d}-operators.pt",
                weights_only=True,
            )
            assert bool(observed["past_exact"])
            for layer_index, snapshot in enumerate(snapshots):
                keys[layer_index] = torch.cat(
                    (keys[layer_index], observed["keys"][layer_index][-1:].to(device))
                )
                values[layer_index] = torch.cat(
                    (
                        values[layer_index],
                        observed["values"][layer_index][-1:].to(device),
                    )
                )
                key, value = keys[layer_index], values[layer_index]
                assert len(key) == len(value) == int(observed["sequence_length"])
                assert key.dtype == value.dtype == torch.bfloat16
                query = (
                    snapshot["rotated_query"]
                    .to(device)
                    .reshape(
                        1, configuration.num_attention_heads, configuration.head_dim
                    )
                )
                expected = snapshot["attention_values"].to(device).reshape_as(query)
                recorded_batch_size = len(snapshot["batch_mlp_normalized"])
                selected_row = int(snapshot["batch_selected_row"])
                comparisons = {}
                for layout in ("dense", "paged", "shared-prefix-batch", "native-width"):
                    batch_size = (
                        recorded_batch_size if layout == "shared-prefix-batch" else 1
                    )
                    cache_lengths = torch.full(
                        (batch_size,), len(key), dtype=torch.int32, device=device
                    )
                    page_table = torch.arange(
                        len(key), dtype=torch.int32, device=device
                    ).repeat(batch_size, 1)
                    if layout == "native-width":
                        assert arguments.decode_page_table_tokens >= len(key)
                        page_table = torch.nn.functional.pad(
                            page_table,
                            (0, arguments.decode_page_table_tokens - len(key)),
                        )
                    else:
                        pass
                    cumulative_query_lengths = torch.arange(
                        batch_size + 1, dtype=torch.int32, device=device
                    )
                    for split_count in (0, 1):
                        if layout == "dense":
                            result = flash_attn_with_kvcache(
                                query.unsqueeze(0),
                                key.unsqueeze(0),
                                value.unsqueeze(0),
                                cache_seqlens=cache_lengths,
                                softmax_scale=configuration.head_dim**-0.5,
                                causal=True,
                                num_splits=split_count,
                                ver=3,
                            ).reshape_as(query)
                        else:
                            results = flash_attn_with_kvcache(
                                query.repeat(batch_size, 1, 1),
                                key.unsqueeze(1),
                                value.unsqueeze(1),
                                page_table=page_table,
                                cache_seqlens=cache_lengths,
                                cu_seqlens_q=cumulative_query_lengths,
                                max_seqlen_q=1,
                                softmax_scale=configuration.head_dim**-0.5,
                                causal=True,
                                num_splits=split_count,
                                ver=3,
                            )
                            row = selected_row if layout == "shared-prefix-batch" else 0
                            result = results[row : row + 1]
                        assert result.shape == expected.shape
                        assert torch.isfinite(result).all()
                        name = f"{layout}-splits-{split_count}"
                        exact = torch.equal(expected, result)
                        error = float((expected.float() - result.float()).abs().max())
                        counts[name] = counts.get(name, 0) + exact
                        maximum_errors[name] = max(maximum_errors.get(name, 0.0), error)
                        comparisons[name] = {
                            "exact": exact,
                            "changed_values": int(
                                torch.count_nonzero(expected != result)
                            ),
                            "maximum_absolute_error": error,
                        }
                output.write(
                    json.dumps(
                        {
                            "frame": frame,
                            "layer": layer_index,
                            "key_tokens": len(key),
                            "recorded_batch_size": recorded_batch_size,
                            "comparisons": comparisons,
                        }
                    )
                    + "\n"
                )
            print(json.dumps({"frame": frame, "exact_counts": counts}), flush=True)
    report = {
        "request_id": arguments.trace_directory.name,
        "decode_frames": len(finished["output_ids"]) - 1,
        "layer_steps": (len(finished["output_ids"]) - 1) * len(keys),
        "exact_counts": counts,
        "maximum_absolute_errors": maximum_errors,
        "decode_page_table_tokens": arguments.decode_page_table_tokens,
        "scope": "Same recorded query and complete native KV prefix. Dense versus paged API, automatic versus one split, and duplicated read-only shared-prefix requests at the recorded padded batch size. Not a replay of uncaptured peer requests or scheduler metadata, and not a production policy change.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
else:
    pass
