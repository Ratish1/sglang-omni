"""Compare actual prefill and decode attention with complete causal histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig

from scripts.debug_qwen3_omni_expert_corpus import compare_outputs
from sglang_omni.models.qwen3_omni.corpus_diagnostics import StepMetadata


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config.text_config
    assert config.sliding_window is None
    requested_ids = set(arguments.request_id)
    histories: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    checked_rows = {
        request_id: {"EXTEND": 0, "DECODE": 0} for request_id in requested_ids
    }
    maximum_relative = [0.0] * config.num_hidden_layers
    maximum_absolute = [0.0] * config.num_hidden_layers
    history_checks = [
        json.loads(line)
        for line in (arguments.capture_directory / "cache-history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert history_checks
    assert all(
        check["previous_keys_exact"]
        and check["previous_values_exact"]
        and check["new_key_exact"]
        and check["new_value_exact"]
        for check in history_checks
    )
    verified_history = {
        (check["request_id"], check["frame_index"], check["layer_index"])
        for check in history_checks
    }
    with (arguments.output_directory / "rows.jsonl").open("w") as reports:
        for line in (
            (arguments.capture_directory / "steps.jsonl").read_text().splitlines()
        ):
            metadata = StepMetadata.model_validate_json(line)
            saved = torch.load(
                arguments.capture_directory / f"experts-{metadata.step:06d}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            snapshots = saved["snapshots"]
            offset = 0
            for request_row, request_id in enumerate(metadata.request_ids):
                query_rows = saved["extend_lengths"][request_row]
                selected_rows = slice(offset, offset + query_rows)
                offset += query_rows
                if request_id not in requested_ids:
                    continue
                else:
                    pass
                sequence_length = metadata.sequence_lengths[request_row]
                if metadata.phase == "EXTEND":
                    prefix = torch.load(
                        arguments.capture_directory
                        / f"prefix-{request_id}-{metadata.step:06d}.pt",
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )
                    keys, values = prefix["keys"], prefix["values"]
                    prefix_rows = saved["prefix_lengths"][request_row]
                    assert prefix_rows + query_rows == sequence_length
                else:
                    assert metadata.phase == "DECODE" and query_rows == 1
                    keys, values = histories[request_id]
                    prefix_rows = sequence_length - 1
                    assert keys[0].shape[0] == prefix_rows
                    keys = [
                        torch.cat(
                            [
                                keys[layer],
                                snapshots[f"backbone.{layer}.key"][
                                    selected_rows
                                ].reshape(
                                    query_rows,
                                    config.num_key_value_heads,
                                    config.head_dim,
                                ),
                            ]
                        )
                        for layer in range(config.num_hidden_layers)
                    ]
                    values = [
                        torch.cat(
                            [
                                values[layer],
                                snapshots[f"backbone.{layer}.value"][
                                    selected_rows
                                ].reshape(
                                    query_rows,
                                    config.num_key_value_heads,
                                    config.head_dim,
                                ),
                            ]
                        )
                        for layer in range(config.num_hidden_layers)
                    ]
                    assert all(
                        (request_id, metadata.frame_indices[request_row], layer)
                        in verified_history
                        for layer in range(config.num_hidden_layers)
                    )
                histories[request_id] = (keys, values)
                checked_rows[request_id][metadata.phase] += query_rows
                for layer in range(config.num_hidden_layers):
                    observed = f"backbone.{layer}."
                    assert (
                        keys[layer].shape[0]
                        == values[layer].shape[0]
                        == sequence_length
                    )
                    for name, cached in (
                        ("key", keys[layer]),
                        ("value", values[layer]),
                    ):
                        assert torch.equal(
                            snapshots[observed + name][selected_rows],
                            cached[prefix_rows:].reshape(query_rows, -1),
                        )
                    queries = (
                        snapshots[observed + "query"][selected_rows]
                        .cuda()
                        .double()
                        .reshape(
                            query_rows, config.num_attention_heads, config.head_dim
                        )
                        .transpose(0, 1)
                    )
                    repetitions = (
                        config.num_attention_heads // config.num_key_value_heads
                    )
                    expanded_keys = (
                        keys[layer]
                        .cuda()
                        .double()
                        .transpose(0, 1)
                        .repeat_interleave(repetitions, dim=0)
                    )
                    expanded_values = (
                        values[layer]
                        .cuda()
                        .double()
                        .transpose(0, 1)
                        .repeat_interleave(repetitions, dim=0)
                    )
                    scores = (
                        queries @ expanded_keys.transpose(-1, -2)
                    ) * config.head_dim**-0.5
                    visible = torch.arange(sequence_length, device="cuda")[None] <= (
                        prefix_rows + torch.arange(query_rows, device="cuda")[:, None]
                    )
                    probabilities = scores.masked_fill(~visible, float("-inf")).softmax(
                        -1
                    )
                    expected = (
                        (probabilities @ expanded_values)
                        .transpose(0, 1)
                        .reshape(query_rows, -1)
                    )
                    actual = snapshots[observed + "attention_projection.input"][
                        selected_rows
                    ].cuda()
                    assert actual.isfinite().all()
                    metrics = compare_outputs(actual, expected)
                    assert metrics["finite"].all()
                    maximum_relative[layer] = max(
                        maximum_relative[layer], float(metrics["relative_l2"].max())
                    )
                    maximum_absolute[layer] = max(
                        maximum_absolute[layer], float(metrics["max_absolute"].max())
                    )
                    reports.write(
                        json.dumps(
                            {
                                "request_id": request_id,
                                "step": metadata.step,
                                "phase": metadata.phase,
                                "frame_index": metadata.frame_indices[request_row],
                                "layer": layer,
                                "query_rows": query_rows,
                                "prefix_rows": prefix_rows,
                                "relative_l2": metrics["relative_l2"].cpu().tolist(),
                                "maximum_absolute": metrics["max_absolute"]
                                .cpu()
                                .tolist(),
                            }
                        )
                        + "\n"
                    )
                reports.flush()
            assert offset == saved["real_rows"]
    assert set(histories) == requested_ids
    assert all(rows["EXTEND"] and rows["DECODE"] for rows in checked_rows.values())
    summary = {
        "model_revision": arguments.model_revision,
        "checked_rows_per_layer": checked_rows,
        "maximum_relative_l2_by_layer": maximum_relative,
        "maximum_absolute_by_layer": maximum_absolute,
        "scope": "Every selected actual prefill query and decode frame, complete causal K/V histories, all layers, FP64 same-operand GQA. Decode histories require independent input-to-store and full-prefix persistence checks. Numerical errors are observations, not an arbitrary tolerance or a speech-quality verdict.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
