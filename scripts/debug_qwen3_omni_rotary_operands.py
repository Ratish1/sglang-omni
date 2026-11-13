"""Replay native Talker rotary arithmetic over complete captured lifetimes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.qwen3_omni_talker_rotary import NativeRotaryOperation


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--trace-directory", type=Path, action="append", required=True)
    parser.add_argument("--native-buffer-directory", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config.text_config
    operation = NativeRotaryOperation(
        configuration, arguments.native_buffer_directory, device
    )
    cache, axis_map = operation.cache, operation.axis_map
    query_dimensions = configuration.num_attention_heads * configuration.head_dim
    key_dimensions = configuration.num_key_value_heads * configuration.head_dim
    records = []
    with (arguments.output_directory / "layers.jsonl").open("w") as output:
        for directory in arguments.trace_directory:
            finished = torch.load(directory / "finished.pt", weights_only=True)
            files = sorted(directory.glob("*-operators.pt"))
            assert [filename.name for filename in files] == [
                f"{frame:04d}-operators.pt"
                for frame in range(len(finished["output_ids"]))
            ]
            layer_steps = 0
            token_rows = 0
            for frame, filename in enumerate(files):
                snapshots = torch.load(filename, weights_only=True)
                assert len(snapshots) == configuration.num_hidden_layers
                for layer_index, snapshot in enumerate(snapshots):
                    field_prefix = "prefix_" if "prefix_token_rows" in snapshot else ""
                    positions = (
                        snapshot[f"{field_prefix}positions"]
                        .reshape(-1, 3)
                        .T.contiguous()
                        .to(device)
                    )
                    query = (
                        snapshot[f"{field_prefix}normalized_query"]
                        .reshape(-1, query_dimensions)
                        .to(device)
                    )
                    key = (
                        snapshot[f"{field_prefix}normalized_key"]
                        .reshape(-1, key_dimensions)
                        .to(device)
                    )
                    expected_query = (
                        snapshot[f"{field_prefix}rotated_query"]
                        .reshape_as(query)
                        .to(device)
                    )
                    expected_key = (
                        snapshot[f"{field_prefix}rotated_key"]
                        .reshape_as(key)
                        .to(device)
                    )
                    assert positions.shape[1] == query.shape[0] == key.shape[0]
                    native_query, native_key = operation.apply(query, key, positions)
                    cosine, sine = cache[positions].chunk(2, dim=-1)
                    lane_axes = axis_map.reshape(1, -1).expand(len(query), -1)
                    cosine = (
                        cosine.permute(1, 2, 0)
                        .gather(2, lane_axes.unsqueeze(-1))
                        .squeeze(-1)
                    )
                    sine = (
                        sine.permute(1, 2, 0)
                        .gather(2, lane_axes.unsqueeze(-1))
                        .squeeze(-1)
                    )
                    differences = {}
                    for name, original, observed, replay in (
                        ("query", query, expected_query, native_query),
                        ("key", key, expected_key, native_key),
                    ):
                        assert torch.equal(observed, replay), (
                            directory.name,
                            frame,
                            layer_index,
                            name,
                        )
                        heads = original.reshape(len(query), -1, configuration.head_dim)
                        first, second = heads.chunk(2, dim=-1)
                        cosine_heads, sine_heads = cosine.unsqueeze(1), sine.unsqueeze(
                            1
                        )
                        rounded_products = torch.cat(
                            (
                                first * cosine_heads - second * sine_heads,
                                second * cosine_heads + first * sine_heads,
                            ),
                            dim=-1,
                        )
                        exact_formula = torch.cat(
                            (
                                first.double() * cosine_heads.double()
                                - second.double() * sine_heads.double(),
                                second.double() * cosine_heads.double()
                                + first.double() * sine_heads.double(),
                            ),
                            dim=-1,
                        ).to(original.dtype)
                        differences[name] = {
                            "native_replay": asdict(compare_tensors(observed, replay)),
                            "rounded_products": asdict(
                                compare_tensors(
                                    observed, rounded_products.reshape_as(original)
                                )
                            ),
                            "float64_formula": asdict(
                                compare_tensors(
                                    observed, exact_formula.reshape_as(original)
                                )
                            ),
                        }
                    layer_steps += 1
                    token_rows += len(query)
                    output.write(
                        json.dumps(
                            {
                                "request_id": directory.name,
                                "frame": frame,
                                "layer": layer_index,
                                "token_rows": len(query),
                                "positions_axes_equal": bool(
                                    torch.equal(positions[0], positions[1])
                                    and torch.equal(positions[0], positions[2])
                                ),
                                "differences": differences,
                            }
                        )
                        + "\n"
                    )
                print(
                    json.dumps({"request_id": directory.name, "frame": frame}),
                    flush=True,
                )
            records.append(
                {
                    "request_id": directory.name,
                    "frames": len(files),
                    "layer_steps": layer_steps,
                    "token_rows": token_rows,
                }
            )
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(
                    {
                        "records": records,
                        "sections": operation.sections,
                        "interleaved": operation.interleaved,
                        "runtime": operation.runtime.model_dump(by_alias=True),
                        "scope": "Exact native rotary replay and independent formulas on captured operands, not a free-running quality repair.",
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
else:
    pass
