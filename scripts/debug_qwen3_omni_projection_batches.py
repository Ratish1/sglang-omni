"""Check native dense projections without changing their recorded batch geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.qwen3_omni_checkpoint_weights import CheckpointIndex


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir()
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    checkpoint_directory = Path(
        hf_hub_download(
            arguments.model_path,
            "model.safetensors.index.json",
            revision=arguments.model_revision,
        )
    ).parent
    index = CheckpointIndex.model_validate_json(
        (checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    projection_weights: dict[str, torch.Tensor] = {}
    for name, shard in index.weight_map.items():
        if (
            name.startswith("talker.model.layers.")
            and any(
                name.endswith(f".self_attn.{projection}.weight")
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
            or name == "talker.codec_head.weight"
        ):
            with safe_open(
                checkpoint_directory / shard, framework="pt", device="cpu"
            ) as checkpoint:
                weight = checkpoint.get_tensor(name)
            assert weight.dtype == torch.bfloat16 and weight.ndim == 2
            projection_weights[name] = weight.to(device)
        else:
            pass
    layer_count = (len(projection_weights) - 1) // 4
    assert len(projection_weights) == 4 * layer_count + 1 and layer_count > 0
    merged_weights = [
        torch.cat(
            [
                projection_weights[
                    f"talker.model.layers.{layer_index}.self_attn.{projection}.weight"
                ]
                for projection in ("q_proj", "k_proj", "v_proj")
            ]
        )
        for layer_index in range(layer_count)
    ]
    native_parameters = json.loads(
        (arguments.capture_directory / "native-talker-parameters.json").read_text()
    )["parameters"]
    checked_weights = {
        "codec_head.weight": projection_weights["talker.codec_head.weight"]
    }
    for layer_index, weight in enumerate(merged_weights):
        checked_weights[f"model.layers.{layer_index}.self_attn.qkv_proj.weight"] = (
            weight
        )
        checked_weights[f"model.layers.{layer_index}.self_attn.o_proj.weight"] = (
            projection_weights[
                f"talker.model.layers.{layer_index}.self_attn.o_proj.weight"
            ]
        )
    for name, weight in checked_weights.items():
        actual = native_parameters[name]
        assert actual["shape"] == list(weight.shape)
        assert actual["dtype"] == str(weight.dtype)
        assert (
            actual["sha256"]
            == hashlib.sha256(
                weight.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
        ), name
    counts = {
        "comparisons": 0,
        "full_batch_mismatches": 0,
        "single_row_mismatches": 0,
        "head_full_mismatches": 0,
        "head_single_mismatches": 0,
        "frames": 0,
    }
    with (arguments.output_directory / "projections.jsonl").open("w") as output:
        for request_id in arguments.request_id:
            directory = arguments.capture_directory / request_id
            finished = torch.load(directory / "finished.pt", weights_only=True)
            filenames = sorted(directory.glob("*-operators.pt"))
            assert len(filenames) == len(finished["output_ids"])
            for frame_index, filename in enumerate(filenames):
                counts["frames"] += 1
                assert filename.name == f"{frame_index:04d}-operators.pt"
                snapshots = torch.load(filename, weights_only=True)
                assert len(snapshots) == layer_count
                for layer_index, snapshot in enumerate(snapshots):
                    selected_row = int(snapshot["batch_selected_row"])
                    for operation, input_name, expected_name, weight in (
                        (
                            "qkv",
                            "attention_normalized",
                            "qkv",
                            merged_weights[layer_index],
                        ),
                        (
                            "output",
                            "attention_values",
                            "attention_output",
                            projection_weights[
                                f"talker.model.layers.{layer_index}.self_attn.o_proj.weight"
                            ],
                        ),
                    ):
                        operands = snapshot[f"batch_{input_name}"]
                        layout = snapshot[f"layout_{input_name}"].tolist()
                        assert operands.ndim == 2 and operands.dtype == torch.bfloat16
                        assert layout == [*operands.shape, *operands.stride(), 0]
                        assert torch.equal(operands[selected_row], snapshot[input_name])
                        operands = operands.to(device)
                        assert operands.stride() == tuple(layout[2:4])
                        assert operands.data_ptr() % 16 == layout[-1]
                        expected = snapshot[expected_name].to(device)
                        full_batch = torch.nn.functional.linear(operands, weight)[
                            selected_row
                        ]
                        single_row = torch.nn.functional.linear(
                            operands[selected_row : selected_row + 1], weight
                        )[0]
                        complete_difference = compare_tensors(full_batch, expected)
                        selected_difference = compare_tensors(single_row, expected)
                        counts["comparisons"] += 1
                        counts["full_batch_mismatches"] += not complete_difference.exact
                        counts["single_row_mismatches"] += not selected_difference.exact
                        output.write(
                            json.dumps(
                                {
                                    "request_id": request_id,
                                    "frame_index": frame_index,
                                    "layer_index": layer_index,
                                    "operation": operation,
                                    "input_shape": list(operands.shape),
                                    "selected_row": selected_row,
                                    "full_batch": asdict(complete_difference),
                                    "single_row": asdict(selected_difference),
                                }
                            )
                            + "\n"
                        )
                    if layer_index == layer_count - 1:
                        batch = json.loads(
                            (directory / f"{frame_index:04d}-batch.json").read_text()
                        )
                        head_row = batch["active_row"]
                        head_inputs = snapshot["batch_codec_head_input"].to(device)
                        head_outputs = snapshot["batch_codec_head_output"].to(device)
                        assert head_inputs.dtype == head_outputs.dtype == torch.bfloat16
                        assert snapshot["layout_codec_head_input"].tolist() == [
                            *head_inputs.shape,
                            *head_inputs.stride(),
                            head_inputs.data_ptr() % 16,
                        ]
                        observed = torch.load(
                            directory / f"{frame_index:04d}-output.pt",
                            weights_only=True,
                        )
                        assert torch.equal(
                            head_inputs[head_row].cpu(), observed["hidden_states"]
                        )
                        weight = projection_weights["talker.codec_head.weight"]
                        for operation, inputs, expected in (
                            ("head_full", head_inputs, head_outputs),
                            (
                                "head_single",
                                head_inputs[head_row : head_row + 1],
                                head_outputs[head_row : head_row + 1],
                            ),
                        ):
                            difference = compare_tensors(
                                torch.nn.functional.linear(inputs, weight), expected
                            )
                            counts[f"{operation}_mismatches"] += not difference.exact
                            output.write(
                                json.dumps(
                                    {
                                        "request_id": request_id,
                                        "frame_index": frame_index,
                                        "operation": operation,
                                        "input_shape": list(inputs.shape),
                                        "head_row": head_row,
                                        "difference": asdict(difference),
                                    }
                                )
                                + "\n"
                            )
                    else:
                        pass
                output.flush()
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Teacher-forced dense operator equality on every recorded frame and layer. Full and selected-row geometry are compared separately. No speech-quality or free-running causal claim.",
                "model_revision": arguments.model_revision,
                "request_ids": arguments.request_id,
                "native_checkpoint_projection_weights_exact": len(checked_weights),
                **counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
