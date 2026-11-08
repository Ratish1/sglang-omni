"""Locate old-versus-new head decisions on captured native predictor states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.utils.hub import cached_file

from scripts.debug_qwen3_omni_head_precision import project_head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--predictor-directory", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    assert not arguments.output_file.exists()
    torch.set_float32_matmul_precision("highest")
    checkpoint_index = json.loads(
        Path(
            cached_file(arguments.model_path, "model.safetensors.index.json")
        ).read_text()
    )
    predictor_frames = json.loads(
        (arguments.predictor_directory / "summary.json").read_text()
    )
    head_weights: dict[int, torch.Tensor] = {}
    decision_reports = []
    with torch.inference_mode():
        for frame_record in predictor_frames:
            predictor_capture = torch.load(
                arguments.predictor_directory
                / f"{frame_record['request_id']}.{frame_record['frame']}.pt",
                map_location="cuda",
                weights_only=True,
            )
            predictor_logits = predictor_capture["native_logits"]
            predictor_hidden_states = predictor_capture[
                "native_predictor_hidden_states"
            ]
            assert predictor_logits.dtype == torch.float32
            for group_index in range(predictor_logits.shape[1]):
                if group_index not in head_weights:
                    tensor_name = f"talker.code_predictor.lm_head.{group_index}.weight"
                    checkpoint_shard = checkpoint_index["weight_map"][tensor_name]
                    with safe_open(
                        cached_file(arguments.model_path, checkpoint_shard),
                        framework="pt",
                        device="cuda:0",
                    ) as checkpoint:
                        head_weights[group_index] = checkpoint.get_tensor(tensor_name)
                else:
                    pass
                head_weight = head_weights[group_index]
                head_hidden_row = predictor_hidden_states[:, group_index]
                assert head_hidden_row.dtype == head_weight.dtype == torch.bfloat16
                rounded_logits = torch.mm(head_hidden_row, head_weight.T)
                projected_logits, head_comparison = project_head(
                    head_hidden_row, head_weight, rounded_logits
                )
                assert torch.equal(projected_logits, predictor_logits[:, group_index])
                assert head_comparison["float32_output_rounds_to_recorded"]
                decision_reports.append(
                    {
                        "request_id": frame_record["request_id"],
                        "frame": frame_record["frame"],
                        "group": group_index + 1,
                        "captured_float32_logits_exact": True,
                        "bfloat16_code": int(rounded_logits.argmax()),
                        "float32_code": int(projected_logits.argmax()),
                        **head_comparison,
                    }
                )
    arguments.output_file.write_text(json.dumps(decision_reports, indent=2))
    print(
        json.dumps(
            {
                "decisions": len(decision_reports),
                "changed": [
                    record
                    for record in decision_reports
                    if record["bfloat16_code"] != record["float32_code"]
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
