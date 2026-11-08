"""Measure feedback rounding over every saved code frame using checkpoint tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from safetensors import safe_open

from sglang_omni.models.qwen3_omni.corpus_diagnostics import RequestTrace


class CheckpointIndex(BaseModel):
    weight_map: dict[str, str]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    requests = sorted(
        [
            RequestTrace.model_validate_json(path.read_text())
            for path in (arguments.capture_directory / "requests").glob("*.json")
        ],
        key=lambda request: request.ordinal,
    )
    assert requests and all(request.main_codes_match for request in requests)
    index_path = hf_hub_download(
        arguments.model_path,
        "model.safetensors.index.json",
        revision=arguments.model_revision,
        local_files_only=True,
    )
    index = CheckpointIndex.model_validate_json(Path(index_path).read_text())
    group_count = len(requests[0].codes[0])
    names = ["talker.model.codec_embedding.weight"] + [
        f"talker.code_predictor.model.codec_embedding.{group}.weight"
        for group in range(group_count - 1)
    ]
    tables = []
    for name in names:
        shard_path = hf_hub_download(
            arguments.model_path,
            index.weight_map[name],
            revision=arguments.model_revision,
            local_files_only=True,
        )
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            tables.append(shard.get_tensor(name))
    assert all(table.dtype == torch.bfloat16 for table in tables)
    codes = torch.tensor(
        [frame for request in requests for frame in request.codes], dtype=torch.long
    )
    sequential = torch.zeros(codes.shape[0], tables[0].shape[1], dtype=torch.bfloat16)
    accurate = sequential.double()
    relative_errors = []
    absolute_errors = []
    changed_coordinates = []
    for group, table in enumerate(tables):
        embedding = table[codes[:, group]]
        sequential.add_(embedding)
        accurate.add_(embedding.double())
        rounded = accurate.bfloat16()
        difference = sequential.double() - rounded.double()
        relative_errors.append(
            difference.norm(dim=-1)
            / rounded.double().norm(dim=-1).clamp_min(torch.finfo(torch.float64).tiny)
        )
        absolute_errors.append(difference.abs().amax(-1))
        changed_coordinates.append((sequential != rounded).sum(-1))
    relative_errors = torch.stack(relative_errors)
    absolute_errors = torch.stack(absolute_errors)
    changed_coordinates = torch.stack(changed_coordinates)
    torch.save(
        {
            "relative_l2": relative_errors,
            "max_absolute": absolute_errors,
            "changed_coordinates": changed_coordinates,
        },
        arguments.output_directory / "feedback.pt",
    )
    request_reports = []
    start = 0
    for request in requests:
        frame_count = len(request.codes)
        assert frame_count == len(request.steps)
        errors = relative_errors[-1, start : start + frame_count]
        request_reports.append(
            {
                "request_id": request.request_id,
                "ordinal": request.ordinal,
                "start": start,
                "frames": frame_count,
                "mean_relative_l2": float(errors.mean()),
                "maximum_relative_l2": float(errors.max()),
            }
        )
        start += frame_count
    summary = {
        "requests": len(requests),
        "frames": codes.shape[0],
        "model_path": arguments.model_path,
        "model_revision": arguments.model_revision,
        "scope": "Checkpoint-table arithmetic on all actual codes, including terminal frames. Not a free-running speech-quality experiment.",
        "groups": [
            {
                "group": group,
                "mean_relative_l2": float(relative_errors[group].mean()),
                "maximum_relative_l2": float(relative_errors[group].max()),
                "maximum_absolute": float(absolute_errors[group].max()),
                "frames_with_changed_coordinates": int(
                    (changed_coordinates[group] > 0).sum()
                ),
            }
            for group in range(group_count)
        ],
        "per_request": request_reports,
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(
        json.dumps(
            {name: value for name, value in summary.items() if name != "per_request"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
