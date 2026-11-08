"""Preserve complete selected-row traces and their actual padded predictor inputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from sglang_omni.models.qwen3_omni.corpus_diagnostics import RequestTrace, StepMetadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    manifests = {
        metadata.step: metadata
        for metadata in [
            StepMetadata.model_validate_json(line)
            for line in (arguments.capture_directory / "steps.jsonl")
            .read_text()
            .splitlines()
        ]
    }
    summaries = []
    for request_id in arguments.request_id:
        request_path = arguments.capture_directory / "requests" / f"{request_id}.json"
        request = RequestTrace.model_validate_json(request_path.read_text())
        directory = arguments.output_directory / request_id
        directory.mkdir()
        shutil.copyfile(request_path, directory / "request.json")
        with (directory / "steps.jsonl").open("w") as output:
            for frame, step in enumerate(request.steps):
                metadata = manifests[step]
                row = metadata.request_ids.index(request_id)
                assert metadata.frame_indices[row] == frame
                saved = torch.load(
                    arguments.capture_directory / f"step-{step:06d}.pt",
                    weights_only=True,
                    map_location="cpu",
                    mmap=True,
                )
                assert torch.equal(
                    saved["codes"][row],
                    torch.tensor(request.codes[frame], dtype=saved["codes"].dtype),
                )
                # note (ratish): clones prevent serialization of the original batch storage.
                snapshots = {
                    name: tensor[row : row + 1].clone()
                    for name, tensor in saved["snapshots"].items()
                }
                dense = {}
                for name, tensor in saved["dense"].items():
                    if name == "seen":
                        dense[name] = tensor.clone()
                    elif name.startswith("opening_"):
                        dense[name] = tensor[:, row : row + 1].clone()
                    else:
                        dense[name] = tensor[:, :, row : row + 1].clone()
                if saved["mrope_positions"].numel() == 0:
                    mrope_positions = saved["mrope_positions"].clone()
                else:
                    mrope_positions = saved["mrope_positions"][:, row : row + 1].clone()
                torch.save(
                    {
                        "snapshots": snapshots,
                        "dense": dense,
                        "codes": saved["codes"][row : row + 1].clone(),
                        "feedback": saved["feedback"][row : row + 1].clone(),
                        "predictor_hidden_all_rows": saved["predictor_hidden"].clone(),
                        "layer0_codes_all_rows": saved["layer0_codes"].clone(),
                        "code_int32": saved["code_int32"].clone(),
                        "predictor_counters": saved["predictor_counters"].clone(),
                        "positions": saved["positions"][row : row + 1].clone(),
                        "mrope_positions": mrope_positions,
                    },
                    directory / f"frame-{frame:06d}.pt",
                )
                report = {
                    "frame": frame,
                    "source_row": row,
                    "original_step": metadata.model_dump(),
                }
                output.write(json.dumps(report) + "\n")
                prefix = (
                    arguments.capture_directory / f"prefix-{request_id}-{step:06d}.pt"
                )
                if prefix.exists():
                    shutil.copyfile(prefix, directory / prefix.name)
                else:
                    pass
        summaries.append(
            {
                "request_id": request_id,
                "ordinal": request.ordinal,
                "frames": len(request.codes),
            }
        )
    report = {
        "requests": summaries,
        "source": str(arguments.capture_directory),
        "scope": "All recorded operation boundaries for each selected row, all actual padded predictor inputs, original metadata, complete code histories and actual prefill KV. Other rows' intermediate operation tensors are not included. This is not the original full-batch recorder format.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
