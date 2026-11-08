"""Compare complete recorded lifetimes, retaining all divergent and terminal frames."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from sglang_omni.models.qwen3_omni.corpus_diagnostics import RequestTrace, StepMetadata


@dataclass(kw_only=True)
class TensorComparison:
    before_shape: list[int]
    after_shape: list[int]
    exact: bool
    relative_l2: float | None
    maximum_absolute: float | None


def compare_tensors(before: torch.Tensor, after: torch.Tensor) -> TensorComparison:
    if before.shape != after.shape:
        return TensorComparison(
            before_shape=list(before.shape),
            after_shape=list(after.shape),
            exact=False,
            relative_l2=None,
            maximum_absolute=None,
        )
    else:
        difference = after.double() - before.double()
        return TensorComparison(
            before_shape=list(before.shape),
            after_shape=list(after.shape),
            exact=torch.equal(before, after),
            relative_l2=float(
                difference.norm()
                / before.double().norm().clamp_min(torch.finfo(torch.float64).tiny)
            ),
            maximum_absolute=float(difference.abs().max()),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-directory", type=Path, required=True)
    parser.add_argument("--after-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    directories = (arguments.before_directory, arguments.after_directory)
    manifests = [
        {
            metadata.step: metadata
            for metadata in [
                StepMetadata.model_validate_json(line)
                for line in (directory / "steps.jsonl").read_text().splitlines()
            ]
        }
        for directory in directories
    ]
    summaries = []
    for request_id in arguments.request_id:
        before, after = [
            RequestTrace.model_validate_json(
                (directory / "requests" / f"{request_id}.json").read_text()
            )
            for directory in directories
        ]
        prefix_comparisons = []
        prefixes = [
            torch.load(
                directory / f"prefix-{request_id}-{request.steps[0]:06d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            for directory, request in zip(directories, (before, after), strict=True)
        ]
        for name in ("keys", "values"):
            for layer, (before_tensor, after_tensor) in enumerate(
                zip(prefixes[0][name], prefixes[1][name], strict=True)
            ):
                prefix_comparisons.append(
                    {
                        "tensor": name,
                        "layer": layer,
                        "comparison": asdict(
                            compare_tensors(before_tensor, after_tensor)
                        ),
                    }
                )
        frame_reports = []
        for frame in range(max(len(before.codes), len(after.codes))):
            if frame >= min(len(before.codes), len(after.codes)):
                frame_reports.append(
                    {
                        "frame": frame,
                        "before_codes": (
                            before.codes[frame] if frame < len(before.codes) else None
                        ),
                        "after_codes": (
                            after.codes[frame] if frame < len(after.codes) else None
                        ),
                    }
                )
                continue
            else:
                pass
            steps = (before.steps[frame], after.steps[frame])
            metadata = [
                manifest[step] for manifest, step in zip(manifests, steps, strict=True)
            ]
            rows = [record.request_ids.index(request_id) for record in metadata]
            saved = [
                torch.load(
                    directory / f"step-{step:06d}.pt",
                    weights_only=True,
                    map_location="cpu",
                    mmap=True,
                )
                for directory, step in zip(directories, steps, strict=True)
            ]
            comparisons = {}
            for name in ("backbone_input", "codec.input", "codec.output"):
                comparisons[name] = asdict(
                    compare_tensors(
                        saved[0]["snapshots"][name][rows[0]],
                        saved[1]["snapshots"][name][rows[1]],
                    )
                )
            for name in ("predictor_hidden", "feedback"):
                comparisons[name] = asdict(
                    compare_tensors(saved[0][name][rows[0]], saved[1][name][rows[1]])
                )
            changed_expert_layers = []
            router_comparisons = []
            for layer in range(len(prefixes[0]["keys"])):
                name = f"backbone.{layer}.expert_ids"
                if not torch.equal(
                    saved[0]["snapshots"][name][rows[0]].sort().values,
                    saved[1]["snapshots"][name][rows[1]].sort().values,
                ):
                    changed_expert_layers.append(layer)
                else:
                    pass
                name = f"backbone.{layer}.router.output"
                router_comparisons.append(
                    asdict(
                        compare_tensors(
                            saved[0]["snapshots"][name][rows[0]],
                            saved[1]["snapshots"][name][rows[1]],
                        )
                    )
                )
            head_comparisons = [
                asdict(
                    compare_tensors(
                        saved[0]["snapshots"][f"predictor.{group}.output"][rows[0]],
                        saved[1]["snapshots"][f"predictor.{group}.output"][rows[1]],
                    )
                )
                for group in range(len(before.codes[frame]) - 1)
            ]
            frame_reports.append(
                {
                    "frame": frame,
                    "before_step": steps[0],
                    "after_step": steps[1],
                    "before_codes": before.codes[frame],
                    "after_codes": after.codes[frame],
                    "preceding_codebooks_identical": before.codes[:frame]
                    == after.codes[:frame],
                    "changed_codebook_groups": [
                        group
                        for group, (before_code, after_code) in enumerate(
                            zip(before.codes[frame], after.codes[frame], strict=True)
                        )
                        if before_code != after_code
                    ],
                    "changed_expert_layers": changed_expert_layers,
                    "before_live_rows": metadata[0].live_rows,
                    "after_live_rows": metadata[1].live_rows,
                    "before_predictor_batch": metadata[0].predictor_batch,
                    "after_predictor_batch": metadata[1].predictor_batch,
                    "comparisons": comparisons,
                    "router_comparisons": router_comparisons,
                    "head_comparisons": head_comparisons,
                }
            )
        report = {
            "request_id": request_id,
            "before_ordinal": before.ordinal,
            "after_ordinal": after.ordinal,
            "before_frame_count": len(before.codes),
            "after_frame_count": len(after.codes),
            "before_termination": before.finish_reason,
            "after_termination": after.finish_reason,
            "prefix_comparisons": prefix_comparisons,
            "frames": frame_reports,
            "scope": "Observed whole-lifetime differences, not a claim that one early difference caused every later change.",
        }
        (arguments.output_directory / f"{request_id}.json").write_text(
            json.dumps(report, indent=2)
        )
        summaries.append(
            {
                name: value
                for name, value in report.items()
                if name not in ("prefix_comparisons", "frames")
            }
        )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summaries, indent=2)
    )
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()
