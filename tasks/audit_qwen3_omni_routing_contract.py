# SPDX-License-Identifier: Apache-2.0
"""Check recorded plain-softmax expert selection against a float64 reference."""

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import validate


@dataclass(kw_only=True)
class RoutingComparison:
    frame_index: int
    layer_index: int
    compared_rows: int
    changed_selection_rows: int
    cutoff_tie_rows: int
    maximum_weight_error: float
    maximum_weight_sum_error: float


@torch.inference_mode()
def main(root: Path, output: Path) -> None:
    torch.set_num_threads(1)
    captures = sorted(root.glob("captures/*/trajectory.pt"))
    assert captures
    reports = {}
    for capture in captures:
        frames = validate(capture.parent)
        metadata = json.loads(capture.with_suffix(".json").read_text())["frames"]
        comparisons: list[RoutingComparison] = []
        for frame_index, (frame, step) in enumerate(zip(frames, metadata, strict=True)):
            if step["phase"] == "prefill":
                continue
            else:
                assert step["phase"] == "decode"
            for layer_index in range(frame["stored_keys"].shape[0]):
                prefix = f"layer{layer_index}_"
                if f"{prefix}router_logits" not in frame:
                    continue
                elif f"{prefix}peer_router_logits" in frame:
                    prefix += "peer_"
                    live_rows = len(step["request_ids"])
                    logits = frame[f"{prefix}router_logits"][:live_rows]
                    selected_ids = frame[f"{prefix}expert_ids"][:live_rows]
                    selected_weights = frame[f"{prefix}expert_weights"][:live_rows]
                else:
                    logits = frame[f"{prefix}router_logits"].unsqueeze(0)
                    selected_ids = frame[f"{prefix}expert_ids"].unsqueeze(0)
                    selected_weights = frame[f"{prefix}expert_weights"].unsqueeze(0)
                assert torch.isfinite(logits).all()
                assert torch.isfinite(selected_weights).all()
                assert selected_weights.dtype == torch.float32
                assert (selected_weights >= 0).all()
                expert_count = selected_ids.shape[-1]
                assert selected_ids.shape == selected_weights.shape
                assert 0 < expert_count < logits.shape[-1]
                sorted_ids = torch.argsort(
                    logits.double(), descending=True, stable=True
                )
                reference_ids = sorted_ids[:, :expert_count]
                reference_logits = logits.double().gather(1, reference_ids)
                reference_weights = reference_logits.softmax(dim=-1)
                cutoff_logits = logits.gather(
                    1, sorted_ids[:, expert_count : expert_count + 1]
                )
                comparisons.append(
                    RoutingComparison(
                        frame_index=frame_index,
                        layer_index=layer_index,
                        compared_rows=logits.shape[0],
                        changed_selection_rows=int(
                            (selected_ids != reference_ids).any(dim=-1).sum()
                        ),
                        cutoff_tie_rows=int(
                            (reference_logits[:, -1:] == cutoff_logits).sum()
                        ),
                        maximum_weight_error=float(
                            (selected_weights.double() - reference_weights).abs().max()
                        ),
                        maximum_weight_sum_error=float(
                            (selected_weights.double().sum(dim=-1) - 1).abs().max()
                        ),
                    )
                )
        assert comparisons
        reports[capture.parent.name] = {
            "capture_sha256": sha256(capture.read_bytes()).hexdigest(),
            "compared_rows": sum(row.compared_rows for row in comparisons),
            "changed_selection_rows": sum(
                row.changed_selection_rows for row in comparisons
            ),
            "cutoff_tie_rows": sum(row.cutoff_tie_rows for row in comparisons),
            "maximum_weight_error": max(
                row.maximum_weight_error for row in comparisons
            ),
            "maximum_weight_sum_error": max(
                row.maximum_weight_sum_error for row in comparisons
            ),
            "comparisons": [asdict(row) for row in comparisons],
        }
        print(
            json.dumps(
                {
                    name: value
                    for name, value in reports[capture.parent.name].items()
                    if name != "comparisons"
                }
            ),
            flush=True,
        )
    with output.open("x") as receipt:
        json.dump(
            {
                "scope": "Recorded live rows where logits and selections exist. Plain softmax, top-k, renormalize, lowest-ID tie breaking. Float64 softmax over selected logits is the mathematical reference. Weight errors are measured, not a speech-quality acceptance threshold.",
                "captures": reports,
            },
            receipt,
            indent=2,
        )
    assert all(report["changed_selection_rows"] == 0 for report in reports.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.root, arguments.output)
else:
    pass
