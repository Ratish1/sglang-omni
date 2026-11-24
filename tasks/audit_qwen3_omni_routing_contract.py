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
    checkpoint_changed_order_rows: int | None
    checkpoint_changed_set_rows: int | None
    checkpoint_changed_set_without_probability_tie_rows: int | None
    checkpoint_probability_cutoff_tie_rows: int | None
    checkpoint_weight_rounding_error: float | None
    checkpoint_changed_rounded_weight_rows: int | None
    checkpoint_order_changes_rounded_weight_rows: int | None


@torch.inference_mode()
def main(root: Path, output: Path, gpu_index: int | None) -> None:
    torch.set_num_threads(1)
    if gpu_index is not None:
        torch.cuda.set_device(gpu_index)
    else:
        pass
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
                    padded_logits = frame[f"{prefix}router_logits"]
                    logits = padded_logits[:live_rows]
                    selected_ids = frame[f"{prefix}expert_ids"][:live_rows]
                    selected_weights = frame[f"{prefix}expert_weights"][:live_rows]
                else:
                    logits = frame[f"{prefix}router_logits"].unsqueeze(0)
                    padded_logits = logits
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
                if gpu_index is None:
                    checkpoint_changed_order_rows = None
                    checkpoint_changed_set_rows = None
                    checkpoint_changed_set_without_probability_tie_rows = None
                    checkpoint_probability_cutoff_tie_rows = None
                    checkpoint_weight_rounding_error = None
                    checkpoint_changed_rounded_weight_rows = None
                    checkpoint_order_changes_rounded_weight_rows = None
                else:
                    checkpoint_logits = padded_logits.to(
                        torch.device("cuda", gpu_index)
                    )
                    assert checkpoint_logits.dtype == torch.bfloat16
                    probabilities = checkpoint_logits.softmax(
                        dim=-1, dtype=torch.float32
                    )
                    checkpoint_weights, checkpoint_ids = torch.topk(
                        probabilities, expert_count, dim=-1
                    )
                    checkpoint_weights /= checkpoint_weights.sum(dim=-1, keepdim=True)
                    rounded_weights = checkpoint_weights.to(checkpoint_logits.dtype)
                    probability_order = torch.argsort(
                        probabilities, descending=True, stable=True
                    )
                    cutoff_values = probabilities.gather(
                        1, probability_order[:, expert_count - 1 : expert_count + 1]
                    )
                    probability_ties = (cutoff_values[:, 0] == cutoff_values[:, 1])[
                        : logits.shape[0]
                    ]
                    checkpoint_ids = checkpoint_ids[: logits.shape[0]].cpu()
                    checkpoint_order = checkpoint_ids.argsort(dim=-1)
                    native_order = selected_ids.argsort(dim=-1)
                    sorted_checkpoint_weights = (
                        rounded_weights[: logits.shape[0]]
                        .cpu()
                        .gather(1, checkpoint_order)
                    )
                    checkpoint_changed_rounded_weight_rows = int(
                        (
                            sorted_checkpoint_weights
                            != selected_weights.to(checkpoint_logits.dtype).gather(
                                1, native_order
                            )
                        )
                        .any(dim=-1)
                        .sum()
                    )
                    native_order_weights = probabilities[: logits.shape[0]].gather(
                        1, selected_ids.to(probabilities.device).long()
                    )
                    native_order_weights /= native_order_weights.sum(
                        dim=-1, keepdim=True
                    )
                    checkpoint_order_changes_rounded_weight_rows = int(
                        (
                            sorted_checkpoint_weights
                            != native_order_weights.to(checkpoint_logits.dtype)
                            .cpu()
                            .gather(1, native_order)
                        )
                        .any(dim=-1)
                        .sum()
                    )
                    changed_sets = (
                        checkpoint_ids.sort(dim=-1).values
                        != selected_ids.sort(dim=-1).values
                    ).any(dim=-1)
                    checkpoint_changed_order_rows = int(
                        (checkpoint_ids != selected_ids).any(dim=-1).sum()
                    )
                    checkpoint_changed_set_rows = int(changed_sets.sum())
                    checkpoint_changed_set_without_probability_tie_rows = int(
                        (changed_sets & ~probability_ties.cpu()).sum()
                    )
                    checkpoint_probability_cutoff_tie_rows = int(probability_ties.sum())
                    checkpoint_weight_rounding_error = float(
                        (checkpoint_weights - rounded_weights.float())[
                            : logits.shape[0]
                        ]
                        .abs()
                        .max()
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
                        checkpoint_changed_order_rows=checkpoint_changed_order_rows,
                        checkpoint_changed_set_rows=checkpoint_changed_set_rows,
                        checkpoint_changed_set_without_probability_tie_rows=checkpoint_changed_set_without_probability_tie_rows,
                        checkpoint_probability_cutoff_tie_rows=checkpoint_probability_cutoff_tie_rows,
                        checkpoint_weight_rounding_error=checkpoint_weight_rounding_error,
                        checkpoint_changed_rounded_weight_rows=checkpoint_changed_rounded_weight_rows,
                        checkpoint_order_changes_rounded_weight_rows=checkpoint_order_changes_rounded_weight_rows,
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
        if gpu_index is not None:
            reports[capture.parent.name]["checkpoint_selection"] = {
                "changed_order_rows": sum(
                    row.checkpoint_changed_order_rows for row in comparisons
                ),
                "changed_set_rows": sum(
                    row.checkpoint_changed_set_rows for row in comparisons
                ),
                "changed_set_without_probability_tie_rows": sum(
                    row.checkpoint_changed_set_without_probability_tie_rows
                    for row in comparisons
                ),
                "probability_cutoff_tie_rows": sum(
                    row.checkpoint_probability_cutoff_tie_rows for row in comparisons
                ),
                "maximum_weight_rounding_error": max(
                    row.checkpoint_weight_rounding_error for row in comparisons
                ),
                "changed_rounded_weight_rows": sum(
                    row.checkpoint_changed_rounded_weight_rows for row in comparisons
                ),
                "order_changes_rounded_weight_rows": sum(
                    row.checkpoint_order_changes_rounded_weight_rows
                    for row in comparisons
                ),
            }
        else:
            pass
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
                "scope": "Recorded live rows where logits and selections exist. Plain softmax, top-k, renormalize, lowest-ID tie breaking. Float64 softmax over selected logits is the mathematical reference. Optional GPU comparison uses the checkpoint router's FP32 softmax, torch.topk, renormalization and activation-dtype weight cast on identical recorded logits. It isolates selection from gate projection; it is not free continuation or a speech-quality test.",
                "checkpoint_comparison_gpu": gpu_index,
                "torch_version": torch.__version__,
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
    parser.add_argument("--gpu", type=int)
    arguments = parser.parse_args()
    main(arguments.root, arguments.output, arguments.gpu)
else:
    pass
