"""Compare captured last-token FA3 attention against an FP32 GQA reference."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    comparisons = []
    for capture_path in sorted(arguments.capture_directory.glob("*/request-*.pt")):
        captured = torch.load(capture_path, map_location="cpu", weights_only=True)
        for step_index, snapshot in enumerate(captured["snapshots"]):
            for name, query in snapshot.items():
                matched = re.search(
                    r"(?:^|\.)layers\.(\d+)\.self_attn\.attn\.diagnostic_query$", name
                )
                if matched is None:
                    continue
                else:
                    pass
                layer_index = int(matched.group(1))
                prefix = name.removesuffix("diagnostic_query")
                position = int(snapshot[f"{prefix}diagnostic_position"])
                sequence_length = int(snapshot["sequence_length"])
                assert position + 1 == sequence_length, (capture_path, name, position)
                assert torch.equal(
                    snapshot["request_locations"], snapshot["read_locations"]
                )
                scaling, logit_cap, window_size = snapshot[
                    f"{prefix}diagnostic_attention_properties"
                ].tolist()
                assert logit_cap == 0.0 and window_size == -1
                keys = snapshot[f"layer_{layer_index}_keys"].float().cuda()
                values = snapshot[f"layer_{layer_index}_values"].float().cuda()
                query = query.float().cuda()
                observed = (
                    snapshot[f"{prefix}diagnostic_attention_output"].float().cuda()
                )
                query_heads = query.shape[0]
                kv_heads = keys.shape[1]
                assert query_heads % kv_heads == 0
                keys = keys.repeat_interleave(query_heads // kv_heads, dim=1)
                values = values.repeat_interleave(query_heads // kv_heads, dim=1)
                scores = torch.einsum("hd,shd->hs", query, keys) * scaling
                reference = torch.einsum(
                    "hs,shd->hd", torch.softmax(scores, dim=-1), values
                )
                difference = observed - reference
                comparisons.append(
                    {
                        "request_id": captured["request_id"],
                        "stage": capture_path.parent.name,
                        "snapshot_step": step_index,
                        "layer": layer_index,
                        "sequence_length": sequence_length,
                        "max_absolute_error": float(difference.abs().max()),
                        "relative_l2_error": float(
                            torch.linalg.vector_norm(difference)
                            / torch.linalg.vector_norm(reference)
                        ),
                        "cosine_similarity": float(
                            torch.nn.functional.cosine_similarity(
                                observed.flatten(), reference.flatten(), dim=0
                            )
                        ),
                        "finite": bool(torch.isfinite(observed).all()),
                    }
                )
    assert comparisons
    report = {
        "reference": "FP32 QK, softmax and V reduction, TF32 disabled, last query attends entire prefix and current token",
        "comparison_count": len(comparisons),
        "max_relative_l2_error": max(row["relative_l2_error"] for row in comparisons),
        "max_absolute_error": max(row["max_absolute_error"] for row in comparisons),
        "minimum_cosine_similarity": min(
            row["cosine_similarity"] for row in comparisons
        ),
        "all_finite": all(row["finite"] for row in comparisons),
        "comparisons": comparisons,
    }
    arguments.output_file.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {name: value for name, value in report.items() if name != "comparisons"}
        )
    )


if __name__ == "__main__":
    main()
