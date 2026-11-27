# SPDX-License-Identifier: Apache-2.0
"""Check native precision continuations against their exact attention operator."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Literal

import torch
from compare_qwen3_omni_native_trajectories import validate
from replay_qwen3_omni_native_attention import (
    reference_attention,
    replay_attention,
    scalar_attention,
)
from replay_qwen3_omni_scheduler_bounds import physical_schedule


@torch.inference_mode()
def main(root: Path, mode: Literal["control", "scalar"], gpu_index: int) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu_index)
    captures = sorted(root.glob("captures/*/trajectory.pt"))
    assert captures
    reports = []
    for capture in captures:
        frames = validate(capture.parent)
        steps = json.loads(capture.with_suffix(".json").read_text())["frames"]
        keys = torch.cat([frame["stored_keys"] for frame in frames], dim=1).cuda()
        values = torch.cat([frame["stored_values"] for frame in frames], dim=1).cuda()
        comparisons = []
        for frame_index, (frame, step) in enumerate(zip(frames, steps, strict=True)):
            assert step["attention_precision_trial"] == mode
            if frame_index == 0:
                continue
            else:
                assert step["phase"] == "decode"
            assert step["precision_captured_layers"] == keys.shape[0]
            assert step["unselected_attention_unchanged"]
            sequence_tokens = step["sequence_length"]
            schedule = physical_schedule(
                frame["attention_scheduler_metadata"], step["graph_batch_rows"]
            )
            native_splits = int(schedule[0, step["request_row"]])
            for layer_index in range(keys.shape[0]):
                query = frame[f"layer{layer_index}_query_after_rope"].cuda()
                query = query.reshape(1, -1, keys.shape[-1])
                layer_keys = keys[layer_index, :sequence_tokens]
                layer_values = values[layer_index, :sequence_tokens]
                if mode == "scalar":
                    replay = scalar_attention(query, layer_keys, layer_values, 1)
                else:
                    replay = replay_attention(
                        query, layer_keys, layer_values, native_splits
                    )
                expected = frame[f"layer{layer_index}_attention_context"].flatten()
                actual = replay.bfloat16().cpu().flatten()
                assert torch.equal(actual, expected), (frame_index, layer_index)
                reference = reference_attention(query, layer_keys, layer_values)
                reference = reference.flatten()
                assert torch.isfinite(reference).all() and reference.norm() > 0
                comparisons.append(
                    {
                        "frame": frame_index,
                        "layer": layer_index,
                        "sequence_tokens": sequence_tokens,
                        "relative_l2_error_fp64": float(
                            (actual.double() - reference).norm() / reference.norm()
                        ),
                        "rounded_reference_changed_values": int(
                            (actual != reference.bfloat16()).count_nonzero()
                        ),
                    }
                )
        for field, actual in (("stored_keys", keys), ("stored_values", values)):
            assert torch.equal(
                actual.cpu(), torch.cat([frame[field] for frame in frames], dim=1)
            )
        reports.append(
            {
                "sample_id": capture.parent.name,
                "capture_sha256": sha256(capture.read_bytes()).hexdigest(),
                "frames_including_eos": len(frames),
                "exact_operator_checks": len(comparisons),
                "comparisons": comparisons,
            }
        )
        print(
            json.dumps(
                {
                    name: value
                    for name, value in reports[-1].items()
                    if name != "comparisons"
                }
            ),
            flush=True,
        )
    with (root / "native-attention-precision.json").open("x") as receipt:
        json.dump({"mode": mode, "captures": reports}, receipt, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", choices=("control", "scalar"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    arguments = parser.parse_args()
    main(arguments.root, arguments.mode, arguments.gpu)
else:
    pass
