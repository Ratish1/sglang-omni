# SPDX-License-Identifier: Apache-2.0
"""Verify paged FP32 attention under graph replay on retained native histories."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import validate
from qwen3_omni_attention_precision_trial import PrecisionAttentionBuffers
from replay_qwen3_omni_native_attention import scalar_attention


@torch.inference_mode()
def main(root: Path, gpu_index: int) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu_index)
    captures = sorted(root.glob("captures/*/trajectory.pt"))
    assert captures
    reports = []
    for capture in captures:
        frames = validate(capture.parent)
        steps = json.loads(capture.with_suffix(".json").read_text())["frames"]
        stored_keys = torch.cat([frame["stored_keys"] for frame in frames], dim=1)
        stored_values = torch.cat([frame["stored_values"] for frame in frames], dim=1)
        slots = torch.cat([frame["output_slots"] for frame in frames]).long()
        assert slots.numel() == stored_keys.shape[1]
        assert slots.unique().numel() == slots.numel()
        cache_slots = int(slots.max()) + 1
        maximum_sequence_tokens = max(step["sequence_length"] for step in steps)
        assert maximum_sequence_tokens == slots.numel()
        graph_checks = 0
        compared_rows = 0
        for layer_index in range(stored_keys.shape[0]):
            frame_indices = [
                index
                for index, step in enumerate(steps)
                if index > 0 and layer_index < step.get("captured_layers", 1)
            ]
            assert frame_indices
            keys = torch.full(
                (cache_slots, *stored_keys.shape[2:]),
                float("nan"),
                device="cuda",
                dtype=torch.bfloat16,
            )
            values = torch.full_like(keys, float("nan"))
            keys.index_copy_(0, slots.cuda(), stored_keys[layer_index].cuda())
            values.index_copy_(0, slots.cuda(), stored_values[layer_index].cuda())
            query_heads = (
                frames[1][f"layer{layer_index}_query_after_rope"].numel()
                // keys.shape[-1]
            )
            batch_sizes = sorted(
                {1, max(step["graph_batch_rows"] for step in steps[1:])}
            )
            for batch_rows in batch_sizes:
                buffers = PrecisionAttentionBuffers(
                    batch_rows,
                    maximum_sequence_tokens,
                    query_heads,
                    keys.shape[-1],
                    keys.device,
                )
                query = torch.empty(
                    (batch_rows, query_heads, keys.shape[-1]),
                    device=keys.device,
                    dtype=keys.dtype,
                )
                page_table = torch.full(
                    (batch_rows, maximum_sequence_tokens),
                    -1,
                    device=keys.device,
                    dtype=torch.int32,
                )
                sequence_lengths = torch.empty(
                    batch_rows, device=keys.device, dtype=torch.int32
                )
                cumulative_lengths = torch.empty(
                    batch_rows + 1, device=keys.device, dtype=torch.int32
                )
                graph = torch.cuda.CUDAGraph()
                captured_output: torch.Tensor | None = None
                for visit, frame_index in enumerate(
                    frame_indices + frame_indices[::-1]
                ):
                    row_frames = [
                        frame_indices[
                            (frame_indices.index(frame_index) + row)
                            % len(frame_indices)
                        ]
                        for row in range(batch_rows)
                    ]
                    query.copy_(
                        torch.stack(
                            [
                                frames[index][
                                    f"layer{layer_index}_query_after_rope"
                                ].reshape(query_heads, -1)
                                for index in row_frames
                            ]
                        )
                    )
                    lengths = [steps[index]["sequence_length"] for index in row_frames]
                    sequence_lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
                    cumulative_lengths[0].zero_()
                    torch.cumsum(sequence_lengths, dim=0, out=cumulative_lengths[1:])
                    page_table.fill_(-1)
                    for row, length in enumerate(lengths):
                        assert torch.equal(
                            frames[row_frames[row]]["attention_read_slots"],
                            slots[:length],
                        )
                        page_table[row, :length].copy_(slots[:length])
                    buffers.kv_indices.fill_(-1)
                    buffers.output.fill_(float("nan"))
                    buffers.partial_output.fill_(float("nan"))
                    buffers.partial_log_partition.fill_(float("nan"))
                    expected = torch.cat(
                        [
                            scalar_attention(
                                query[row : row + 1],
                                stored_keys[layer_index, :length].cuda(),
                                stored_values[layer_index, :length].cuda(),
                                1,
                            ).bfloat16()
                            for row, length in enumerate(lengths)
                        ]
                    )
                    eager = buffers.forward(
                        query,
                        keys,
                        values,
                        page_table,
                        sequence_lengths,
                        cumulative_lengths,
                        True,
                    )
                    assert torch.equal(eager, expected), (
                        layer_index,
                        batch_rows,
                        frame_index,
                        "eager",
                    )
                    if visit == 0:
                        torch.cuda.synchronize()
                        with torch.cuda.graph(graph):
                            captured_output = buffers.forward(
                                query,
                                keys,
                                values,
                                page_table,
                                sequence_lengths,
                                cumulative_lengths,
                                True,
                            )
                    else:
                        pass
                    buffers.kv_indices.fill_(-1)
                    buffers.output.fill_(float("nan"))
                    buffers.partial_output.fill_(float("nan"))
                    buffers.partial_log_partition.fill_(float("nan"))
                    graph.replay()
                    assert captured_output is not None
                    assert torch.equal(captured_output, expected), (
                        layer_index,
                        batch_rows,
                        frame_index,
                        "graph",
                    )
                    expected_indices = torch.cat(
                        [slots[:length] for length in lengths]
                    ).int()
                    assert torch.equal(
                        buffers.kv_indices[: expected_indices.numel()].cpu(),
                        expected_indices,
                    )
                    assert torch.equal(
                        keys.index_select(0, slots.cuda()).cpu(),
                        stored_keys[layer_index],
                    )
                    assert torch.equal(
                        values.index_select(0, slots.cuda()).cpu(),
                        stored_values[layer_index],
                    )
                    graph_checks += 1
                    compared_rows += batch_rows
        reports.append(
            {
                "sample_id": capture.parent.name,
                "capture_sha256": sha256(capture.read_bytes()).hexdigest(),
                "graph_checks": graph_checks,
                "compared_rows": compared_rows,
                "native_slots_preserved": True,
                "eager_and_graph_match_scalar_reference": True,
                "scratch_poison_and_reverse_order_pass": True,
            }
        )
        print(json.dumps(reports[-1]), flush=True)
    with (root / "attention-precision-graph.json").open("x") as receipt:
        json.dump(
            {
                "scope": "Real retained queries and logical prefixes at their original physical slots. Controlled peer rows cycle through retained states of the same request, not the original live cohort. Graph replay grows/shrinks row lengths and poisons scratch between calls. This is an operator/metadata gate, not native speech or performance qualification.",
                "captures": reports,
            },
            receipt,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    arguments = parser.parse_args()
    main(arguments.root, arguments.gpu)
else:
    pass
