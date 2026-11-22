# SPDX-License-Identifier: Apache-2.0
"""Compare FA3 scheduling bounds on recorded native decode batches."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sgl_kernel.flash_attn import get_scheduler_metadata


@dataclass(kw_only=True)
class FrameComparison:
    capture: str
    frame_index: int
    batch_requests: int
    maximum_sequence_tokens: int
    selected_sequence_tokens: int
    selected_static_splits: int
    selected_tight_splits: int
    changed_requests: int
    static_splits: list[int]
    tight_splits: list[int]


def physical_schedule(metadata: torch.Tensor, batch_requests: int) -> torch.Tensor:
    aligned_requests = (batch_requests + 3) // 4 * 4
    assert metadata.numel() == 4 * aligned_requests + 1
    vectors = metadata[:-1].reshape(4, aligned_requests)[:, :batch_requests]
    permutation = vectors[2].long()
    assert torch.equal(permutation.sort().values, torch.arange(batch_requests))
    schedule = torch.empty((3, batch_requests), dtype=torch.int32)
    schedule[:, permutation] = vectors[[0, 1, 3]]
    return schedule


def main(root: Path, maximum_context_tokens: int) -> None:
    torch.set_num_threads(1)
    comparisons: list[FrameComparison] = []
    for capture in sorted(root.glob("*/captures/*/trajectory.pt")):
        frames = torch.load(capture, weights_only=True, map_location="cpu")
        steps = json.loads(capture.with_suffix(".json").read_text())["frames"]
        assert len(frames) == len(steps)
        for frame_index, (frame, step) in enumerate(zip(frames, steps, strict=True)):
            if step["phase"] == "prefill":
                continue
            else:
                assert step["phase"] == "decode" and step["attention_num_splits"] == 0
            sequence_lengths = frame["attention_sequence_lengths"]
            query_offsets = frame["attention_query_offsets"]
            batch_requests = sequence_lengths.numel()
            assert batch_requests == step["graph_batch_rows"]
            assert torch.equal(query_offsets, torch.arange(batch_requests + 1))
            key_heads, head_width = frame["attention_read_keys"].shape[1:]
            query_heads = frame["layer0_query_after_rope"].numel() // head_width
            assert (query_heads, key_heads, head_width) == (16, 2, 128)
            assert frame["attention_read_keys"].dtype == torch.bfloat16
            maximum_sequence_tokens = int(sequence_lengths.max())
            assert maximum_sequence_tokens <= maximum_context_tokens
            schedules: list[torch.Tensor] = []
            for sequence_bound_tokens in (
                maximum_context_tokens,
                maximum_sequence_tokens,
            ):
                metadata = get_scheduler_metadata(
                    batch_size=batch_requests,
                    max_seqlen_q=1,
                    max_seqlen_k=sequence_bound_tokens,
                    num_heads=query_heads,
                    num_heads_k=key_heads,
                    headdim=head_width,
                    cache_seqlens=sequence_lengths.cuda(),
                    qkv_dtype=torch.bfloat16,
                    cu_seqlens_q=query_offsets.cuda(),
                    page_size=1,
                    causal=True,
                    has_softcap=False,
                    num_splits=0,
                )
                schedules.append(physical_schedule(metadata.cpu(), batch_requests))
            native_schedule = physical_schedule(
                frame["attention_scheduler_metadata"], batch_requests
            )
            assert torch.equal(schedules[0], native_schedule), (capture, frame_index)
            request_row = step["request_row"]
            comparisons.append(
                FrameComparison(
                    capture=str(capture.relative_to(root)),
                    frame_index=frame_index,
                    batch_requests=batch_requests,
                    maximum_sequence_tokens=maximum_sequence_tokens,
                    selected_sequence_tokens=step["sequence_length"],
                    selected_static_splits=int(schedules[0][0, request_row]),
                    selected_tight_splits=int(schedules[1][0, request_row]),
                    changed_requests=int((schedules[0][0] != schedules[1][0]).sum()),
                    static_splits=schedules[0][0].tolist(),
                    tight_splits=schedules[1][0].tolist(),
                )
            )
    assert comparisons
    destination = root / "scheduler-bound-replay.json"
    with destination.open("x") as receipt:
        json.dump([asdict(comparison) for comparison in comparisons], receipt, indent=2)
    changed_frames = sum(comparison.changed_requests > 0 for comparison in comparisons)
    selected_changes = sum(
        comparison.selected_static_splits != comparison.selected_tight_splits
        for comparison in comparisons
    )
    print(
        f"frames={len(comparisons)} changed_frames={changed_frames} selected_changes={selected_changes}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--maximum-context-tokens", type=int, required=True)
    arguments = parser.parse_args()
    main(arguments.root, arguments.maximum_context_tokens)
else:
    pass
