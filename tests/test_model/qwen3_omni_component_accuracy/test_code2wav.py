# SPDX-License-Identifier: Apache-2.0
"""Omni's served code2wav against the HF code2wav in FP32, with HF's own BF16 copy as the floor.

Omni builds code2wav as the code2wav stage does (channels-last convs, fused pre-transformer and SnakeBeta, CUDA
graphs per window) and decodes a stream in windows of 10 new frames with up to 25 frames of left context
(code2wav_scheduler.py decode_windows and decode_delta). The floor rule covers one whole-sequence forward and that
window walk run on both sides (graph outputs are borrowed until the next replay, so each window is copied out as the
scheduler does). The walk's own departure from a whole-sequence decode, beside the departure of HF's chunked_decode
(300 frames, 25 of context, the HF generate path) from the same decode, is a semantic difference, reported, not
gated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration

from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    create_code2wav_scheduler,
)
from tests.test_model.qwen3_omni_component_accuracy.comparison import (
    CaseRecord,
    deviation,
    floor_violations,
    geometric_mean_ratio,
    parameter_mismatches,
    reference_copy,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

FRAME_COUNTS = (1, 10, 35, 36, 72, 73, 150, 301, 600)
CODE2WAV_MEMORY_FRACTION = 0.02
HF_CHUNK_FRAMES = 300
HF_CONTEXT_FRAMES = 25
SEED = 20261011


def window_walk(
    forward, codes: torch.Tensor, chunk: int, context: int, upsample: int
) -> torch.Tensor:
    """The served stream: each window adds chunk frames over up to context frames and keeps its new samples."""
    frames = codes.shape[-1]
    pieces = []
    emitted = 0
    while emitted < frames:
        new_frames = min(chunk, frames - emitted)
        left = min(context, emitted)
        wav = forward(codes[..., emitted - left : emitted + new_frames])
        pieces.append(wav[..., -new_frames * upsample :])
        emitted += new_frames
    return torch.cat(pieces, dim=-1)


def best_alignment(
    longer: torch.Tensor, shorter: torch.Tensor, *, expected: int, radius: int
) -> tuple[int, float]:
    """The offset of shorter inside longer, searched within radius of expected, with the smallest relative L2."""
    longer = longer.flatten().float().cpu()
    shorter = shorter.flatten().float().cpu()
    offsets = range(
        max(expected - radius, 0),
        min(expected + radius, longer.numel() - shorter.numel()) + 1,
    )
    errors = {
        offset: (
            (longer[offset : offset + shorter.numel()] - shorter).norm()
            / shorter.norm()
        ).item()
        for offset in offsets
    }
    best = min(errors, key=errors.__getitem__)
    return best, errors[best]


def test_code2wav_follows_reference(
    checkpoint_dir: str,
    reference_model: Qwen3OmniMoeForConditionalGeneration,
    component_device: str,
    report_dir: Path | None,
) -> None:
    truth = reference_copy(
        reference_model.code2wav, device=component_device, dtype=torch.float32
    )
    floor = reference_copy(
        reference_model.code2wav, device=component_device, dtype=torch.bfloat16
    )
    card = torch.device(component_device)
    scheduler = create_code2wav_scheduler(
        checkpoint_dir,
        device=card.type,
        gpu_id=card.index,
        enable_cuda_graph=True,
        total_gpu_memory_fraction=CODE2WAV_MEMORY_FRACTION,
    )
    assert scheduler.cuda_graph_runner is not None
    served_state = {
        name.replace("pre_transformer.transformer.", "pre_transformer."): tensor
        for name, tensor in scheduler.model.state_dict().items()
    }
    mismatches = parameter_mismatches(served_state, floor.state_dict())
    assert not mismatches, f"omni code2wav weights differ: {mismatches[:10]}"

    chunk = scheduler.stream_chunk_size
    context = scheduler.left_context_size
    upsample = scheduler.total_upsample
    generator = torch.Generator().manual_seed(SEED)
    quantizers = truth.config.num_quantizers
    codebook = truth.config.codebook_size

    def served_forward(codes: torch.Tensor) -> torch.Tensor:
        return scheduler.forward_codes(codes, graph_eligible=True)[0].clone()

    records: list[CaseRecord] = []
    semantics: list[dict[str, object]] = []
    for frames in FRAME_COUNTS:
        codes = torch.randint(
            0, codebook, (1, quantizers, frames), generator=generator
        ).to(component_device)
        with torch.inference_mode():
            whole_truth = truth(codes)
            records.append(
                CaseRecord(
                    case=f"whole/{frames}",
                    omni=deviation(
                        scheduler.forward_codes(codes, graph_eligible=False)[0],
                        whole_truth,
                    ),
                    floor=deviation(floor(codes), whole_truth),
                )
            )
            walk_truth = window_walk(truth, codes, chunk, context, upsample)
            records.append(
                CaseRecord(
                    case=f"walk/{frames}",
                    omni=deviation(
                        window_walk(served_forward, codes, chunk, context, upsample),
                        walk_truth,
                    ),
                    floor=deviation(
                        window_walk(floor, codes, chunk, context, upsample), walk_truth
                    ),
                )
            )
            chunked_truth = truth.chunked_decode(
                codes, chunk_size=HF_CHUNK_FRAMES, left_context_size=HF_CONTEXT_FRAMES
            )
        trim = frames * upsample - whole_truth.shape[-1]
        whole_offset, whole_error = best_alignment(
            walk_truth, whole_truth, expected=0, radius=trim
        )
        chunk_errors = []
        cursor = 0
        for chunk_start in range(0, frames, HF_CHUNK_FRAMES):
            samples = min(HF_CHUNK_FRAMES, frames - chunk_start) * upsample - trim
            piece = chunked_truth[..., cursor : cursor + samples]
            cursor += samples
            chunk_errors.append(
                best_alignment(
                    whole_truth, piece, expected=chunk_start * upsample, radius=trim
                )[1]
            )
        semantics.append(
            {
                "frames": frames,
                "walk_samples": walk_truth.shape[-1],
                "whole_samples": whole_truth.shape[-1],
                "chunked_samples": chunked_truth.shape[-1],
                "walk_vs_whole_offset": whole_offset,
                "walk_vs_whole_relative_l2": whole_error,
                "chunked_pieces_vs_whole_relative_l2": chunk_errors,
            }
        )

    if report_dir is not None:
        (report_dir / "code2wav.json").write_text(
            json.dumps(
                {
                    "records": [record.as_json() for record in records],
                    "semantics": semantics,
                },
                indent=1,
            )
        )
    else:
        pass
    print(
        f"\n[code2wav] floor ratio geometric mean {geometric_mean_ratio(records):.3f}, per case: "
        + ", ".join(f"{record.case} {record.floor_ratio:.2f}" for record in records)
        + "\n[code2wav] against HF's FP32 whole-sequence decode (relative L2 at best offset): "
        + ", ".join(
            f"{entry['frames']}f walk {entry['walk_vs_whole_relative_l2']:.2e}@{entry['walk_vs_whole_offset']} "
            f"hf chunked {max(entry['chunked_pieces_vs_whole_relative_l2']):.2e}"
            for entry in semantics
        )
    )
    violations = floor_violations(records)
    assert not violations, "code2wav strays past the BF16 floor:\n" + "\n".join(
        violations
    )
