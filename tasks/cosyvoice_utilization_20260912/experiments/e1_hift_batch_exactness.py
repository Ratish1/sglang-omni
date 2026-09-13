#!/usr/bin/env python3
"""E1: the batched non-final HiFT call equals the per-row call, bit for bit.

Run on the H100 venv at the branch head:

  python tasks/cosyvoice_utilization_20260912/experiments/e1_hift_batch_exactness.py \
      --checkpoint /path/to/Fun-CosyVoice3-0.5B-2512 --device cuda:0

Random mel exercises the padding, slicing and offset math and the unvoiced
source path; voiced speech is covered by the corpus WER gate. The assert inside
hift_delta_batch also checks the derived sample count against the real HiFT.
"""

from __future__ import annotations

import argparse
import random

import torch

from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    load_cosyvoice3_flow_hift,
)

# Generated frames after each hop of the 25, 50, 100, 100, 100 token ladder.
LADDER_FRAMES = (50, 150, 350, 550, 750)


def _random_mel(
    frames: int, *, device: str, generator: torch.Generator
) -> torch.Tensor:
    # Log-mel range of the Flow output; the values only steer the source path.
    return torch.randn(1, 80, frames, device=device, generator=generator) * 2.0 - 6.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[2, 8, 16])
    args = parser.parse_args()

    flow, hift = load_cosyvoice3_flow_hift(args.checkpoint, args.device)
    vocoder = CosyVoice3Vocoder(flow, hift)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    rng = random.Random(args.seed)

    worst = 0.0
    checked = 0
    for rows in args.rows:
        histories: list[torch.Tensor | None] = []
        new_mels: list[torch.Tensor] = []
        offsets: list[int] = []
        for _ in range(rows):
            hop = rng.randrange(len(LADDER_FRAMES))
            history_frames = LADDER_FRAMES[hop - 1] if hop else 0
            if history_frames:
                histories.append(
                    _random_mel(history_frames, device=args.device, generator=generator)
                )
                offsets.append(vocoder.hift_streaming_samples(history_frames))
            else:
                histories.append(None)
                offsets.append(0)
            new_mels.append(
                _random_mel(
                    LADDER_FRAMES[hop] - history_frames,
                    device=args.device,
                    generator=generator,
                )
            )
        with torch.inference_mode():
            expected = [
                vocoder.hift_delta(
                    new_mel, hift_mel=history, speech_offset=offset, finalize=False
                )
                for new_mel, history, offset in zip(
                    new_mels, histories, offsets, strict=True
                )
            ]
            batched = vocoder.hift_delta_batch(
                new_mels, hift_mels=histories, speech_offsets=offsets
            )
        for index, (
            (delta, mel, offset),
            (want_delta, want_mel, want_offset),
        ) in enumerate(zip(batched, expected, strict=True)):
            assert offset == want_offset, (rows, index, offset, want_offset)
            assert torch.equal(mel, want_mel), (rows, index)
            if delta.shape != want_delta.shape:
                diff = float("inf")
            else:
                diff = float((delta - want_delta).abs().max())
            worst = max(worst, diff)
            checked += 1
            print(
                f"rows={rows} row={index} frames={mel.shape[-1]} "
                f"samples={delta.shape[-1]} max_abs_diff={diff:.3e}"
            )
    verdict = "BIT EXACT" if worst == 0.0 else "NOT bit exact"
    print(f"checked {checked} rows; worst max_abs_diff {worst:.3e}; {verdict}")


if __name__ == "__main__":
    main()
