#!/usr/bin/env python3
"""Prove Code2Wav non-streaming materialization equivalence.

This isolates the Code2Wav scheduler change:

Old non-streaming path:
    tensor chunk -> CPU float32 numpy per chunk -> np.concatenate -> payload

New non-streaming path:
    tensor chunk -> keep tensor per chunk -> torch.cat -> payload

The full serving benchmark cannot prove byte identity because the thinker/talker
may generate different codec-token sequences across independent runs. This
script holds the decoded vocoder tensors fixed and compares the serialized
audio payload bytes exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sglang_omni.utils.audio_payload import audio_waveform_payload

SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class Case:
    name: str
    dtype: torch.dtype
    chunk_lengths: tuple[int, ...]
    noncontiguous: bool = False


def _make_chunks(case: Case, *, device: torch.device, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    chunks: list[torch.Tensor] = []
    for idx, length in enumerate(case.chunk_lengths):
        # Keep values in a realistic audio range and avoid special values. The
        # proof is about materialization/serialization, not NaN policy.
        values = torch.randn(length, generator=generator, dtype=torch.float32) * 0.2
        values = values.clamp(-1.0, 1.0)
        if case.noncontiguous:
            storage = torch.empty(length * 2, dtype=torch.float32)
            storage[::2] = values
            storage[1::2] = 0
            values = storage[::2]
        chunks.append(values.to(device=device, dtype=case.dtype))
        seed += idx + 1
    return chunks


def _old_non_stream_payload(chunks: list[torch.Tensor]) -> dict:
    numpy_chunks = [
        chunk.reshape(-1).detach().cpu().float().numpy().copy()
        for chunk in chunks
        if chunk.numel() > 0
    ]
    full_audio = np.concatenate(numpy_chunks).astype(np.float32, copy=False)
    return audio_waveform_payload(
        full_audio,
        sample_rate=SAMPLE_RATE,
        modality="audio",
        source_hint="old Code2Wav materialization",
    )


def _new_non_stream_payload(chunks: list[torch.Tensor]) -> dict:
    tensor_chunks = [
        chunk.reshape(-1).detach() for chunk in chunks if chunk.numel() > 0
    ]
    full_audio = torch.cat(tensor_chunks, dim=0)
    return audio_waveform_payload(
        full_audio,
        sample_rate=SAMPLE_RATE,
        modality="audio",
        source_hint="new Code2Wav materialization",
    )


def _payload_digest(payload: dict) -> str:
    return hashlib.sha256(payload["audio_waveform"]).hexdigest()


def _assert_payload_equal(case: Case, chunks: list[torch.Tensor]) -> None:
    old = _old_non_stream_payload(chunks)
    new = _new_non_stream_payload(chunks)

    comparable_keys = (
        "audio_waveform",
        "audio_waveform_shape",
        "audio_waveform_dtype",
        "sample_rate",
        "modality",
    )
    mismatches = [key for key in comparable_keys if old[key] != new[key]]
    if mismatches:
        raise AssertionError(
            f"{case.name}: payload mismatch for {mismatches}; "
            f"old_digest={_payload_digest(old)} new_digest={_payload_digest(new)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Run tensor-side proof on CPU or CUDA. CUDA is optional.",
    )
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    cases = [
        Case("float32_many_chunks", torch.float32, (1, 7, 128, 513, 4096)),
        Case("float16_many_chunks", torch.float16, (3, 31, 257, 2048)),
        Case("bfloat16_many_chunks", torch.bfloat16, (5, 64, 1024, 3333)),
        Case("float32_single_chunk", torch.float32, (8192,)),
        Case("float32_noncontiguous", torch.float32, (17, 255, 4097), True),
    ]

    for case_idx, case in enumerate(cases):
        chunks = _make_chunks(case, device=device, seed=args.seed + case_idx)
        _assert_payload_equal(case, chunks)
        total_samples = sum(int(chunk.numel()) for chunk in chunks)
        print(
            f"PASS {case.name}: dtype={case.dtype} "
            f"chunks={len(chunks)} samples={total_samples}"
        )

    print("PASS: old and new non-streaming Code2Wav payload bytes are identical")


if __name__ == "__main__":
    main()
