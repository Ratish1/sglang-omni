"""Qwen3-Omni's code2wav decodes before and after a refactor: seeded decodes' bytes, hashed.

Run on the box from each tree: cd <tree> && PYTHONPATH=<tree> python3 /workspace/box/tasks/pr2800_review/box/omni_slot_identity_probe.py
The test suite's tiny fused code2wav (seed 0) in bf16 on card 0: two rows decoded from their slots in parts
(4, 6, 7) and (1,) * 5, and the stateless forward of the same codes. Prints the sha256 of each output's bytes;
two trees agree to the bit when the hashes match. Takes the slot decoder under either tree's name.
"""

from __future__ import annotations

import hashlib

import torch

from sglang_omni.models.qwen3_omni.components import code2wav
from tests.unit_test.qwen3_omni.test_code2wav import decode_in_parts, make_fused_tiny_code2wav

if hasattr(code2wav, "Qwen3OmniIncrementalCodec"):
    SLOT_DECODER = code2wav.Qwen3OmniIncrementalCodec
else:
    SLOT_DECODER = code2wav.Code2WavIncrementalDecoder


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    model = make_fused_tiny_code2wav(0)
    decoder = SLOT_DECODER(model)
    generator = torch.Generator(device="cuda").manual_seed(5)
    codes = torch.randint(0, 16, (2, 2, 17), device="cuda", generator=generator)
    print(f"slot decoder {SLOT_DECODER.__name__}", flush=True)
    with torch.inference_mode():
        for parts in ((4, 6, 7), (1,) * 5):
            waveform = decode_in_parts(decoder, codes[..., : sum(parts)], parts)
            print(f"parts {parts}: {tuple(waveform.shape)} sha256 {digest(waveform)}", flush=True)
        whole = model(codes)
        print(f"stateless forward: {tuple(whole.shape)} sha256 {digest(whole)}", flush=True)


if __name__ == "__main__":
    main()
