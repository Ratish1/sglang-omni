# SPDX-License-Identifier: Apache-2.0
"""Is CosyVoice's native DiT bit identical with x_transformers' RoPE autocast
regions removed? Eager against eager, the serving dtypes, both streaming modes.

    python c9_rope_exact.py --out DIR

RotaryEmbedding.forward and apply_rotary_pos_emb run under
@autocast('cuda', enabled=False); the region puts _enter_autocast into every
compiled native graph and the AOTAutograd cache refuses it. The replacements
keep the math: the rotary einsum is an outer product (no reduction), written as
a float32 broadcast multiply; apply_rotary_pos_emb is elementwise, which
autocast does not change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.utils.checkpoint import resolve_checkpoint

SHAPES = ((1, 100), (2, 128), (4, 256), (16, 544), (32, 512), (8, 1024))


def rotary_forward(self, t, offset=0):
    if t.ndim == 1:
        t = t[None]
    freqs = t.type_as(self.inv_freq)[..., None] * self.inv_freq
    freqs = freqs / self.interpolation_factor
    freqs = torch.stack((freqs, freqs), dim=-1).flatten(-2)
    assert self.scale is None
    return freqs, 1.0


def apply_rotary_pos_emb(t, freqs, scale=1):
    from x_transformers.x_transformers import rotate_half

    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:, :]
    assert not torch.is_tensor(scale)
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs[:, None]
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim=-1).type(orig_dtype)


def inputs(batch: int, frames: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(batch * 10000 + frames)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    mask = torch.ones(batch, 1, frames, device="cuda", dtype=dtype)
    mask[batch // 2 :, :, frames * 3 // 4 :] = 0
    return (
        randn(batch, 80, frames),
        mask,
        randn(batch, 80, frames),
        torch.full((batch,), 0.37, device="cuda", dtype=dtype),
        randn(batch, 80),
        randn(batch, 80, frames),
    )


def forwards(estimator) -> dict[str, torch.Tensor]:
    outputs = {}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for streaming in (False, True):
            for batch, frames in SHAPES:
                name = f"b{batch}_t{frames}_{'stream' if streaming else 'full'}"
                args = inputs(batch, frames, torch.bfloat16)
                outputs[name] = estimator(*args, streaming=streaming).float().cpu()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    flow, _ = stages.load_cosyvoice3_flow_hift(
        resolve_checkpoint(args.model), device="cuda:0"
    )
    stages.patch_chunk_mask()
    estimator = flow.decoder.estimator
    for module in estimator.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
        else:
            pass
    reference = forwards(estimator)

    from cosyvoice.flow.DiT import modules as dit_modules

    estimator.rotary_embed.forward = rotary_forward.__get__(estimator.rotary_embed)
    dit_modules.apply_rotary_pos_emb = apply_rotary_pos_emb
    candidate = forwards(estimator)

    report = {
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "equal": {
            name: torch.equal(reference[name], candidate[name]) for name in reference
        },
        "max_abs": {
            name: (reference[name] - candidate[name]).abs().max().item()
            for name in reference
        },
    }
    (args.out / "rope_exact.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
