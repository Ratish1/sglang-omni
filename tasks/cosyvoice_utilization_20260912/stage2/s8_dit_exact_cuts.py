"""Launch cuts in the packed DiT's attention that must not change a bit.

Loads the real Flow as the vocoder factory does (DiT weights cast once), runs
hops and finals through the serving entry points, then swaps PackedDiT.rope and
PackedDiT.attend for the variants below and runs the same inputs again. Reports
per shape and per variant: wall time, kernels launched, kernel time, and whether
the mel is bit identical to main's.

  shared_cast  the norm output is cast to the weight dtype once, not once in
               each of to_q, to_k and to_v
  rope         cos and sin are taken once per Euler step instead of once per
               block for q and again for k, and only the rotary dims are
               rotated, in place; main's apply_rotary_pos_emb concatenates the
               whole float32 tensor and casts all of it back
  both         the two together

  python s8_dit_exact_cuts.py --model .../snapshots/master
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.fun_cosyvoice3 import packed_dit
from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT, PackedRows
from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    FlowBatchInput,
    load_cosyvoice3_flow_hift,
    patch_chunk_mask,
)

WARMUP = 3
ITERATIONS = 10

# rows, prompt tokens, generated tokens so far (the hop reads all of them)
SHAPES: tuple[tuple[int, int, int], ...] = (
    (1, 100, 28),
    (4, 150, 128),
    (8, 150, 300),
    (16, 150, 128),
)


def items(rows: int, prompt: int, tokens: int, flow) -> list[FlowBatchInput]:
    generator = torch.Generator().manual_seed(rows * 1000 + tokens)
    return [
        FlowBatchInput(
            token=torch.randint(
                0, 6000, (1, tokens + 8 * row), generator=generator, dtype=torch.int32
            ),
            prompt_token=torch.randint(
                0, 6000, (1, prompt), generator=generator, dtype=torch.int32
            ),
            prompt_feat=torch.randn(
                1, prompt * 2, flow.output_size, generator=generator
            ),
            embedding=torch.randn(
                1, flow.spk_embed_affine_layer.in_features, generator=generator
            ),
        )
        for row in range(rows)
    ]


def rope_cos_sin(self: PackedDiT, rows: PackedRows) -> tuple[torch.Tensor, Any]:
    freqs, scale = self.dit.rotary_embed.forward_from_seq_len(rows.width)
    assert not isinstance(scale, torch.Tensor), "the DiT's RoPE has no xpos scale"
    freqs = freqs[:, rows.positions]
    return (freqs.cos(), freqs.sin()), scale


def rotate(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """t: (1, total, heads * head_dim); only its first cos.shape[-1] dims turn,
    as in apply_rotary_pos_emb. The float32 result rounds into t's dtype."""
    turned = t[..., : cos.shape[-1]]
    half = torch.stack((-turned[..., 1::2], turned[..., ::2]), dim=-1).flatten(-2)
    turned.copy_(turned * cos + half * sin)


def make_attend(*, shared_cast: bool, rope: bool):
    from x_transformers.x_transformers import apply_rotary_pos_emb

    def attend(attn, x, rope_state, attention):
        if shared_cast:
            x = x.to(attn.to_q.weight.dtype)
        query, key, value = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        if rope:
            (cos, sin), _ = rope_state
            rotate(query, cos, sin)
            rotate(key, cos, sin)
        else:
            freqs, scale = rope_state
            query = apply_rotary_pos_emb(query, freqs, scale)
            key = apply_rotary_pos_emb(key, freqs, scale**-1.0)
        out = attention(query, key, value).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](out))

    return staticmethod(attend)


VARIANTS = {
    "main": None,
    "shared_cast": {"shared_cast": True, "rope": False},
    "rope": {"shared_cast": False, "rope": True},
    "both": {"shared_cast": True, "rope": True},
}


def measure(call) -> dict[str, float | int]:
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    walls = []
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - started) * 1000)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        call()
        torch.cuda.synchronize()
    kernels = [
        event
        for event in trace.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    return {
        "wall_ms_median": round(statistics.median(walls), 2),
        "wall_ms_min": round(min(walls), 2),
        "kernels": len(kernels),
        "kernel_ms": round(sum(event.device_time for event in kernels) / 1000, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    flow, hift = load_cosyvoice3_flow_hift(args.model, device=args.device)
    patch_chunk_mask()
    for module in flow.decoder.estimator.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
    vocoder = CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    inputs = {shape: items(*shape, flow) for shape in SHAPES}
    main_rope, main_attend = PackedDiT.rope, PackedDiT.__dict__["attend"]

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "packed_dit": packed_dit.__file__,
        "rows": [],
    }
    truth: dict[str, list[torch.Tensor]] = {}
    for variant, switches in VARIANTS.items():
        if switches is None:
            PackedDiT.rope, PackedDiT.attend = main_rope, main_attend
        else:
            PackedDiT.rope = rope_cos_sin if switches["rope"] else main_rope
            PackedDiT.attend = make_attend(**switches)
        for shape, batch in inputs.items():
            for kind, call in (
                ("hop", lambda batch=batch: vocoder.hop_batch(batch)),
                ("final", lambda batch=batch: vocoder.leftover_batch(batch)),
            ):
                name = f"{kind} {shape}"
                result = measure(call)
                mel = [row.clone() for row in call()]
                repeatable = all(torch.equal(a, b) for a, b in zip(mel, call()))
                truth.setdefault(name, mel)
                report["rows"].append(
                    {
                        "call": name,
                        "variant": variant,
                        "frames": sum(int(row.shape[-1]) for row in mel),
                        **result,
                        "repeatable": repeatable,
                        "bit_identical_to_main": all(
                            torch.equal(a, b) for a, b in zip(truth[name], mel)
                        ),
                        "max_abs_diff": max(
                            float((a - b).abs().max()) for a, b in zip(truth[name], mel)
                        ),
                    }
                )
                print(json.dumps(report["rows"][-1]), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
