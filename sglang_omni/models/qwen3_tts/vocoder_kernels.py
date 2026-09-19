# SPDX-License-Identifier: Apache-2.0
"""Fused snake activation for the Qwen3-TTS vocoder decoder.

Snake is y = x + r * sin(a * x)^2 with one a and one r per channel. The qwen-tts
SnakeBeta derives a = exp(alpha) and r = 1 / (exp(beta) + eps) and evaluates the
whole expression as separate elementwise kernels, 29 times per decode, on the
largest tensors of the decoder. On a bf16 tensor every one of those kernels
computes in fp32 and rounds its result to bf16.

The kernel here fuses the five x-sized steps (mul, sin, square, mul, add) into
one pass and rounds to bf16 after each of them, so it is bitwise identical to
the eager chain. a and r depend on the weights only: they are built once per
module with the eager expression itself, on the module's device and dtype.
"""

from __future__ import annotations

import logging

import torch

try:  # keep the module importable when Triton is unavailable
    import triton
    import triton.language as tl
    from sglang.kernels.ops.diffusion.common.numerics import round_bf16_to_fp32
    from triton.language.extra import libdevice

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on runtime image
    _HAS_TRITON = False

from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

SNAKE_BLOCK_SIZE = 1024
SNAKE_NUM_WARPS = 4
SNAKE_SELF_CHECK_FRAMES = 257

if _HAS_TRITON:

    # note(ratish): sizes and pointers stay unspecialized so one binary serves
    # every shape and alignment; a second variant would compile at serving time.
    @triton.jit(
        do_not_specialize=[
            "out_ptr",
            "x_ptr",
            "a_ptr",
            "r_ptr",
            "numel",
            "channels",
            "length",
        ]
    )
    def _snake_kernel(
        out_ptr,
        x_ptr,
        a_ptr,
        r_ptr,
        numel,
        channels,
        length,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        channel = (offs // length) % channels
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        scaled = round_bf16_to_fp32(x * a)
        sine = round_bf16_to_fp32(libdevice.sin(scaled))
        squared = round_bf16_to_fp32(sine * sine)
        periodic = round_bf16_to_fp32(r * squared)
        tl.store(out_ptr + offs, x + periodic, mask=mask)  # store rounds the add


def can_use_fused_snake(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor) -> bool:
    return (
        _HAS_TRITON
        and x.dtype is torch.bfloat16
        and a.dtype is torch.bfloat16
        and r.dtype is torch.bfloat16
        and x.is_cuda
        and a.device == x.device
        and r.device == x.device
        and x.dim() == 3
        and x.is_contiguous()
        and a.shape == (x.shape[1],)
        and r.shape == (x.shape[1],)
        and 0 < x.numel() < 2**31
    )


def _fake_snake(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@register_custom_op(
    op_name="qwen3_tts_fused_snake_bitexact",
    mutates_args=[],
    fake_impl=_fake_snake,
)
def fused_snake(x: torch.Tensor, a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """x + r * sin(a * x)^2 over [B, C, T], bit-exact vs the eager bf16 chain."""
    out = torch.empty_like(x)
    numel = x.numel()
    with torch.cuda.device(x.device):
        # note(ratish): libdevice sin must keep denormals like the eager kernel
        _snake_kernel[(triton.cdiv(numel, SNAKE_BLOCK_SIZE),)](
            out,
            x,
            a,
            r,
            numel,
            x.shape[1],
            x.shape[2],
            BLOCK=SNAKE_BLOCK_SIZE,
            num_warps=SNAKE_NUM_WARPS,
            enable_reflect_ftz=False,
        )
    return out


class FusedSnakeBeta(torch.nn.Module):
    """qwen-tts SnakeBeta with the fused kernel; the original runs any other input.

    a and r are fixed at construction, so the weights must be final by then.
    """

    def __init__(self, original: torch.nn.Module) -> None:
        super().__init__()
        self.original = original
        with torch.inference_mode():
            alpha = torch.exp(original.alpha.unsqueeze(0).unsqueeze(-1))
            beta = torch.exp(original.beta.unsqueeze(0).unsqueeze(-1))
            scale = 1.0 / (beta + original.no_div_by_zero)
        self.register_buffer("a", alpha.reshape(-1).contiguous(), persistent=False)
        self.register_buffer("r", scale.reshape(-1).contiguous(), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if can_use_fused_snake(hidden_states, self.a, self.r):
            return fused_snake(hidden_states, self.a, self.r)
        return self.original(hidden_states)


def fuse_vocoder_decoder(decoder: torch.nn.Module, snake_cls: type) -> int:
    """Replace every snake_cls module the kernel reproduces bit for bit.

    Each candidate is run fused and eager on one random tensor before it is
    installed, outside any graph capture; a mismatch keeps the eager module.
    Returns the number of modules replaced.
    """
    replaced = 0
    for parent in list(decoder.modules()):
        if isinstance(parent, FusedSnakeBeta):
            continue
        for name, child in list(parent.named_children()):
            if not isinstance(child, snake_cls):
                continue
            fused = FusedSnakeBeta(child)
            probe = torch.randn(
                (2, fused.a.shape[0], SNAKE_SELF_CHECK_FRAMES),
                dtype=fused.a.dtype,
                device=fused.a.device,
            )
            if not can_use_fused_snake(probe, fused.a, fused.r):
                continue
            with torch.inference_mode():
                identical = torch.equal(fused(probe), child(probe))
            if not identical:
                logger.warning(
                    "Qwen3-TTS fused snake differs from eager on %s; keeping eager",
                    name,
                )
                continue
            setattr(parent, name, fused)
            replaced += 1
    return replaced


__all__ = [
    "FusedSnakeBeta",
    "can_use_fused_snake",
    "fuse_vocoder_decoder",
    "fused_snake",
]
