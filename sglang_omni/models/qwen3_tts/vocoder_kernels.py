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
    from triton.language.extra import libdevice

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on runtime image
    _HAS_TRITON = False

from sglang.srt.utils.custom_op import register_custom_op

logger = logging.getLogger(__name__)

SNAKE_TILE_ELEMENTS = 1024
SNAKE_TILE_COLUMNS = (16, 64, 256, 1024)
SNAKE_NUM_WARPS = 4
# note(ratish): a CUDA launch takes at most 65,535 programs on its second axis
SNAKE_MAX_FRAMES = 65535 * SNAKE_TILE_COLUMNS[-1]

if _HAS_TRITON:

    # note(ratish): sizes and pointers stay unspecialized so one binary per tile
    # shape serves every input; another variant would compile at serving time.
    @triton.jit(
        do_not_specialize=[
            "out_ptr",
            "x_ptr",
            "a_ptr",
            "r_ptr",
            "rows",
            "channels",
            "length",
        ]
    )
    def _snake_kernel(
        out_ptr,
        x_ptr,
        a_ptr,
        r_ptr,
        rows,
        channels,
        length,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
    ):
        # note(ratish): one program is a ROWS x COLS tile of the [B * C, T] rows, so
        # the channel costs one modulo per row and a short T still fills the tile
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        col = tl.program_id(1) * COLS + tl.arange(0, COLS)
        row_mask = row < rows
        if ROWS == 1:
            mask = (col < length)[None, :]
        else:
            mask = row_mask[:, None] & (col < length)[None, :]
        offs = row[:, None] * length + col[None, :]
        channel = row % channels
        a = tl.load(a_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)[:, None]
        r = tl.load(r_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)[:, None]
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # note(ratish): eager writes a bf16 tensor after each of these steps
        scaled = (x * a).to(tl.bfloat16).to(tl.float32)
        sine = libdevice.sin(scaled).to(tl.bfloat16).to(tl.float32)
        squared = (sine * sine).to(tl.bfloat16).to(tl.float32)
        periodic = (r * squared).to(tl.bfloat16).to(tl.float32)
        tl.store(out_ptr + offs, x + periodic, mask=mask)


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
        and x.shape[2] <= SNAKE_MAX_FRAMES
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
    rows = x.shape[0] * x.shape[1]
    length = x.shape[2]
    cols = next((c for c in SNAKE_TILE_COLUMNS if length <= c), SNAKE_TILE_COLUMNS[-1])
    tile_rows = SNAKE_TILE_ELEMENTS // cols
    with torch.cuda.device(x.device):
        # note(ratish): sin keeps denormals and no mul-add is contracted, as in the
        # eager kernels; without either flag the result differs from eager
        _snake_kernel[(triton.cdiv(rows, tile_rows), triton.cdiv(length, cols))](
            out,
            x,
            a,
            r,
            rows,
            x.shape[1],
            length,
            ROWS=tile_rows,
            COLS=cols,
            num_warps=SNAKE_NUM_WARPS,
            enable_reflect_ftz=False,
            enable_fp_fusion=False,
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

    Each candidate is run fused and eager on a random tensor per tile shape
    before it is installed, which also compiles every kernel variant outside
    any graph capture; a mismatch keeps the eager module. Returns the number of
    modules replaced.
    """
    replaced = 0
    for parent in list(decoder.modules()):
        if isinstance(parent, FusedSnakeBeta):
            continue
        for name, child in list(parent.named_children()):
            if not isinstance(child, snake_cls):
                continue
            fused = FusedSnakeBeta(child)
            probes = [
                torch.randn(
                    (2, fused.a.shape[0], columns),
                    dtype=fused.a.dtype,
                    device=fused.a.device,
                )
                for columns in SNAKE_TILE_COLUMNS
            ]
            if not can_use_fused_snake(probes[0], fused.a, fused.r):
                continue
            with torch.inference_mode():
                identical = all(
                    torch.equal(fused(probe), child(probe)) for probe in probes
                )
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
