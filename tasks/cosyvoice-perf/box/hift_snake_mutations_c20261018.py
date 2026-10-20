"""The tree's Snake equality test against the real kernel and two broken ones: one that ignores both alphas, one
whose last product fuses into an FMA. A test that protects Snake passes the first and fails the other two.
usage: PYTHONPATH=<tree> python3 hift_snake_mutations_c20261018.py"""

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.diffusion.common.numerics import mul_rn_f32
from triton.language.extra import libdevice

from sglang_omni.models.fun_cosyvoice3 import hift_kernels as kernels
from sglang_omni.models.fun_cosyvoice3 import stages
from tests.unit_test.fun_cosyvoice3.test_vocoder import test_vocoder_hift_snake_is_bit_identical_to_cosyvoice


@triton.jit(do_not_specialize=["samples", "channels"])
def fused_snake_kernel(x_ptr, alpha_ptr, inverse_alpha_ptr, out_ptr, samples, channels,
                       BLOCK_SAMPLES: tl.constexpr):
    channel = tl.program_id(1)
    row = tl.program_id(2).to(tl.int64) * channels + channel
    sample_indices = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
    is_sample = sample_indices < samples
    alpha = tl.load(alpha_ptr + channel)
    inverse_alpha = tl.load(inverse_alpha_ptr + channel)
    x = tl.load(x_ptr + row * samples + sample_indices, mask=is_sample)
    sine = libdevice.sin(mul_rn_f32(x, alpha))
    out = x + inverse_alpha * mul_rn_f32(sine, sine)
    tl.store(out_ptr + row * samples + sample_indices, out, mask=is_sample)


def fused_snake(x, alpha, inverse_alpha):
    batch, channels, samples = x.shape
    out = torch.empty_like(x)
    grid = (triton.cdiv(samples, kernels.SAMPLES_PER_PROGRAM), channels, batch)
    compiled = fused_snake_kernel[grid](x, alpha, inverse_alpha, out, samples, channels,
                                        BLOCK_SAMPLES=kernels.SAMPLES_PER_PROGRAM)
    fused_snake.fma_count = compiled.asm["ptx"].count("fma.rn.f32")
    return out


def ignore_alpha_snake(x, alpha, inverse_alpha):
    return kernels.snake(x, torch.ones_like(alpha), torch.ones_like(inverse_alpha))


for name, snake in (("real", kernels.snake), ("ignore_alpha", ignore_alpha_snake), ("fused_fma", fused_snake)):
    stages.snake = snake
    try:
        test_vocoder_hift_snake_is_bit_identical_to_cosyvoice()
        verdict = "passes"
    except AssertionError:
        verdict = "fails"
    print(f"{name}: test {verdict}")
print(f"fused kernel fma.rn.f32 count {fused_snake.fma_count}")
