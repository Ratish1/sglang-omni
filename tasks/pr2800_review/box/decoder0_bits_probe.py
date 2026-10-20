"""Which step of decoder.0 and its SnakeBeta changes bits on the real checkpoint: the conv's bias, or the activation.

Run on the box from a tree: cd <tree> && PYTHONPATH=<tree> python3 /workspace/box/tasks/pr2800_review/box/decoder0_bits_probe.py
The served bf16 code2wav (load_code2wav_model, fused SnakeBeta) on card 0. For a random (B, 40 + history, 1024) padded
input at 1 and 8 rows: decoder.0 through channels_last_conv1d with its bias, against the conv without its bias plus
the bias added in bf16; then the first decoder block's SnakeBeta of the biased output through the module, against
fused_bias_snake_beta of the unbiased output. Prints whether each pair is equal and the largest difference.
"""

from __future__ import annotations

import torch

from sglang_omni.kernels.bias_snake_beta import fused_bias_snake_beta
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import load_code2wav_model
from sglang_omni.utils.channels_last_conv import channels_last_conv1d
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def compare(name: str, first: torch.Tensor, second: torch.Tensor) -> None:
    difference = (first.float() - second.float()).abs()
    print(f"  {name}: equal {torch.equal(first, second)}, differing elements {int((difference > 0).sum())} of "
          f"{difference.numel()}, max difference {float(difference.max()):.3e}")


def main() -> None:
    model = load_code2wav_model(MODEL, device="cuda:0", dtype="bfloat16")
    fuse_vocoder_decoder(model.decoder)
    module = model.decoder[0]
    activation = model.decoder[1].block[0]
    conv = module.conv
    torch.manual_seed(0)
    for rows in (1, 8):
        padded = torch.randn(rows, module.padding + 40, conv.in_channels, device="cuda:0", dtype=torch.bfloat16)
        with torch.inference_mode():
            biased = channels_last_conv1d(padded, conv, conv.weight, 40)
            unbiased = channels_last_conv1d(padded, conv, conv.weight, 40, add_bias=False)
            print(f"rows {rows}: decoder.0 {conv.in_channels} to {conv.out_channels}, kernel {conv.kernel_size[0]}")
            compare("conv with bias against conv then + bias in bf16", biased, unbiased + conv.bias)
            through_module = activation(biased.transpose(1, 2)).transpose(1, 2)
            fused = torch.empty_like(unbiased)
            fused_bias_snake_beta(unbiased, conv.bias, activation, fused)
            compare("module SnakeBeta of the biased conv against the fused bias SnakeBeta", through_module, fused)
            eager = torch.ops.aten.add(
                biased,
                (1.0 / (torch.exp(activation.beta) + activation.no_div_by_zero))
                * torch.pow(torch.sin(biased * torch.exp(activation.alpha)), 2),
            )
            compare("module SnakeBeta against eager SnakeBeta, both of the biased conv", through_module, eager)


if __name__ == "__main__":
    main()
