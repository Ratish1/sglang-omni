# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import SnakeBeta

from sglang_omni.kernels.bias_snake_beta import fused_bias_snake_beta
from sglang_omni.utils.snake_beta import FusedSnakeBeta


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("with_residual", "with_sum"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_bias_snake_matches_the_eager_adds_and_snake(
    with_residual: bool, with_sum: bool
) -> None:
    torch.manual_seed(0)
    original = SnakeBeta(96).to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        original.alpha.normal_(0.0, 0.5)
        original.beta.normal_(0.0, 0.5)
    activation = FusedSnakeBeta(original)
    conv_buffer = torch.randn(3, 41, 96, device="cuda", dtype=torch.bfloat16)
    conv_output = conv_buffer[:, 2:39]
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(3, 37, 96, device="cuda", dtype=torch.bfloat16)
    padded_input = torch.zeros(3, 46, 96, device="cuda", dtype=torch.bfloat16)
    sum_out = torch.zeros(3, 37, 96, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        fused_bias_snake_beta(
            conv_output,
            bias,
            activation,
            padded_input[:, 6:43],
            residual=residual if with_residual else None,
            sum_out=sum_out if with_sum else None,
        )
        expected_sum = (
            conv_output + bias + residual if with_residual else conv_output + bias
        )
        expected = original(expected_sum.transpose(1, 2)).transpose(1, 2)

    assert torch.equal(padded_input[:, 6:43], expected)
    assert not padded_input[:, :6].any() and not padded_input[:, 43:].any()
    assert torch.equal(sum_out, expected_sum) if with_sum else not sum_out.any()
