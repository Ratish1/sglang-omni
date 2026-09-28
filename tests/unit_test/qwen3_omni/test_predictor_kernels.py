# SPDX-License-Identifier: Apache-2.0
"""The predictor's one-launch add and RMSNorm against the plain two-launch path."""

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.predictor_kernels import (
    HIDDEN_SIZE,
    add_rmsnorm_rounded,
    supports_exact_add_rmsnorm,
)
from sglang_omni.platforms import current_platform


@pytest.mark.accelerator
@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="the kernel matches the CUDA RMSNorm",
)
@pytest.mark.parametrize("rows", [1, 3, 12, 32])
def test_add_rmsnorm_rounded_matches_add_then_rmsnorm_bit_for_bit(rows: int) -> None:
    """Same bits as torch's bf16 add followed by the plain RMSNorm, over many seeds."""
    from sgl_kernel import rmsnorm

    device = torch.device("cuda")
    for seed in range(50):
        torch.manual_seed(seed)
        x = (torch.randn(rows, HIDDEN_SIZE, device=device) * 3).to(torch.bfloat16)
        residual = (torch.randn(rows, HIDDEN_SIZE, device=device) * 3).to(
            torch.bfloat16
        )
        weight = (1 + 0.1 * torch.randn(HIDDEN_SIZE, device=device)).to(torch.bfloat16)
        expected_sum = residual + x
        expected = rmsnorm(expected_sum, weight, 1e-6)
        normed, summed = add_rmsnorm_rounded(x.clone(), residual.clone(), weight, 1e-6)
        assert torch.equal(summed, expected_sum)
        assert torch.equal(normed, expected)


def test_exact_add_rmsnorm_applies_to_the_predictor_shape_only() -> None:
    cuda = torch.device("cuda")
    assert supports_exact_add_rmsnorm(HIDDEN_SIZE, torch.bfloat16, cuda)
    assert not supports_exact_add_rmsnorm(2048, torch.bfloat16, cuda)
    assert not supports_exact_add_rmsnorm(HIDDEN_SIZE, torch.float16, cuda)
    assert not supports_exact_add_rmsnorm(
        HIDDEN_SIZE, torch.bfloat16, torch.device("cpu")
    )
