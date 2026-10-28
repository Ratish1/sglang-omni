# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.kernels.tf32x3_linear import Tf32x3Linear, tf32x3_linear
from sglang_omni.models.qwen3_tts.stages import use_tf32x3_transformer_linears

FLOAT32_RELATIVE_ERROR = 1e-6


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        ((actual.double() - expected).abs().max() / expected.abs().max()).item()
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "out_features, in_features", [(512, 512), (2048, 512), (512, 2048)]
)
@pytest.mark.parametrize("rows", [64, 70, 512])
def test_tf32x3_linear_is_the_float32_product(
    out_features: int, in_features: int, rows: int
) -> None:
    torch.manual_seed(0)
    hidden_states = torch.randn(2, rows, in_features, device="cuda")
    weight = torch.randn(out_features, in_features, device="cuda") / in_features**0.5

    actual = tf32x3_linear(hidden_states, weight)

    assert actual.shape == (2, rows, out_features)
    assert (
        relative_error(actual, hidden_states.double() @ weight.double().T)
        < FLOAT32_RELATIVE_ERROR
    )


def make_encoder(device: str) -> torch.nn.Module:
    encoder = torch.nn.Module()
    encoder.encoder_transformer = torch.nn.Sequential(
        torch.nn.Linear(512, 2048, bias=False), torch.nn.Linear(2048, 512, bias=False)
    ).to(device)
    return encoder


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires TF32 tensor cores",
)
def test_encoder_transformer_projections_run_the_tf32x3_kernel() -> None:
    torch.manual_seed(0)
    encoder = make_encoder("cuda")
    weights = [layer.weight for layer in encoder.encoder_transformer]
    hidden_states = torch.randn(1, 128, 512, device="cuda")

    replaced = use_tf32x3_transformer_linears(encoder)

    assert replaced == 2
    assert all(isinstance(layer, Tf32x3Linear) for layer in encoder.encoder_transformer)
    with torch.no_grad():
        assert torch.equal(
            encoder.encoder_transformer(hidden_states),
            tf32x3_linear(tf32x3_linear(hidden_states, weights[0]), weights[1]),
        )


def test_encoder_transformer_projections_stay_float32_linears_off_cuda() -> None:
    encoder = make_encoder("cpu")

    assert use_tf32x3_transformer_linears(encoder) == 0
    assert all(type(layer) is torch.nn.Linear for layer in encoder.encoder_transformer)
