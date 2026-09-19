# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import sglang_omni.models.qwen3_tts.vocoder_kernels as vocoder_kernels


class _StubSnakeBeta(torch.nn.Module):
    """Stand-in with the qwen-tts SnakeBeta attribute layout and arithmetic."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.in_features = channels
        self.alpha = torch.nn.Parameter(torch.randn(channels) * 0.1)
        self.beta = torch.nn.Parameter(torch.randn(channels) * 0.1)
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )


def test_fuse_vocoder_decoder_leaves_a_cpu_decoder_untouched() -> None:
    torch.manual_seed(0)
    first = _StubSnakeBeta(4)
    second = _StubSnakeBeta(4)
    decoder = torch.nn.Sequential(first, torch.nn.Sequential(second))

    assert vocoder_kernels.fuse_vocoder_decoder(decoder, _StubSnakeBeta) == 0
    assert decoder[0] is first
    assert decoder[1][0] is second


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused snake needs CUDA")
def test_fuse_vocoder_decoder_keeps_eager_when_the_kernel_differs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    first = _StubSnakeBeta(96).to(device=device, dtype=torch.bfloat16)
    decoder = torch.nn.Sequential(first)
    monkeypatch.setattr(vocoder_kernels, "fused_snake", lambda x, a, r: x.clone())

    assert vocoder_kernels.fuse_vocoder_decoder(decoder, _StubSnakeBeta) == 0
    assert decoder[0] is first


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused snake needs CUDA")
@pytest.mark.parametrize(
    ("batch", "channels", "frames"),
    [
        (1, 1536, 1),
        (8, 1536, 8),
        (1, 5, 33),
        (9, 192, 96),
        (1, 96, 122880),
    ],
)
def test_fused_snake_beta_is_bitwise_identical_to_eager(
    batch: int, channels: int, frames: int
) -> None:
    assert vocoder_kernels._HAS_TRITON, "Triton is required on accelerator CI"

    torch.manual_seed(0)
    device = torch.device("cuda")
    original = _StubSnakeBeta(channels).to(device=device, dtype=torch.bfloat16)
    decoder = torch.nn.Sequential(original)
    x = torch.randn((batch, channels, frames), device=device, dtype=torch.bfloat16)
    expected = original(x)

    assert vocoder_kernels.fuse_vocoder_decoder(decoder, _StubSnakeBeta) == 1
    assert vocoder_kernels.fuse_vocoder_decoder(decoder, _StubSnakeBeta) == 0
    fused = decoder[0]
    assert isinstance(fused, vocoder_kernels.FusedSnakeBeta)
    assert vocoder_kernels.can_use_fused_snake(x, fused.a, fused.r)
    assert torch.equal(fused(x).view(torch.int16), expected.view(torch.int16))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused snake needs CUDA")
def test_fused_snake_beta_matches_eager_on_every_bf16_encoding() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    original = _StubSnakeBeta(96).to(device=device, dtype=torch.bfloat16)
    fused = vocoder_kernels.FusedSnakeBeta(original)
    encodings = (
        torch.arange(-32768, 32768, dtype=torch.int32, device=device)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    x = encodings.repeat(96).reshape(1, 96, 65536).contiguous()

    assert torch.equal(fused(x).view(torch.int16), original(x).view(torch.int16))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused snake needs CUDA")
def test_fused_snake_beta_runs_the_original_outside_the_kernel_contract() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    original = _StubSnakeBeta(96).to(device=device, dtype=torch.bfloat16)
    fused = vocoder_kernels.FusedSnakeBeta(original)
    strided = torch.randn((2, 33, 96), device=device, dtype=torch.bfloat16).transpose(
        1, 2
    )

    assert not vocoder_kernels.can_use_fused_snake(strided, fused.a, fused.r)
    assert torch.equal(fused(strided), original(strided))
