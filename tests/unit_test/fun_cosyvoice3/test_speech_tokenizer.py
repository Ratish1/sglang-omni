# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3.speech_tokenizer import (
    SpeechTokenizerCudaGraphs,
    SpeechTokenizerV3Model,
    conv_frames,
)


def small_model() -> SpeechTokenizerV3Model:
    torch.manual_seed(0)
    return SpeechTokenizerV3Model(
        layer_count=2, mel_channels=128, state_channels=64, head_count=4, fsmn_kernel=5
    )


def test_codes_of_a_mel_ignore_what_fills_its_padded_width() -> None:
    model = small_model()
    frames = 37
    mel = torch.randn(1, 128, frames)
    padded = torch.cat((mel, 100 * torch.randn(1, 128, 27)), dim=2)

    with torch.inference_mode():
        codes = model(mel, torch.tensor([frames]))
        padded_codes = model(padded, torch.tensor([frames]))

    assert torch.equal(padded_codes[:, : conv_frames(conv_frames(frames))], codes)


@pytest.mark.accelerator
def test_graph_replays_match_the_eager_model_at_the_mel_width() -> None:
    device = torch.device("cuda")
    model = small_model().to(device)
    graphs = SpeechTokenizerCudaGraphs(model, device)

    for frames in (300, 290, 37):
        mel = torch.randn(1, 128, frames)
        with torch.inference_mode():
            expected = model(mel.to(device), torch.tensor([frames], device=device))

        assert graphs.tokenize(mel) == expected[0].tolist()
