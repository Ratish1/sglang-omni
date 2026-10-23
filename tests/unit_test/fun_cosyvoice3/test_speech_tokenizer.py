# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import bisect
import gc
import weakref

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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_replays_match_the_tf32_model_at_their_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda")
    model = small_model().to(device)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    graphs = SpeechTokenizerCudaGraphs(model, device)
    assert not torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True

    for frames in (300, 290, 37):
        mel = torch.randn(1, 128, frames)
        width = graphs.widths[bisect.bisect_left(graphs.widths, frames)]
        padded = torch.cat((mel, torch.zeros(1, 128, width - frames)), dim=2)
        with torch.inference_mode():
            expected = model(padded.to(device), torch.tensor([frames], device=device))

        assert (
            graphs.tokenize(mel)
            == expected[0, : conv_frames(conv_frames(frames))].tolist()
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graphs_keep_the_model_their_replays_read() -> None:
    """The graphs keep the model whose weights their replays read for as long as
    they live, with no other reference to it."""
    device = torch.device("cuda")
    model = small_model().to(device)
    model_reference = weakref.ref(model)
    graphs = SpeechTokenizerCudaGraphs(model, device)
    del model
    gc.collect()

    assert model_reference() is not None
    del graphs
    gc.collect()
    assert model_reference() is None
