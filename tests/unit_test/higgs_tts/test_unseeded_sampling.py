# SPDX-License-Identifier: Apache-2.0
"""CUDA contracts for the optional Higgs exponential-race draw."""

import pytest
import torch

from sglang_omni.models.higgs_tts import sampler

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Higgs sampler requires CUDA"
    ),
]


def test_disabled_draw_preserves_multinomial_rng_and_output(monkeypatch):
    monkeypatch.setattr(sampler, "USE_GUMBEL_SAMPLE", False)
    torch.cuda.manual_seed(38117)
    probabilities = torch.rand(32, 1026, device="cuda").softmax(-1)
    rng = torch.cuda.get_rng_state()
    actual = sampler._draw_unseeded_probs(probabilities)
    state_after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng)
    expected = probabilities.multinomial(1).squeeze(-1)
    assert torch.equal(actual, expected)
    assert torch.equal(state_after, torch.cuda.get_rng_state())


def test_gumbel_draw_preserves_input_and_sparse_distribution(monkeypatch):
    monkeypatch.setattr(sampler, "USE_GUMBEL_SAMPLE", True)
    torch.cuda.manual_seed(38117)
    probabilities = torch.tensor([0.0, 0.9, 0.1, 0.0], device="cuda")
    probabilities = probabilities.expand(20_000, -1).contiguous()
    original = probabilities.clone()
    selected = sampler._draw_unseeded_probs(probabilities)
    assert torch.equal(probabilities, original)
    assert selected.dtype == torch.int64 and selected.shape == (20_000,)
    assert bool(((selected == 1) | (selected == 2)).all())
    assert 0.88 < float((selected == 1).float().mean()) < 0.92


def test_mixed_batch_seeded_and_greedy_outputs_survive_switch(monkeypatch):
    torch.cuda.manual_seed(38117)
    logits = torch.randn(5, 8, 1026, device="cuda")
    kwargs = {
        "temperature": torch.tensor([0.0, 0.8, 0.8, 0.8, 0.8], device="cuda"),
        "top_p": torch.full((5,), 0.9, device="cuda"),
        "top_k_buf": torch.tensor([50, 1, 50, 50, 50], device="cuda"),
        "seeds_B": torch.tensor([-1, -1, 101, 202, -1], device="cuda"),
        "step_B": torch.full((5,), 9, device="cuda"),
    }
    monkeypatch.setattr(sampler, "USE_GUMBEL_SAMPLE", False)
    baseline = sampler._sample_independent_batched(logits, **kwargs)
    monkeypatch.setattr(sampler, "USE_GUMBEL_SAMPLE", True)
    candidate = sampler._sample_independent_batched(logits, **kwargs)
    assert torch.equal(baseline[:4], candidate[:4])
    assert torch.equal(candidate[:2], logits[:2].argmax(-1))
