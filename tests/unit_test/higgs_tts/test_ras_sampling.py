# SPDX-License-Identifier: Apache-2.0
"""Repetition-aware sampling (RAS) contracts of the Higgs batched sampler.

``multinomial_with_seed`` and the fused renorm kernels need CUDA, so these
run on CUDA only.
"""

from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.sampler import multinomial_with_seed

from sglang_omni.models.higgs_tts import sampler as higgs_sampler
from sglang_omni.models.higgs_tts.sampler import (
    NO_SEED,
    RAS_MAX_REPEAT,
    RAS_WIN_LEN,
    HiggsBatchedSamplerState,
    _sample_independent_batched,
    batched_step,
)
from sglang_omni.models.higgs_tts.utils import BOC_ID, EOC_ID

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused sampling kernels need CUDA"
)

N = 8
V = 1026


def _no_seeds(B: int) -> torch.Tensor:
    return torch.full((B,), NO_SEED, dtype=torch.long, device="cuda")


def _steps(B: int, step: int = 0) -> torch.Tensor:
    return torch.full((B,), step, dtype=torch.long, device="cuda")


def test_ras_redraw_restores_raw_support():
    """The redraw must come from the RAW distribution: a token that top_p
    excludes from the primary draw becomes reachable once the repetition
    trigger fires."""
    B = 512
    top_tok, excluded_tok = 5, 9
    logits = torch.full((B, N, V), -12.0, device="cuda")
    logits[..., top_tok] = 8.0
    logits[..., excluded_tok] = 7.9  # ~47% raw mass, outside the nucleus below
    recent = torch.full((B, N, RAS_WIN_LEN), -1, dtype=torch.long, device="cuda")
    recent[:, :, -RAS_MAX_REPEAT:] = top_tok

    out = _sample_independent_batched(
        logits,
        temperature=torch.ones(B, device="cuda"),
        top_p=torch.full((B,), 0.01, device="cuda"),  # nucleus keeps only top_tok
        seeds_B=_no_seeds(B),
        step_B=_steps(B),
        recent_codes=recent,
    )

    frac_excluded = (out == excluded_tok).float().mean().item()
    assert 0.25 < frac_excluded < 0.65
    frac_other = (~((out == top_tok) | (out == excluded_tok))).float().mean().item()
    assert frac_other < 0.01


def test_ras_not_triggered_below_max_repeat():
    """One occurrence in the window (below RAS_MAX_REPEAT) must not redraw:
    the nucleus-excluded token stays unreachable."""
    B = 512
    top_tok, excluded_tok = 5, 9
    logits = torch.full((B, N, V), -12.0, device="cuda")
    logits[..., top_tok] = 8.0
    logits[..., excluded_tok] = 7.9
    recent = torch.full((B, N, RAS_WIN_LEN), -1, dtype=torch.long, device="cuda")
    recent[:, :, -1] = top_tok  # a single occurrence

    out = _sample_independent_batched(
        logits,
        temperature=torch.ones(B, device="cuda"),
        top_p=torch.full((B,), 0.01, device="cuda"),
        seeds_B=_no_seeds(B),
        step_B=_steps(B),
        recent_codes=recent,
    )

    assert bool((out == top_tok).all())


def test_ras_seeded_redraw_follows_salted_hash_formula():
    """Seeded rows stay deterministic: the output equals the primary hash
    draw where no repetition triggered and the salted-position raw redraw
    where it did, both recomputed here with the sampler's own formulas."""
    B = 2
    top_tok, alt_tok = 7, 13
    logits = torch.full((B, N, V), -6.0, device="cuda")
    logits[..., top_tok] = 2.0
    logits[..., alt_tok] = 1.9  # near-tie: hash draws vary by position
    recent = torch.full((B, N, RAS_WIN_LEN), -1, dtype=torch.long, device="cuda")
    recent[0, :, -RAS_MAX_REPEAT:] = top_tok  # row 0 can trigger, row 1 cannot

    seeds = torch.tensor([31337, 424242], device="cuda")
    step = _steps(B, step=9)

    out = _sample_independent_batched(
        logits,
        temperature=torch.ones(B, device="cuda"),
        top_p=None,
        seeds_B=seeds,
        step_B=step,
        recent_codes=recent,
    )

    cb = torch.arange(N, device="cuda").view(1, N).expand(B, N)
    positions = (step.view(B, 1) * N + cb).reshape(B * N)
    seeds_flat = seeds.view(B, 1).expand(B, N).reshape(B * N)
    probs = logits.float().softmax(dim=-1).reshape(B * N, V).contiguous()
    primary = (
        multinomial_with_seed(torch.log(probs), seeds_flat, positions)
        .squeeze(-1)
        .view(B, N)
    )
    redraw = (
        multinomial_with_seed(
            torch.log(probs),
            seeds_flat,
            positions + higgs_sampler._RAS_POSITION_OFFSET,
        )
        .squeeze(-1)
        .view(B, N)
    )
    trigger = (recent == primary.unsqueeze(-1)).sum(dim=-1) >= RAS_MAX_REPEAT
    expected = torch.where(trigger, redraw, primary)

    assert torch.equal(out, expected)
    assert not bool(trigger[1].any())
    # Repeat call: fully deterministic.
    again = _sample_independent_batched(
        logits,
        temperature=torch.ones(B, device="cuda"),
        top_p=None,
        seeds_B=seeds,
        step_B=step,
        recent_codes=recent,
    )
    assert torch.equal(out, again)


def test_ras_greedy_rows_exempt():
    """Explicit greedy decode (top_k=1) stays argmax even with a saturated
    repetition window: no RNG enters the greedy contract."""
    B = 64
    top_tok, alt_tok = 11, 17
    logits = torch.full((B, N, V), -12.0, device="cuda")
    logits[..., top_tok] = 8.0
    logits[..., alt_tok] = 7.9  # a redraw would frequently pick this
    recent = torch.full((B, N, RAS_WIN_LEN), top_tok, dtype=torch.long, device="cuda")

    out = _sample_independent_batched(
        logits,
        temperature=torch.ones(B, device="cuda"),
        top_p=None,
        top_k_buf=torch.ones(B, dtype=torch.long, device="cuda"),
        seeds_B=_no_seeds(B),
        step_B=_steps(B),
        recent_codes=recent,
    )

    assert bool((out == top_tok).all())


def test_batched_step_maintains_recent_window_and_reset():
    """The state machine commits the final (post delay-mask) codes into the
    rolling window, and reset_row wipes it back to -1."""
    pool = HiggsBatchedSamplerState(2, N, device="cuda")
    row_indices = torch.arange(2, device="cuda")
    temp = torch.ones(2, device="cuda")

    targets = [10, 20, 30]  # distinct per step, so RAS never triggers
    for t in targets:
        logits = torch.full((2, N, V), -12.0, device="cuda")
        logits[..., t] = 12.0
        batched_step(logits, pool, row_indices, temperature=temp)

    hist = pool.recent_codes
    assert hist.shape == (2, N, RAS_WIN_LEN)
    assert torch.equal(
        hist[0, 0, -3:], torch.tensor(targets, dtype=torch.long, device="cuda")
    )
    # Codebook 4 stays BOC-masked for the first three delay steps.
    assert torch.equal(
        hist[0, 4, -3:],
        torch.full((3,), BOC_ID, dtype=torch.long, device="cuda"),
    )
    assert bool((hist[0, 0, :-3] == -1).all())

    pool.reset_row(0)
    assert bool((pool.recent_codes[0] == -1).all())
    assert bool((pool.recent_codes[1] != -1).any())


def test_ras_does_not_block_eoc_winddown():
    """A saturated repetition window must not disturb EOC detection or the
    wind-down: EOC is never repetition-blocked because it cannot already
    appear in the window."""
    pool = HiggsBatchedSamplerState(1, N, device="cuda")
    rows = torch.zeros(1, dtype=torch.long, device="cuda")
    temp = torch.ones(1, device="cuda")
    same = 42

    logits_same = torch.full((1, N, V), -12.0, device="cuda")
    logits_same[..., same] = 12.0
    for _ in range(N + RAS_WIN_LEN):
        batched_step(logits_same, pool, rows, temperature=temp)
    assert not bool(pool.generation_done[0].item())

    logits_eoc = logits_same.clone()
    logits_eoc[:, 0, :] = -12.0
    logits_eoc[:, 0, EOC_ID] = 12.0
    batched_step(logits_eoc, pool, rows, temperature=temp)
    assert int(pool.eoc_countdown[0].item()) == N - 2

    for _ in range(N - 2):
        batched_step(logits_same, pool, rows, temperature=temp)
    assert bool(pool.generation_done[0].item())
