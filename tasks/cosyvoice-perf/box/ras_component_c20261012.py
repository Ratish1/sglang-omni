# SPDX-License-Identifier: Apache-2.0
"""Component test of #2626's repetition aware redraw (sglang_omni/sampling/repetition_aware.py), on the GPU,
with the real function and SGLang's own pytorch sampler, no serving.

1. Distribution: one probability row p (temperature applied), its first draw x from SGLang's top k 20 /
   top p 0.8 sampler (distribution q), then the redraw. The emitted token's distribution must be
   P(t) = q(t) [t not in W] + sum over x in W of q(x) p(t) [t != x] / (1 - p(x)), W the 10 token window.
   Measured over 2,000,000 draws as the total variation distance from P, against the distance two samples of
   P itself reach (the noise floor).
2. Seeds: a seeded batch through the redraw twice gives the same ids.
3. Cost per decode step of the hook as the runner calls it (the window built from output_ids lists, the H2D,
   the redraw over every row), at 1, 16 and 32 rows: host time per call and device time by events.

    python3 ras_component_c20261012.py
"""

from __future__ import annotations

import time
from array import array

import torch
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

from sglang_omni.sampling.repetition_aware import repetition_aware_redraw

VOCAB = 6761
SPEECH = 6561
WINDOW = 10
TOP_K, TOP_P, TEMPERATURE = 20, 0.8, 0.7


def first_draw_distribution(probs: torch.Tensor) -> torch.Tensor:
    """q: SGLang's top k then top p mask on the sorted row, renormalized."""
    sorted_probs, order = probs.sort(descending=True)
    cumulative = sorted_probs.cumsum(0)
    sorted_probs[TOP_K:] = 0.0
    sorted_probs[(cumulative - sorted_probs) > TOP_P] = 0.0
    q = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
    return q / q.sum()


def main() -> None:
    torch.manual_seed(0)
    device = "cuda"
    logits = torch.randn(VOCAB, device=device, dtype=torch.float64) * 3.0
    probs = torch.softmax(logits / TEMPERATURE, dim=0)
    q = first_draw_distribution(probs.clone())
    top = q.argsort(descending=True)
    window = top[[0, 2, 5]].tolist() + list(range(100, 107))
    expected = q.clone()
    for token in set(window):
        expected[token] = 0.0
    for token in set(window):
        redraw = probs.clone()
        redraw[token] = 0.0
        expected += q[token] * redraw / (1.0 - probs[token])
    print(f"window holds {q[window].sum().item():.3f} of the first draw's mass", flush=True)

    rows, rounds = 20_000, 100
    counts = torch.zeros(VOCAB, dtype=torch.float64, device=device)
    reference = torch.zeros(VOCAB, dtype=torch.float64, device=device)
    batch_probs = probs.float().expand(rows, VOCAB).contiguous()
    output_ids = [array("q", window)] * rows
    allowed = torch.ones(rows, dtype=torch.bool, device=device)
    top_ks = torch.full((rows,), TOP_K, device=device)
    top_ps = torch.full((rows,), TOP_P, device=device)
    min_ps = torch.zeros(rows, device=device)
    positions = torch.arange(rows, device=device)
    for _ in range(rounds):
        first = top_k_top_p_min_p_sampling_from_probs_torch(
            batch_probs.clone(), top_ks, top_ps, min_ps, False, None, positions
        )
        emitted, _ = repetition_aware_redraw(batch_probs, first, output_ids, WINDOW, allowed, None, positions)
        counts += torch.bincount(emitted.long(), minlength=VOCAB).double()
        reference += torch.bincount(
            torch.multinomial(expected.float().expand(rows, VOCAB), 1).view(-1), minlength=VOCAB
        ).double()
    total = rows * rounds
    distance = 0.5 * (counts / total - expected).abs().sum().item()
    floor = 0.5 * (reference / total - expected).abs().sum().item()
    print(f"distribution: total variation {distance:.5f} against the noise floor {floor:.5f} over {total} draws",
          flush=True)

    seeds = torch.arange(1, 33, device=device, dtype=torch.int64) * 7919
    seeded_probs = batch_probs[:32]
    seeded_first = top_k_top_p_min_p_sampling_from_probs_torch(
        seeded_probs.clone(), top_ks[:32], top_ps[:32], min_ps[:32], False, seeds, positions[:32]
    )
    one, repeated = repetition_aware_redraw(
        seeded_probs, seeded_first, output_ids[:32], WINDOW, allowed[:32], seeds, positions[:32]
    )
    two, _ = repetition_aware_redraw(
        seeded_probs, seeded_first, output_ids[:32], WINDOW, allowed[:32], seeds, positions[:32]
    )
    print(f"seeds: two seeded calls equal {torch.equal(one, two)}, rows redrawn {int(repeated.sum())} of 32",
          flush=True)

    for batch in (1, 16, 32):
        step_probs = torch.softmax(torch.randn(batch, VOCAB, device=device) / TEMPERATURE, dim=1)
        step_ids = torch.randint(0, SPEECH, (batch,), device=device, dtype=torch.int32)
        step_outputs = [array("q", torch.randint(0, SPEECH, (120,)).tolist()) for _ in range(batch)]
        step_top_ks = torch.full((batch,), TOP_K, device=device)
        step_positions = torch.arange(batch, device=device)
        for _ in range(50):
            repetition_aware_redraw(step_probs, step_ids, step_outputs, WINDOW,
                                    (step_ids < SPEECH) & (step_top_ks != 1), None, step_positions)
        torch.cuda.synchronize()
        host, device_ms = [], []
        for _ in range(500):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            began = time.perf_counter()
            repetition_aware_redraw(step_probs, step_ids, step_outputs, WINDOW,
                                    (step_ids < SPEECH) & (step_top_ks != 1), None, step_positions)
            host.append(time.perf_counter() - began)
            end.record()
            end.synchronize()
            device_ms.append(start.elapsed_time(end))
        host.sort()
        device_ms.sort()
        print(f"cost at {batch:2d} rows: host {host[len(host) // 2] * 1e3:.3f} ms median,"
              f" device span {device_ms[len(device_ms) // 2]:.3f} ms median", flush=True)


if __name__ == "__main__":
    main()
