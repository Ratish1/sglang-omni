# SPDX-License-Identifier: Apache-2.0
"""Does #2626's seeded redraw compile a second multinomial_with_seed graph in serving? SGLang's seeded top k / top
p draw feeds its torch.compile(dynamic=True) multinomial_with_seed float64 log probabilities, the redraw fed it the
log of a float32 copy. Per tree: SGLang's seeded draw twice (its graph compiled, then warm), then the tree's
seeded redraw twice. Prints each call's wall time and Dynamo's unique graph count after it. With the float64
redraw the redraw's first call should cost what a warm call does and add no graph.

    python3 seeded_redraw_compile_c20261012.py
"""

from __future__ import annotations

import time

import torch
import torch._dynamo.utils
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

from sglang_omni.sampling.repetition_aware import repetition_aware_redraw

ROWS, VOCAB = 16, 6761


def timed(label: str, call) -> None:
    torch.cuda.synchronize()
    started = time.perf_counter()
    call()
    torch.cuda.synchronize()
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    print(f"{label:28s} {(time.perf_counter() - started) * 1e3:9.1f} ms  unique graphs {graphs}", flush=True)


def main() -> None:
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(ROWS, VOCAB, device="cuda") / 0.7, dim=1)
    seeds = torch.arange(1, ROWS + 1, dtype=torch.int64, device="cuda") * 7919
    positions = torch.arange(100, 100 + ROWS, device="cuda")
    top_ks = torch.full((ROWS,), 20, device="cuda")
    top_ps = torch.full((ROWS,), 0.8, device="cuda")
    min_ps = torch.zeros(ROWS, device="cuda")
    sampled = top_k_top_p_min_p_sampling_from_probs_torch(probs.clone(), top_ks, top_ps, min_ps, False, seeds, positions)
    for attempt in ("first", "second"):
        timed(f"sglang seeded draw, {attempt}", lambda: top_k_top_p_min_p_sampling_from_probs_torch(
            probs.clone(), top_ks, top_ps, min_ps, False, seeds, positions))
    window = [[int(token)] for token in sampled.tolist()]
    allowed = torch.ones(ROWS, dtype=torch.bool, device="cuda")
    for attempt in ("first", "second"):
        timed(f"seeded redraw, {attempt}", lambda: repetition_aware_redraw(
            probs, sampled, window, 10, allowed, seeds, positions))


if __name__ == "__main__":
    main()
