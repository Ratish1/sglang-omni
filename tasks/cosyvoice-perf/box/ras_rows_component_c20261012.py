# SPDX-License-Identifier: Apache-2.0
"""Component test of the redraw moved into collect_tokens (our #2626 branch 021f62916), against the redraw over
every row it replaces (#2626 with our float64 fix, ff5a2305c, copied below as old_redraw):
1. Seeded identity: for random tempered rows with seeds and positions, the rows that repeat get the same ids from
   the new rows only redraw as from the old one over every row (SGLang's seeded draw hashes each seed, position
   and token on its own, so a row's draw does not depend on the other rows).
2. Cost per decode step at 1, 16 and 32 rows, seeded and not: the old path (window H2D and the redraw over every
   row, every step) against the new path on a step without a repeat (the window checked on host ids) and on a
   step with one repeated row (the redraw of that row and its re-staging). The new path includes the step's id
   read, which collect_tokens does with or without the redraw, so that read is timed alone too. Host time and
   device span, medians.

    python3 ras_rows_component_c20261012.py
"""

from __future__ import annotations

import time
from array import array

import torch
from sglang.srt.layers.sampler import sampling_from_probs_torch

from sglang_omni.sampling.repetition_aware import REDRAW_SEED_SALT, repetition_aware_redraw

VOCAB, SPEECH, WINDOW = 6761, 6561, 10


def old_redraw(probs, sampled_ids, output_ids, redraw_allowed, sampling_seeds, positions):
    recent_rows = [list(row_ids[-WINDOW:]) for row_ids in output_ids]
    recent_ids = torch.tensor(
        [row + [-1] * (WINDOW - len(row)) for row in recent_rows], dtype=torch.long
    ).to(probs.device, non_blocking=True)
    candidate_ids = sampled_ids.long().unsqueeze(1)
    is_repeated = (recent_ids == candidate_ids).any(dim=1) & redraw_allowed
    redraw_probs = probs.to(torch.float64, copy=True)
    redraw_probs.scatter_(1, candidate_ids, 0.0)
    redraw_probs = torch.where(redraw_probs.sum(dim=1, keepdim=True) > 0, redraw_probs, probs)
    redraw_seeds = None if sampling_seeds is None else sampling_seeds ^ REDRAW_SEED_SALT
    redraw_ids = sampling_from_probs_torch(redraw_probs, redraw_seeds, positions)
    return torch.where(is_repeated, redraw_ids, sampled_ids), is_repeated


def new_step(probs, sampled_ids, output_ids, top_ks, sampling_seeds, positions):
    """The collect_tokens path: host ids, the window on the host, the redraw of repeated rows, re-staging."""
    host_ids = sampled_ids.to("cpu", non_blocking=True)
    torch.cuda.current_stream().synchronize()
    token_ids = host_ids.tolist()
    rows = [
        row for row, (token_id, history, top_k) in enumerate(zip(token_ids, output_ids, top_ks, strict=True))
        if token_id < SPEECH and top_k != 1 and token_id in history[-WINDOW:]
    ]
    if not rows:
        return sampled_ids
    redrawn = repetition_aware_redraw(
        probs, sampled_ids, torch.tensor(rows, device=probs.device), sampling_seeds, positions
    )
    redrawn.to("cpu", non_blocking=True)
    torch.cuda.current_stream().synchronize()
    return redrawn


def timed(call, repeats: int = 400) -> tuple[float, float]:
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    host, device = [], []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        began = time.perf_counter()
        call()
        host.append(time.perf_counter() - began)
        end.record()
        end.synchronize()
        device.append(start.elapsed_time(end))
    host.sort()
    device.sort()
    return host[len(host) // 2] * 1e3, device[len(device) // 2]


def main() -> None:
    torch.manual_seed(0)
    rows = 32
    probs = torch.softmax(torch.randn(rows, VOCAB, device="cuda") * 3 / 0.7, dim=1)
    sampled = torch.randint(0, SPEECH, (rows,), device="cuda", dtype=torch.int32)
    seeds = torch.arange(1, rows + 1, device="cuda", dtype=torch.int64) * 104729
    positions = torch.arange(200, 200 + rows, device="cuda")
    host_sampled = sampled.tolist()
    output_ids = [array("q", [host_sampled[row], 5]) if row % 3 == 0 else array("q", [5, 6]) for row in range(rows)]
    allowed = torch.ones(rows, dtype=torch.bool, device="cuda")
    old_ids, is_repeated = old_redraw(probs, sampled, output_ids, allowed, seeds, positions)
    repeated = [row for row in range(rows) if bool(is_repeated[row])]
    new_ids = repetition_aware_redraw(probs, sampled, torch.tensor(repeated, device="cuda"), seeds, positions)
    print(f"seeded identity: {len(repeated)} repeated rows, ids equal {torch.equal(old_ids, new_ids)}", flush=True)

    for batch in (1, 16, 32):
        step_probs = probs[:batch].contiguous()
        step_sampled = torch.randint(0, SPEECH, (batch,), device="cuda", dtype=torch.int32)
        quiet = [array("q", torch.randint(0, SPEECH, (120,)).tolist()) for _ in range(batch)]
        first = step_sampled[:1].tolist()[0]
        one_repeat = [array("q", list(quiet[0]) + [first])] + quiet[1:]
        top_ks = [20] * batch
        step_allowed = torch.ones(batch, dtype=torch.bool, device="cuda")
        step_positions = positions[:batch]
        for seeded in (False, True):
            step_seeds = seeds[:batch] if seeded else None
            read = timed(lambda: (step_sampled.to("cpu", non_blocking=True), torch.cuda.current_stream().synchronize()))
            old = timed(lambda: old_redraw(step_probs, step_sampled, quiet, step_allowed, step_seeds, step_positions))
            new_quiet = timed(lambda: new_step(step_probs, step_sampled, quiet, top_ks, step_seeds, step_positions))
            new_repeat = timed(lambda: new_step(step_probs, step_sampled, one_repeat, top_ks, step_seeds, step_positions))
            print(
                f"{batch:2d} rows {'seeded  ' if seeded else 'unseeded'}: old every step {old[0]:.3f} / {old[1]:.3f} ms"
                f" | new, no repeat {new_quiet[0]:.3f} / {new_quiet[1]:.3f} ms"
                f" | new, one repeat {new_repeat[0]:.3f} / {new_repeat[1]:.3f} ms"
                f" | the step's id read alone {read[0]:.3f} / {read[1]:.3f} ms   (host / device span)",
                flush=True,
            )


if __name__ == "__main__":
    main()
