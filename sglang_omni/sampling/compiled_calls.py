# SPDX-License-Identifier: Apache-2.0
"""SGLang's torch.compile'd sampler functions, compiled before a stage serves.

Each function compiles one entry per batch size class (one row, two or more), per
dtype, per view or standalone input and per grad mode, on the first call that brings
it. A runner whose default requests reach one states the calls it makes; the
scheduler runs them once at bind time so no request pays the compile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties

# note (ratish): Dynamo specializes a one-row batch; the entry built at two rows is
# dynamic and serves every larger batch.
WARMUP_BATCH_SIZES = (1, 2)


class SamplerCallRun(Protocol):
    def __call__(self, rows: int) -> None: ...


@dataclass(frozen=True, kw_only=True)
class CompiledSamplerCall:
    """One compiled call as a runner's requests make it, for a batch of rows."""

    run: SamplerCallRun
    grad_enabled: bool


def warm_compiled_sampler_calls(calls: tuple[CompiledSamplerCall, ...]) -> None:
    for call in calls:
        with torch.inference_mode(False), torch.set_grad_enabled(call.grad_enabled):
            for rows in WARMUP_BATCH_SIZES:
                call.run(rows)


def seeded_sampling_call(
    *,
    device: torch.device,
    log_probability_dtype: torch.dtype,
    vocab_size: int,
    seed_buffer: torch.Tensor | None,
    grad_enabled: bool,
) -> CompiledSamplerCall:
    """multinomial_with_seed with standalone log-probabilities and positions; the
    seeds standalone, or rows of the runner's own seed buffer when it installs those."""

    def run(rows: int) -> None:
        seeds = (
            seed_buffer[:rows]
            if seed_buffer is not None
            else torch.zeros(rows, dtype=torch.long, device=device)
        )
        multinomial_with_seed(
            torch.zeros((rows, vocab_size), dtype=log_probability_dtype, device=device),
            seeds,
            torch.zeros(rows, dtype=torch.long, device=device),
        )

    return CompiledSamplerCall(run=run, grad_enabled=grad_enabled)


def scaling_penalty_calls(
    *, device: torch.device, logits_dtype: torch.dtype, vocab_size: int
) -> tuple[CompiledSamplerCall, CompiledSamplerCall]:
    """apply_scaling_penalties as the base runner's sampling reaches it: the logits a
    view of a graph's output buffer on replayed steps and standalone on eager steps,
    the float32 penalties standalone, grad enabled."""

    def run_on_view(rows: int) -> None:
        logits = torch.zeros((rows, vocab_size), dtype=logits_dtype, device=device)
        apply_scaling_penalties(
            logits[:rows],
            torch.ones((rows, vocab_size), dtype=torch.float32, device=device),
        )

    def run_on_standalone(rows: int) -> None:
        apply_scaling_penalties(
            torch.zeros((rows, vocab_size), dtype=logits_dtype, device=device),
            torch.ones((rows, vocab_size), dtype=torch.float32, device=device),
        )

    return (
        CompiledSamplerCall(run=run_on_view, grad_enabled=True),
        CompiledSamplerCall(run=run_on_standalone, grad_enabled=True),
    )


__all__ = [
    "CompiledSamplerCall",
    "WARMUP_BATCH_SIZES",
    "scaling_penalty_calls",
    "seeded_sampling_call",
    "warm_compiled_sampler_calls",
]
