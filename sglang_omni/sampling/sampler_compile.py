# SPDX-License-Identifier: Apache-2.0
"""The torch.compile'd sampler functions, compiled while a model loads.

multinomial_with_seed and apply_scaling_penalties each compile one entry per batch
size class (one row, two or more), per dtype and per grad mode, on the first call that
brings it. A model whose default requests reach them compiles those entries from its
precompile_kernels_after_loading, so no request pays the compile.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.layers.sampler import multinomial_with_seed
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.utils import DynamicGradMode

# note (ratish): the sampler converts filtered probabilities to float64 before a seeded
# draw, so top-k or top-p sampling reaches multinomial_with_seed in float64.
SAMPLER_LOG_PROBABILITY_DTYPE = torch.float64


@dataclass(frozen=True, kw_only=True)
class SamplerCompileForms:
    """The inputs a model's default requests give the compiled sampler functions outside
    graph replay, for rows of vocab_size entries. A dtype of None means the default
    requests never reach that function."""

    device: torch.device
    vocab_size: int
    seeded_log_probability_dtype: torch.dtype | None = None
    penalized_logits_dtype: torch.dtype | None = None


def compile_sampler_functions(forms: SamplerCompileForms) -> None:
    """Compile the entries forms reaches, on standalone tensors at one row and at two.

    An entry built on a standalone tensor also serves views of a larger buffer (one built
    on a view guards the view's base, which a standalone tensor then fails), so graph
    replayed steps' views and eager steps' tensors share it. Sampling runs inside the
    scheduler's event loops, which run under DynamicGradMode, so the entries are built
    under it.
    """
    with DynamicGradMode():
        # note (ratish): Dynamo specializes a one-row batch; the entry built at two rows
        # is dynamic and serves every larger batch.
        for rows in (1, 2):
            shape = (rows, forms.vocab_size)
            if forms.seeded_log_probability_dtype is not None:
                multinomial_with_seed(
                    torch.zeros(
                        shape,
                        dtype=forms.seeded_log_probability_dtype,
                        device=forms.device,
                    ),
                    torch.zeros(rows, dtype=torch.long, device=forms.device),
                    torch.zeros(rows, dtype=torch.long, device=forms.device),
                )
            else:
                pass
            if forms.penalized_logits_dtype is not None:
                apply_scaling_penalties(
                    torch.zeros(
                        shape, dtype=forms.penalized_logits_dtype, device=forms.device
                    ),
                    torch.ones(shape, dtype=torch.float32, device=forms.device),
                )
            else:
                pass


__all__ = [
    "SAMPLER_LOG_PROBABILITY_DTYPE",
    "SamplerCompileForms",
    "compile_sampler_functions",
]
