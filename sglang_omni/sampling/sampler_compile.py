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


@dataclass(frozen=True, kw_only=True)
class SamplerCompileForms:
    device: torch.device
    vocab_size: int
    # note (ratish): None means the model's default requests never reach that function.
    seeded_log_probability_dtype: torch.dtype | None = None
    penalized_logits_dtype: torch.dtype | None = None


def compile_sampler_functions(forms: SamplerCompileForms) -> None:
    # note (ratish): the scheduler's event loops sample under DynamicGradMode, and each
    # entry is guarded on the grad mode it was built in.
    with DynamicGradMode():
        # note (ratish): Dynamo specializes a one-row batch; the entry built at two rows
        # is dynamic and serves every larger batch.
        for rows in (1, 2):
            shape = (rows, forms.vocab_size)
            # note (ratish): standalone tensors, as an entry built on one also serves
            # views of a graph buffer, while one built on a view guards its base.
            if forms.seeded_log_probability_dtype is not None:
                multinomial_with_seed(
                    logprobs=torch.zeros(
                        shape,
                        dtype=forms.seeded_log_probability_dtype,
                        device=forms.device,
                    ),
                    seed=torch.zeros(rows, dtype=torch.long, device=forms.device),
                    positions=torch.zeros(rows, dtype=torch.long, device=forms.device),
                )
            else:
                pass
            if forms.penalized_logits_dtype is not None:
                apply_scaling_penalties(
                    logits=torch.zeros(
                        shape, dtype=forms.penalized_logits_dtype, device=forms.device
                    ),
                    scaling_penalties=torch.ones(
                        shape, dtype=torch.float32, device=forms.device
                    ),
                )
            else:
                pass
