# SPDX-License-Identifier: Apache-2.0
"""Fixed-order projection for the Talker's one-column shared-expert gate."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


HAS_TRITON = triton is not None
SHARED_GATE_HIDDEN_SIZE = 1024

if HAS_TRITON:

    @triton.jit
    def shared_gate_projection_kernel(
        hidden_states,
        gate_weight,
        gate_logits,
        hidden_size: tl.constexpr,
    ):
        row_index = tl.program_id(0).to(tl.int64)
        channel_indices = tl.arange(0, hidden_size)
        weight = tl.load(gate_weight + channel_indices).to(tl.float32)
        activation = tl.load(
            hidden_states + row_index * hidden_size + channel_indices
        ).to(tl.float32)
        gate_logit = tl.sum(activation * weight, axis=0)
        tl.store(gate_logits + row_index, gate_logit)

else:
    pass


def project_shared_expert_gate(
    hidden_states: torch.Tensor, gate_weight: torch.Tensor
) -> torch.Tensor:
    """Use one fixed FP32 reduction per row, storing the raw BF16 logit."""
    token_count, hidden_size = hidden_states.shape
    gate_logits = hidden_states.new_empty((token_count, 1))
    shared_gate_projection_kernel[(token_count,)](
        hidden_states, gate_weight, gate_logits, hidden_size, num_warps=4
    )
    return gate_logits
