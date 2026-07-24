# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from sglang.srt.model_executor.forward_context import (
    get_forward_context,
    has_forward_context,
)

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner


def test_custom_omni_forward_publishes_sglang_forward_context():
    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    attn_backend = SimpleNamespace(init_forward_metadata=lambda _batch: None)
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=attn_backend)
    )

    seen = []

    def model(**kwargs):
        seen.append(get_forward_context().attn_backend)
        assert kwargs["input_deepstack_embeds"] is None
        return torch.ones(1, 2)

    def logits_processor(input_ids, hidden_states, lm_head, forward_batch):
        seen.append(get_forward_context().attn_backend)
        assert input_ids is forward_batch.input_ids
        assert lm_head == "lm_head"
        return "logits"

    runner._outer_model = SimpleNamespace(
        model=model,
        logits_processor=logits_processor,
        lm_head="lm_head",
    )
    forward_batch = SimpleNamespace(
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_ids=torch.tensor([1]),
    )

    assert not has_forward_context()
    result = runner._forward_with_omni_embeds(
        forward_batch,
        torch.ones(1, 2),
    )

    assert result.logits_output == "logits"
