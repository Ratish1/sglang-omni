from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.dots_tts.sglang_model import DotsTTSSGLangModel


def test_forward_reads_decode_feedback_from_forward_batch():
    class _Backbone(nn.Module):
        def forward(self, **kwargs):
            return kwargs["input_embeds"]

    model = DotsTTSSGLangModel.__new__(DotsTTSSGLangModel)
    nn.Module.__init__(model)
    model.qwen2 = _Backbone()
    feedback = torch.randn(1, 4)

    output = model.forward(
        input_ids=torch.zeros(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        forward_batch=SimpleNamespace(input_embeds=feedback),
    )

    assert output is feedback
