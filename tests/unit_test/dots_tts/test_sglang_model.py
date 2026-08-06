from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.dots_tts.sglang_model import DotsTTSSGLangModel


def test_forward_reads_decode_feedback_from_forward_batch() -> None:
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


def test_weight_loader_routes_every_checkpoint_namespace() -> None:
    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received: list[tuple[str, torch.Tensor]] = []

        def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
            self.received = list(weights)
            return {name for name, _tensor in self.received}

    class _Flow(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for root in (
                "patch_encoder",
                "hidden_proj",
                "latent_proj",
                "coordinate_proj",
                "xvec_proj",
                "velocity_field_predictor",
                "eos_proj",
            ):
                setattr(self, root, nn.Linear(1, 1, bias=False))

    model = DotsTTSSGLangModel.__new__(DotsTTSSGLangModel)
    nn.Module.__init__(model)
    model.qwen2 = _Backbone()
    model.flow = _Flow()
    flow_weights = [
        (name, torch.full_like(parameter, float(index + 1)))
        for index, (name, parameter) in enumerate(model.flow.named_parameters())
    ]
    codec_weights = [
        (f"{root}.unused", torch.ones(1))
        for root in (
            "audio_encoder",
            "dec_mi_layer",
            "decoder",
            "enc_mi_layer",
            "model",
            "post_proj",
            "pre_proj",
            "resample",
        )
    ]

    loaded = model.load_weights(
        [
            ("llm.model.embed_tokens.weight", torch.tensor([11.0])),
            *flow_weights,
            *codec_weights,
            ("new_acoustic_block.weight", torch.ones(1, 1)),
        ]
    )

    assert [name for name, _tensor in model.qwen2.received] == [
        "model.embed_tokens.weight"
    ]
    assert loaded == {
        "qwen2.model.embed_tokens.weight",
        *(f"flow.{name}" for name, _tensor in flow_weights),
    }
    for index, (_name, parameter) in enumerate(model.flow.named_parameters()):
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, float(index + 1)),
        )
