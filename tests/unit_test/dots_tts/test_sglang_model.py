# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.dots_tts.sglang_model import DotsTTSSGLangModel


class FakeQwen2(nn.Module):
    def __init__(self, config, quant_config=None, prefix="") -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.prefix = prefix
        self.loaded_weights = None

    def load_weights(self, weights):
        self.loaded_weights = list(weights)
        return {"qwen2-loaded"}

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return "qwen2-output"


def test_model_owns_qwen2_backbone(monkeypatch) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", FakeQwen2)
    config = SimpleNamespace(torch_dtype="bfloat16", hidden_size=2048)

    model = DotsTTSSGLangModel(config, quant_config=None, prefix="dots")

    assert isinstance(model.qwen2, FakeQwen2)
    assert model.qwen2.config is config
    assert model.qwen2.prefix == "dots.qwen2"


class FakeNativeDotsModel:
    def __init__(self) -> None:
        self.appended_hidden = None

    def _append_hidden_chunk(self, fm_state, hidden_state):
        assert fm_state == {"history": []}
        self.appended_hidden = hidden_state

    def _decode_next_audio(self, **kwargs):
        assert kwargs["state"] == {"history": []}
        assert kwargs["ode_method"] == "euler"
        assert kwargs["num_steps"] == 2
        assert kwargs["guidance_scale"] == 1.2
        return torch.ones(1, 4, 128)

    def _encode_audio_patch_feedback(self, fm_state, *, audio_patch):
        assert fm_state == {"history": []}
        return audio_patch.mean(dim=1, keepdim=True)

    def _should_stop_after_current_audio(self, fm_state, *, eos_threshold):
        assert fm_state == {"history": []}
        assert eos_threshold == 0.8
        return False


def test_model_steps_audio_latent_without_adapter(monkeypatch) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", FakeQwen2)
    model = DotsTTSSGLangModel(SimpleNamespace(torch_dtype="bfloat16"))
    native_model = FakeNativeDotsModel()
    model._native_model = native_model
    data = SimpleNamespace(
        fm_state={"history": []},
        generation_kwargs={
            "device": torch.device("cpu"),
            "g_cond": None,
            "ode_method": "euler",
            "num_steps": 2,
            "guidance_scale": 1.2,
            "eos_threshold": 0.8,
        },
    )
    hidden_state = torch.ones(1, 1, 2048)

    result = model.step_audio_latent(data, hidden_state)

    assert native_model.appended_hidden is hidden_state
    assert result.latent_patch.shape == (1, 4, 128)
    assert result.feedback_embedding.shape == (1, 1, 128)
    assert torch.equal(result.eos_score, torch.tensor([0.0]))


def test_legacy_generate_latent_patch_uses_model_step(monkeypatch) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", FakeQwen2)
    model = DotsTTSSGLangModel(SimpleNamespace(torch_dtype="bfloat16"))
    native_model = FakeNativeDotsModel()
    model._native_model = native_model

    result = model.generate_latent_patch(
        hidden_state=torch.ones(1, 1, 2048),
        fm_state={"history": []},
        generation_kwargs={
            "device": torch.device("cpu"),
            "g_cond": None,
            "ode_method": "euler",
            "num_steps": 2,
            "guidance_scale": 1.2,
            "eos_threshold": 0.8,
        },
    )

    assert result.latent_patch.shape == (1, 4, 128)
    assert native_model.appended_hidden.shape == (1, 1, 2048)


def test_model_load_weights_routes_llm_weights_to_qwen2(monkeypatch) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", FakeQwen2)
    model = DotsTTSSGLangModel(SimpleNamespace(torch_dtype="bfloat16"))
    embed = torch.ones(2, 3)
    q_proj = torch.ones(3, 4)
    side = torch.ones(1)

    result = model.load_weights(
        [
            ("llm.model.embed_tokens.weight", embed),
            ("llm.model.layers.0.self_attn.q_proj.weight", q_proj),
            ("core.dit.blocks.0.weight", side),
        ]
    )

    assert model.qwen2.loaded_weights == [
        ("model.embed_tokens.weight", embed),
        ("model.layers.0.self_attn.q_proj.weight", q_proj),
    ]
    assert result == {"qwen2-loaded"}


def test_model_forward_delegates_to_qwen2(monkeypatch) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", FakeQwen2)
    model = DotsTTSSGLangModel(SimpleNamespace(torch_dtype="bfloat16"))
    input_ids = torch.tensor([1])
    positions = torch.tensor([0])
    input_embeds = torch.ones(1, 8)
    forward_batch = SimpleNamespace()

    result = model(
        input_ids=input_ids,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
    )

    assert result == "qwen2-output"
    assert model.qwen2.forward_kwargs == {
        "input_ids": input_ids,
        "positions": positions,
        "forward_batch": forward_batch,
        "input_embeds": input_embeds,
    }
