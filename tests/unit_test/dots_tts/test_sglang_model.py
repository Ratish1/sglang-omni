# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.dots_tts.serving_types import (
    DotsTTSBatchedAudioStepResult,
    DotsTTSFlowBatchKey,
)
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
        self.decode_batches = []

    def prepare_flow_batch_key(
        self,
        *,
        fm_state,
        generation_kwargs,
        precision,
    ):
        del fm_state, precision
        return generation_kwargs["batch_key"]

    def decode_audio_batch_step(self, items, *, precision):
        self.decode_batches.append(
            (precision, [item.request_index for item in items])
        )
        return DotsTTSBatchedAudioStepResult(
            request_indices=[item.request_index for item in items],
            latent_patches=[
                torch.full((1, 4, 2), float(item.request_index)) for item in items
            ],
            feedback_embeddings=[
                torch.full((1, 1, 3), float(item.request_index)) for item in items
            ],
            eos_scores=[torch.zeros(1) for _ in items],
        )


def flow_key(name: str) -> DotsTTSFlowBatchKey:
    return DotsTTSFlowBatchKey(
        device=torch.device("cpu"),
        dtype=torch.float32,
        mode="flow_matching",
        ode_method=name,
        num_steps=2,
        guidance_scale=1.2,
        history_bucket_capacity=8,
        latent_patch_size=4,
        hidden_patch_size=1,
    )


def test_decode_audio_batch_uses_native_batch_step(monkeypatch) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", FakeQwen2)
    model = DotsTTSSGLangModel(SimpleNamespace(torch_dtype="bfloat16"))
    native_model = FakeNativeDotsModel()
    model._native_model = native_model
    data = SimpleNamespace(
        fm_state={"history": []},
        generation_kwargs={
            "device": torch.device("cpu"),
            "batch_key": flow_key("euler"),
            "ode_method": "euler",
            "num_steps": 2,
            "guidance_scale": 1.2,
            "eos_threshold": 0.8,
        },
        control_token_id=3,
        latent_patches=[],
        max_generate_length=4,
    )
    hidden_state = torch.ones(1, 1, 2048)
    batch = mod.DotsTTSLatentBatch(
        requests=[SimpleNamespace(data=data)],
        active_indices=[0],
        hidden_states=[hidden_state],
    )

    output = model.decode_audio_batch(batch)

    assert native_model.decode_batches == [("bfloat16", [0])]
    assert output.latent_patches[0].shape == (1, 4, 2)
    assert output.feedback_embeddings[0].shape == (1, 1, 3)
    assert torch.equal(output.eos_scores[0], torch.tensor([0.0]))


def test_forward_latent_decode_step_returns_control_ids_and_latents(
    monkeypatch,
) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    class HiddenQwen2(nn.Module):
        def __init__(self, config, quant_config=None, prefix="") -> None:
            super().__init__()
            self.embed = nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embed

        def forward(self, **kwargs):
            return SimpleNamespace(
                hidden_states=torch.ones((kwargs["input_ids"].numel(), 1, 4))
            )

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", HiddenQwen2)
    model = DotsTTSSGLangModel(SimpleNamespace(torch_dtype="bfloat16"))
    native_model = FakeNativeDotsModel()
    model._native_model = native_model
    data = SimpleNamespace(
        finish_reason=None,
        fm_state=object(),
        generation_kwargs={"batch_key": flow_key("euler")},
        latest_hidden_state=None,
        latest_latent_patch=None,
        latent_patches=[],
        decode_input_embeds=[],
        eos_score=None,
        control_token_id=0,
        max_generate_length=2,
    )
    req = SimpleNamespace(data=data)

    output = model.forward_latent_decode_step(
        input_ids=torch.tensor([0]),
        positions=torch.tensor([0]),
        forward_batch=SimpleNamespace(mrope_positions=None),
        requests=[req],
    )

    assert output.next_token_ids.tolist() == [0]
    assert len(output.latent_patches) == 1
    assert output.latent_patches[0].shape == (1, 4, 2)
    assert output.feedback_embeddings[0].shape == (1, 1, 3)
    assert native_model.decode_batches == [("bfloat16", [0])]


def test_decode_audio_batch_skips_finished_requests(monkeypatch) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", FakeQwen2)
    model = DotsTTSSGLangModel(SimpleNamespace(torch_dtype="bfloat16"))
    native_model = FakeNativeDotsModel()
    model._native_model = native_model
    active = SimpleNamespace(
        finish_reason=None,
        fm_state=object(),
        generation_kwargs={"batch_key": flow_key("euler")},
        control_token_id=101,
        latent_patches=[],
        max_generate_length=4,
    )
    finished = SimpleNamespace(
        finish_reason="stop",
        fm_state=object(),
        control_token_id=202,
        latent_patches=[],
        max_generate_length=4,
    )
    batch = mod.DotsTTSLatentBatch(
        requests=[SimpleNamespace(data=active), SimpleNamespace(data=finished)],
        active_indices=[0],
        hidden_states=[
            torch.ones((1, 1, 4)),
            torch.full((1, 1, 4), 2.0),
        ],
    )

    output = model.decode_audio_batch(batch)

    assert native_model.decode_batches == [("bfloat16", [0])]
    assert output.latent_patches[0].shape == (1, 4, 2)
    assert output.latent_patches[1] is None
    assert output.feedback_embeddings[0].shape == (1, 1, 3)
    assert output.feedback_embeddings[1] is None


def test_decode_audio_batch_splits_incompatible_flow_keys_and_preserves_order(
    monkeypatch,
) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", FakeQwen2)
    model = DotsTTSSGLangModel(SimpleNamespace(torch_dtype="bfloat16"))
    native_model = FakeNativeDotsModel()
    model._native_model = native_model
    first = SimpleNamespace(
        finish_reason=None,
        fm_state=object(),
        generation_kwargs={"batch_key": flow_key("euler")},
        control_token_id=11,
        latent_patches=[],
        max_generate_length=4,
    )
    inactive = SimpleNamespace(
        finish_reason="stop",
        fm_state=object(),
        generation_kwargs={"batch_key": flow_key("euler")},
        control_token_id=22,
        latent_patches=[],
        max_generate_length=4,
    )
    second = SimpleNamespace(
        finish_reason=None,
        fm_state=object(),
        generation_kwargs={"batch_key": flow_key("midpoint")},
        control_token_id=33,
        latent_patches=[],
        max_generate_length=4,
    )
    third = SimpleNamespace(
        finish_reason=None,
        fm_state=object(),
        generation_kwargs={"batch_key": flow_key("euler")},
        control_token_id=44,
        latent_patches=[],
        max_generate_length=4,
    )
    batch = mod.DotsTTSLatentBatch(
        requests=[
            SimpleNamespace(data=first),
            SimpleNamespace(data=inactive),
            SimpleNamespace(data=second),
            SimpleNamespace(data=third),
        ],
        active_indices=[0, 2, 3],
        hidden_states=[
            torch.ones((1, 1, 4)),
            None,
            torch.full((1, 1, 4), 2.0),
            torch.full((1, 1, 4), 3.0),
        ],
    )

    output = model.decode_audio_batch(batch)

    assert native_model.decode_batches == [
        ("bfloat16", [0, 3]),
        ("bfloat16", [2]),
    ]
    assert output.next_token_ids.tolist() == [11, 22, 33, 44]
    assert torch.equal(output.latent_patches[0], torch.zeros((1, 4, 2)))
    assert output.latent_patches[1] is None
    assert torch.equal(output.latent_patches[2], torch.full((1, 4, 2), 2.0))
    assert torch.equal(output.latent_patches[3], torch.full((1, 4, 2), 3.0))


def test_model_does_not_expose_legacy_generate_latent_patch(monkeypatch) -> None:
    import sglang_omni.models.dots_tts.sglang_model as mod

    monkeypatch.setattr(mod, "Qwen2ForCausalLM", FakeQwen2)
    model = DotsTTSSGLangModel(SimpleNamespace(torch_dtype="bfloat16"))

    assert not hasattr(model, "generate_latent_patch")


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
