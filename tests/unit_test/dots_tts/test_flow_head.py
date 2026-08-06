# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead

LLM_HIDDEN = 48
FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2
NFE = 2


def _flow_head(tmp_path) -> DotsTTSFlowHead:
    torch.save(
        {"mean": torch.zeros(LATENT_DIM), "var": torch.ones(LATENT_DIM)},
        tmp_path / "latent_stats.pt",
    )
    config = {
        "latent_dim": LATENT_DIM,
        "patch_size": PATCH_SIZE,
        "PatchEncoder": {
            "num_layers": 1,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "causal": True,
        },
        "DiT": {
            "num_layers": 2,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "modulation": True,
            "qk_norm": True,
            "rotary_bias": True,
        },
        "vocoder": {"sample_rate": 48000},
        "meanflow": {"enabled": True, "use_duration_embedding": True},
    }
    flow = DotsTTSFlowHead(
        config,
        llm_hidden_size=LLM_HIDDEN,
        latent_stats_path=str(tmp_path / "latent_stats.pt"),
        optimize=False,
    )
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.normal_(0.0, 0.2)
    return flow.eval()


def test_single_stream_decode_batch_accepts_2d_hidden(tmp_path) -> None:
    torch.manual_seed(1234)
    flow = _flow_head(tmp_path)
    state, prompt_embeddings = flow.new_request(
        max_audio_patch_count=8,
        prompt_latents=None,
        speaker_embedding=torch.randn(1, 512),
        speaker_scale=1.5,
    )
    assert prompt_embeddings is None
    flow.append_hidden(state, torch.randn(1, 1, LLM_HIDDEN))

    def _decode(*, append_hidden: bool):
        # The model runner passes rank-2 [batch, hidden] rows for decode.
        return flow.decode_batch(
            [state],
            hidden_states=torch.randn(1, LLM_HIDDEN),
            num_steps=[NFE],
            ode_methods=["euler"],
            guidance_scales=[1.2],
            eos_thresholds=[2.0],
            append_hidden=append_hidden,
        )

    [first] = _decode(append_hidden=False)
    [second] = _decode(append_hidden=True)

    for step in (first, second):
        assert step.latent_patch.shape == (1, PATCH_SIZE, LATENT_DIM)
        assert step.feedback_embedding.shape[-1] == LLM_HIDDEN
        assert step.emit
        assert not step.finished
    assert state.decoded_patches == 2
