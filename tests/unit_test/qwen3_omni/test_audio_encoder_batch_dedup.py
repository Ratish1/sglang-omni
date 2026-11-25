# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni encoder batching behavior."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeConfig,
)

from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    AUDIO_TOWER_CLASS,
    Qwen3OmniAudioEncoder,
)
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.stages import (
    AUDIO_STAGE,
    batch_audio_encoder_payloads,
    encoder_batch_wait_ms,
    run_single_encoder_payload,
)
from sglang_omni.proto import OmniRequest, StagePayload


class FakeAudioEncoder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        lengths = kwargs["audio_feature_lengths"].to(dtype=torch.long).view(-1)
        self.calls.append(int(lengths.shape[0]))
        total = int(lengths.sum().item())
        return {
            "audio_embeds": torch.arange(total, dtype=torch.float32).unsqueeze(1),
            "audio_feature_lengths": lengths,
            "audio_output_lengths": lengths,
        }


@pytest.mark.parametrize("raw", [None, "invalid", "-1"])
def test_encoder_batch_wait_falls_back_to_zero(
    monkeypatch: pytest.MonkeyPatch, raw: str | None
) -> None:
    if raw is None:
        monkeypatch.delenv("SGLANG_OMNI_ENCODER_BATCH_WAIT_MS", raising=False)
    else:
        monkeypatch.setenv("SGLANG_OMNI_ENCODER_BATCH_WAIT_MS", raw)

    assert encoder_batch_wait_ms() == 0


def make_payload(request_id: str, cache_key: str, time_steps: int) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hi"),
        data={
            "encoder_inputs": {
                "audio_encoder": {
                    "cache_key": cache_key,
                    "input_features": torch.ones(4, time_steps),
                    "audio_feature_lengths": torch.tensor([time_steps]),
                }
            }
        },
    )


def test_audio_encoder_batch_dedups_same_cache_key() -> None:
    model = FakeAudioEncoder()
    payloads = [
        make_payload("req-a1", "spk-a", 3),
        make_payload("req-b", "spk-b", 5),
        make_payload("req-a2", "spk-a", 3),
    ]

    out = batch_audio_encoder_payloads(payloads, model=model, cache=None)

    assert model.calls == [2]
    assert len(out) == 3
    states = [Qwen3OmniPipelineState.from_dict(p.data) for p in out]
    embeds = []
    for s in states:
        encoder_out = s.encoder_outs["audio_encoder"]
        assert isinstance(encoder_out, dict)
        embeds.append(encoder_out["audio_embeds"])
    assert all(e.shape[0] == n for e, n in zip(embeds, [3, 5, 3]))
    assert torch.equal(embeds[0], embeds[2])


def test_audio_encoder_batch_without_cache_keys_runs_every_request() -> None:
    model = FakeAudioEncoder()
    payloads = [make_payload("req-1", "k1", 3), make_payload("req-2", "k2", 3)]
    for payload in payloads:
        payload.data["encoder_inputs"]["audio_encoder"].pop("cache_key")

    out = batch_audio_encoder_payloads(payloads, model=model, cache=None)

    assert model.calls == [2]
    assert len(out) == 2


def test_packed_multiclip_audio_survives_an_encoder_batch(tmp_path: Path) -> None:
    """Co-scheduling a packed pair of clips must not drop frames or fail encoding."""
    configuration = Qwen3OmniMoeAudioEncoderConfig(
        num_mel_bins=4,
        d_model=16,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=32,
        downsample_hidden_size=8,
        output_dim=16,
    )
    Qwen3OmniMoeConfig(
        thinker_config={"audio_config": configuration.to_dict()}
    ).save_pretrained(tmp_path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        tower = AUDIO_TOWER_CLASS(configuration)
    save_file(
        {
            f"thinker.audio_tower.{name}": tensor.contiguous()
            for name, tensor in tower.state_dict().items()
        },
        tmp_path / "model.safetensors",
    )
    encoder = Qwen3OmniAudioEncoder(
        str(tmp_path), device="cpu", dtype=torch.float32
    ).eval()
    packed_request = make_payload("packed", "packed", 34)
    packed_inputs = packed_request.data["encoder_inputs"][AUDIO_STAGE]
    packed_inputs["audio_feature_lengths"] = torch.tensor([13, 21])
    packed_inputs["input_features"] = torch.linspace(-1, 1, 4 * 34).reshape(4, 34)
    payloads = [packed_request, make_payload("peer", "peer", 27)]
    standalone = [
        run_single_encoder_payload(
            deepcopy(payload), stage_name=AUDIO_STAGE, model=encoder
        )
        for payload in payloads
    ]

    batched = batch_audio_encoder_payloads(payloads, model=encoder)

    assert [payload.request_id for payload in batched] == ["packed", "peer"]
    for expected_payload, actual_payload in zip(standalone, batched, strict=True):
        expected = Qwen3OmniPipelineState.from_dict(expected_payload.data).encoder_outs[
            AUDIO_STAGE
        ]
        actual = Qwen3OmniPipelineState.from_dict(actual_payload.data).encoder_outs[
            AUDIO_STAGE
        ]
        assert isinstance(expected, dict) and isinstance(actual, dict)
        for name in ("audio_feature_lengths", "audio_output_lengths", "audio_embeds"):
            assert torch.equal(actual[name], expected[name])


def test_audio_encoder_cache_preserves_hits_in_mixed_batch():
    from sglang_omni.scheduling.stage_cache import StageOutputCache

    model = FakeAudioEncoder()
    cache = StageOutputCache(max_size=64, max_bytes=4 * 1024**3, cache_device="cpu")
    first = batch_audio_encoder_payloads(
        [
            make_payload("a1", "a", 3),
            make_payload("b1", "b", 5),
            make_payload("a2", "a", 3),
        ],
        model=model,
        cache=cache,
    )
    second = batch_audio_encoder_payloads(
        [
            make_payload("b2", "b", 5),
            make_payload("c", "c", 2),
            make_payload("a3", "a", 3),
        ],
        model=model,
        cache=cache,
    )
    assert model.calls == [2, 1]
    for old, new in [(first[1], second[0]), (first[0], second[2])]:
        old_state = Qwen3OmniPipelineState.from_dict(old.data).encoder_outs[
            "audio_encoder"
        ]
        new_state = Qwen3OmniPipelineState.from_dict(new.data).encoder_outs[
            "audio_encoder"
        ]
        assert isinstance(old_state, dict)
        assert isinstance(new_state, dict)
        assert old_state.keys() == new_state.keys()
        for key in old_state:
            assert torch.equal(old_state[key], new_state[key])


@pytest.mark.accelerator
def test_audio_encoder_cache_owns_cuda_output_before_replay():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    cuda_device = torch.device("cuda", torch.cuda.current_device())
    from sglang_omni.scheduling.stage_cache import StageOutputCache

    class ReusedOutputEncoder:
        def __init__(self):
            self.calls = 0
            self.buffer = torch.zeros(8, 1, device=cuda_device)
            self.length = torch.zeros(1, dtype=torch.long, device=cuda_device)

        def __call__(self, **kwargs):
            self.calls += 1
            size = int(kwargs["audio_feature_lengths"].sum())
            self.buffer.fill_(self.calls)
            self.length.fill_(size)
            return {
                "audio_embeds": self.buffer[:size],
                "audio_feature_lengths": self.length,
                "audio_output_lengths": self.length,
            }

    model = ReusedOutputEncoder()
    cache = StageOutputCache(max_size=64, max_bytes=4 * 1024**3, cache_device="cpu")
    first = batch_audio_encoder_payloads(
        [make_payload("a1", "a", 3)], model=model, cache=cache
    )
    first_encoder_out = Qwen3OmniPipelineState.from_dict(first[0].data).encoder_outs[
        "audio_encoder"
    ]
    assert isinstance(first_encoder_out, dict)
    expected = {key: value.cpu().clone() for key, value in first_encoder_out.items()}
    batch_audio_encoder_payloads([make_payload("b", "b", 5)], model=model, cache=cache)
    again = batch_audio_encoder_payloads(
        [make_payload("a2", "a", 3)], model=model, cache=cache
    )
    assert model.calls == 2
    actual = Qwen3OmniPipelineState.from_dict(again[0].data).encoder_outs[
        "audio_encoder"
    ]
    assert isinstance(actual, dict)
    for key, value in expected.items():
        assert actual[key].device.type == "cpu"
        assert torch.equal(actual[key], value)
