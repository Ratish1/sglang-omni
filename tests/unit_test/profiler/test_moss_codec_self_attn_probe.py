# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.debug.moss_codec_self_attn_probe import (
    _call_parity,
    _snapshot_streaming_state,
    _target_probe_patch,
    _target_self_attn,
)


class _FakeKVCache:
    def __init__(self) -> None:
        self.cache = torch.zeros(2, 1, 1, 2, 1)
        self.end_offset = torch.zeros(1, dtype=torch.long)


class _FakeState:
    def __init__(self) -> None:
        self.exec_mask = torch.ones(1, dtype=torch.bool)
        self.offset = torch.zeros(1, dtype=torch.long)
        self.offset_cpu = 0
        self.kv_cache = _FakeKVCache()


class _FakeSelfAttention:
    def __init__(self) -> None:
        self._streaming_state = _FakeState()

    def forward(self, x):
        self._streaming_state.offset += 1
        self._streaming_state.offset_cpu += 1
        self._streaming_state.kv_cache.end_offset += 1
        self._streaming_state.kv_cache.cache += x.view(1, 1, 1, 1, 1)
        return x + 1


def _fake_processor(target: _FakeSelfAttention) -> SimpleNamespace:
    layer = SimpleNamespace(self_attn=target)
    transformer = SimpleNamespace(layers=[layer])
    decoder_module = SimpleNamespace(transformer=transformer)
    codec = SimpleNamespace(decoder=[decoder_module])
    return SimpleNamespace(audio_tokenizer=codec)


def test_self_attn_probe_resolves_target_module() -> None:
    target = _FakeSelfAttention()
    processor = _fake_processor(target)

    assert _target_self_attn(processor, 0, 0) is target


def test_target_probe_patch_restores_forward_and_captures_state() -> None:
    target = _FakeSelfAttention()
    original_forward = target.forward

    with _target_probe_patch(target, "target_identity") as (calls, stats):
        output = target.forward(torch.tensor([2.0]))

    assert output.item() == 3.0
    assert target.forward == original_forward
    assert stats["candidate"] == "target_identity"
    assert len(calls) == 1
    assert torch.equal(calls[0]["output"], torch.tensor([3.0]))
    state = calls[0]["state"]
    assert state["offset_cpu"] == 1
    assert torch.equal(state["offset"], torch.tensor([1]))
    assert torch.equal(state["kv_cache"]["end_offset"], torch.tensor([1]))


def test_snapshot_streaming_state_detaches_tensor_values() -> None:
    target = _FakeSelfAttention()
    snapshot = _snapshot_streaming_state(target)
    target._streaming_state.offset += 7

    assert torch.equal(snapshot["offset"], torch.tensor([0]))


def test_call_parity_detects_output_and_state_drift() -> None:
    reference = [
        {
            "output": torch.tensor([1.0]),
            "state": {
                "offset": torch.tensor([1]),
                "offset_cpu": 1,
                "kv_cache": {"cache": torch.ones(1)},
            },
        }
    ]
    same = [
        {
            "output": torch.tensor([1.0]),
            "state": {
                "offset": torch.tensor([1]),
                "offset_cpu": 1,
                "kv_cache": {"cache": torch.ones(1)},
            },
        }
    ]
    drift = [
        {
            "output": torch.tensor([1.25]),
            "state": {
                "offset": torch.tensor([2]),
                "offset_cpu": 2,
                "kv_cache": {"cache": torch.ones(1) * 3},
            },
        }
    ]

    assert _call_parity(reference, same, "output")["max_abs"] == 0.0
    assert _call_parity(reference, same, "state")["max_abs"] == 0.0
    assert _call_parity(reference, drift, "output")["max_abs"] == 0.25
    assert _call_parity(reference, drift, "state")["max_abs"] == 2.0
