# SPDX-License-Identifier: Apache-2.0
"""Replay recorded c16 streaming inbox sequences through the vocoder scheduler.

The fixtures are the tts_engine to vocoder message order captured on an H100
(16 requests, and 64 requests with a compiled DiT). The replay feeds them at
their recorded arrival times against a fake clock, charges every decode step a
fixed cost, and checks order, completion and liveness.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.stages import FunCosyVoice3Flow
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    SAMPLE_RATE,
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
)
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FINAL,
    FunCosyVoice3StreamingVocoderScheduler,
    _slack_s,
    _step_kind,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.scheduling.messages import IncomingMessage, OutgoingMessage
from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow
from tests.unit_test.fun_cosyvoice3.test_streaming import (
    _Clock,
    _FakeHiFT,
    _stream_payload,
    _waveform,
)

FIXTURES = Path(__file__).parent / "fixtures"
# CosyVoice3 mel runs at 50 frames per second.
SAMPLES_PER_FRAME = SAMPLE_RATE // 50
# Arbitrary; the liveness bound scales with it and one hop of audio (1 s)
# stays far above it, which is what the bound's argument needs.
STEP_COST_S = 0.02
PROMPT_TOKENS = TOKEN_HOP_LEN
# The packed fake embeds 32 token ids.
FAKE_VOCAB = 32


class _ReplayFlow(_PackedFlow):
    def __init__(self) -> None:
        super().__init__(channels=80, max_frames=2048)
        self.spk_embed_affine_layer = torch.nn.Linear(192, 80, bias=False)

    def inference(self, **kwargs):
        token_count = int(kwargs["token"].shape[1]) - PRE_LOOKAHEAD_LEN
        return torch.ones(1, 80, token_count * TOKEN_MEL_RATIO), None


class _ReplayHiFT(_FakeHiFT):
    def inference(self, *, speech_feat, finalize):
        frames = int(speech_feat.shape[-1])
        self.calls.append((frames, finalize))
        return torch.zeros(1, frames * SAMPLES_PER_FRAME), None


class _ReplayVocoder(stages.CosyVoice3Vocoder):
    def __init__(self, flow: _ReplayFlow, hift: _ReplayHiFT) -> None:
        super().__init__(FunCosyVoice3Flow(flow), hift)
        self.leftover_tokens: list[list[int]] = []

    def leftover_batch(self, items):
        self.leftover_tokens.extend(item.token.flatten().tolist() for item in items)
        return super().leftover_batch(items)


@dataclass
class _ReplayStats:
    expected: dict[str, list[int]] = field(default_factory=dict)
    messages: list[OutgoingMessage] = field(default_factory=list)
    final_ready: dict[str, tuple[float, float]] = field(default_factory=dict)
    result_at: dict[str, float] = field(default_factory=dict)
    steps: int = 0
    max_inflight: int = 0


def _prompt_metadata() -> dict:
    return {
        "modality": "audio_codes",
        "stream": True,
        "flow_prompt_speech_token": torch.zeros(1, PROMPT_TOKENS, dtype=torch.int32),
        "flow_prompt_speech_feat": torch.zeros(1, PROMPT_TOKENS * TOKEN_MEL_RATIO, 80),
        "flow_embedding": torch.ones(1, 192),
    }


def _replay(
    events: list[dict],
) -> tuple[_ReplayFlow, FunCosyVoice3StreamingVocoderScheduler, _ReplayStats]:
    flow = _ReplayFlow()
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        _ReplayVocoder(flow, _ReplayHiFT()), max_batch_size=8
    )
    clock = _Clock(events[0]["t_ns"] / 1e9)
    scheduler._clock = clock
    rng = random.Random(0)
    stats = _ReplayStats()
    pending = list(events)

    def feed_due() -> None:
        while pending and pending[0]["t_ns"] / 1e9 <= clock.now:
            event = pending.pop(0)
            request_id = event["request_id"]
            if event["type"] == "stream_chunk":
                tokens = [rng.randrange(FAKE_VOCAB) for _ in range(event["tokens"])]
                first = request_id not in stats.expected
                stats.expected.setdefault(request_id, []).extend(tokens)
                metadata = (
                    _prompt_metadata()
                    if first
                    else {"modality": "audio_codes", "stream": True}
                )
                item = StreamItem(
                    chunk_id=event["chunk_id"],
                    data=torch.tensor(tokens, dtype=torch.long),
                    from_stage="tts_engine",
                    metadata=metadata,
                )
                scheduler.inbox.put(IncomingMessage(request_id, "stream_chunk", item))
                continue
            scheduler.inbox.put(IncomingMessage(request_id, "stream_done"))
            payload = _stream_payload(
                request_id,
                codes=stats.expected[request_id],
                prompt_token_len=PROMPT_TOKENS,
                prompt_feat_frames=PROMPT_TOKENS * TOKEN_MEL_RATIO,
            )
            scheduler.inbox.put(IncomingMessage(request_id, "new_request", payload))

    def observe() -> None:
        stats.max_inflight = max(stats.max_inflight, len(scheduler._stream_states))
        for request_id, state in scheduler._stream_state_items():
            if request_id in stats.final_ready:
                continue
            if _step_kind(state) == FINAL:
                slack = 0.0
                if state.first_emit_at is not None:
                    slack = _slack_s(
                        state, now=clock.now, sample_rate=scheduler._sample_rate
                    )
                stats.final_ready[request_id] = (clock.now, slack)
        while True:
            try:
                message = scheduler.outbox.get_nowait()
            except Empty:
                return
            stats.messages.append(message)
            if message.type == "result":
                stats.result_at[message.request_id] = clock.now

    while True:
        feed_due()
        try:
            message = scheduler._get_batch_message()
        except Empty:
            if scheduler._has_ready_work():
                scheduler._run_ready_step()
                stats.steps += 1
                clock.now += STEP_COST_S
            elif pending:
                clock.now = pending[0]["t_ns"] / 1e9
            else:
                break
        else:
            scheduler._handle_message(message, None)
        observe()
    return flow, scheduler, stats


@pytest.mark.parametrize(
    "fixture",
    ["streaming_c16_gpu0_inbox.json", "streaming_c16_gpu1_compile_64req_inbox.json"],
)
def test_recorded_c16_inbox_replays_in_order_and_completes(fixture: str) -> None:
    """Under a c16 backlog every request's tokens reach Flow in arrival order
    and its final runs within one round of the in-flight streams."""
    events = json.loads((FIXTURES / fixture).read_text())
    flow, scheduler, stats = _replay(events)

    assert scheduler._stream_states == {}
    assert not scheduler._pending_messages
    assert not any(message.type == "error" for message in stats.messages)
    request_ids = set(stats.expected)
    results = [m.request_id for m in stats.messages if m.type == "result"]
    assert sorted(results) == sorted(request_ids)
    last_type = {m.request_id: m.type for m in stats.messages}
    assert set(last_type.values()) == {"result"}

    remaining = {tuple(tokens): rid for rid, tokens in stats.expected.items()}
    for tokens in scheduler._vocoder.leftover_tokens:
        remaining.pop(tuple(tokens))
    assert remaining == {}

    samples = {rid: 0 for rid in request_ids}
    for message in stats.messages:
        if message.type == "stream":
            samples[message.request_id] += _waveform(message.data).shape[0]
    assert samples == {
        rid: len(tokens) * TOKEN_MEL_RATIO * SAMPLES_PER_FRAME
        for rid, tokens in stats.expected.items()
    }

    round_s = (stats.max_inflight + 1) * STEP_COST_S
    for request_id, (ready_at, slack) in stats.final_ready.items():
        waited = stats.result_at[request_id] - ready_at
        assert waited <= max(slack, 0.0) + round_s + 1e-9, request_id

    assert any(call["x"].shape[0] >= 4 for call in flow.decoder.estimator.calls)
