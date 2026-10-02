# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def sglang_req(request_id: str, *, output_len: int = 3, finished: bool = False):
    return SimpleNamespace(
        rid=request_id,
        output_ids=list(range(output_len)),
        sampling_params=SimpleNamespace(
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            min_new_tokens=0,
        ),
        custom_logit_processor=None,
        is_retracted=False,
        finished=lambda: finished,
    )


def make_runner(model: SimpleNamespace) -> QwenTalkerModelRunner:
    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = model
    runner.feedback_enabled = True
    runner.code2wav_target = "code2wav"
    runner.code2wav_in_process = False
    runner.codec_coalesce_frames = 0
    runner.codec_coalesce_early_frames = 0
    runner.codec_coalesce_first_frames = 0
    runner.inflight_steps = {}
    runner.outbox = SimpleNamespace(sent=[])
    runner.outbox.put = runner.outbox.sent.append
    return runner


def step_model(batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        sampled_token_ids=torch.arange(batch_size, dtype=torch.long) + 10,
        output_codes=torch.arange(batch_size * 2, dtype=torch.long).reshape(
            batch_size, 2
        ),
        output_embeds=torch.arange(batch_size * 4, dtype=torch.float32).reshape(
            batch_size, 4
        ),
    )


def scheduler_requests(reqs: list[SimpleNamespace]) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            data=SimpleNamespace(
                req=req,
                pending_feedback_queue=deque(),
                stage_payload=None,
            )
        )
        for req in reqs
    ]


def test_launch_queues_feedback_and_resolve_sends_only_live_rows() -> None:
    model = step_model(2)
    runner = make_runner(model)
    reqs = [sglang_req("live"), sglang_req("done", finished=True)]
    requests = scheduler_requests(reqs)
    result = SimpleNamespace(next_token_ids=None)

    codes = runner.post_decode_launch(result, None, requests)
    launch_codes = model.output_codes.clone()
    model.output_codes += 100

    assert runner.outbox.sent == []
    assert [len(r.data.pending_feedback_queue) for r in requests] == [1, 1]
    assert runner.inflight_steps == {"live": 1, "done": 1}
    assert result.next_token_ids.tolist() == [10, 11]

    runner.post_decode_resolve(
        codes, result, None, SimpleNamespace(reqs=reqs), requests
    )

    assert [m.request_id for m in runner.outbox.sent] == ["live"]
    assert torch.equal(runner.outbox.sent[0].data, launch_codes[0])
    assert runner.inflight_steps == {}


def test_lookahead_needs_the_same_rows_one_sampled_token_further() -> None:
    model = SimpleNamespace(decode_prep_rids=["a", "b"], decode_prep_out_lens=[3, 3])
    model.can_reuse_decode_buffers = Qwen3OmniTalker.can_reuse_decode_buffers.__get__(
        model
    )
    runner = make_runner(model)
    batch = SimpleNamespace(reqs=[sglang_req("a"), sglang_req("b")])

    assert not runner.lookahead_eligible(batch)
    runner.inflight_steps = {"a": 1, "b": 1}
    assert runner.lookahead_eligible(batch)
    assert not runner.lookahead_eligible(SimpleNamespace(reqs=batch.reqs[::-1]))
    model.decode_prep_rids = None
    assert not runner.lookahead_eligible(batch)


@pytest.mark.parametrize(
    ("waiting", "batch_is_full", "decode_fits", "drains"),
    [
        (["req"], False, True, True),
        (["req"], True, True, False),
        ([], False, False, True),
        ([], False, True, False),
    ],
)
def test_talker_resolves_the_launched_step_before_a_prefill_or_retract(
    monkeypatch: pytest.MonkeyPatch,
    waiting: list[str],
    batch_is_full: bool,
    decode_fits: bool,
    drains: bool,
) -> None:
    monkeypatch.setattr(OmniScheduler, "get_next_batch_to_run", lambda self: None)
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler.async_pending = object()
    scheduler.waiting_queue = waiting
    scheduler.running_batch = SimpleNamespace(
        batch_is_full=batch_is_full, check_decode_mem=lambda: decode_fits
    )
    resolved: list[bool] = []
    scheduler.resolve_pending_async = lambda: resolved.append(True)

    assert scheduler.get_next_batch_to_run() is None
    assert resolved == ([True] if drains else [])
