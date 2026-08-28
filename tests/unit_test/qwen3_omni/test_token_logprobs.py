# SPDX-License-Identifier: Apache-2.0
"""Token logprobs with decoded text, built by the decode stage."""

from __future__ import annotations

import pytest

from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
    StreamingDetokenizeScheduler,
    build_token_logprobs,
)
from sglang_omni.proto import OmniRequest, StagePayload


class _Tokenizer:
    def __init__(self, vocab: dict[int, str], eos_token_id: int | None = None):
        self._vocab = vocab
        self.eos_token_id = eos_token_id

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return "".join(
            self._vocab[tid]
            for tid in ids
            if not (skip_special_tokens and tid == self.eos_token_id)
        )


TOP = [[[-0.1, 1], [-2.5, 3]], [[-0.7, 2], [-0.9, 1]]]


def test_build_token_logprobs_decodes_sampled_and_top_tokens() -> None:
    out = build_token_logprobs(
        [[-0.1, 1], [-0.7, 2]],
        TOP,
        _Tokenizer({1: "A", 2: "<|im_end|>", 3: "B"}),
    )

    assert out == [
        {
            "token": "A",
            "token_id": 1,
            "logprob": -0.1,
            "top_logprobs": [
                {"token": "A", "token_id": 1, "logprob": -0.1},
                {"token": "B", "token_id": 3, "logprob": -2.5},
            ],
        },
        {
            "token": "<|im_end|>",
            "token_id": 2,
            "logprob": -0.7,
            "top_logprobs": [
                {"token": "<|im_end|>", "token_id": 2, "logprob": -0.7},
                {"token": "A", "token_id": 1, "logprob": -0.9},
            ],
        },
    ]


def test_build_token_logprobs_without_top_k() -> None:
    out = build_token_logprobs([[-0.1, 1]], [], _Tokenizer({1: "A"}))

    assert out == [{"token": "A", "token_id": 1, "logprob": -0.1, "top_logprobs": []}]


def test_build_token_logprobs_rejects_misaligned_steps() -> None:
    with pytest.raises(ValueError, match="2 steps for 1 sampled tokens"):
        build_token_logprobs(
            [[-0.1, 1]], [[[-0.1, 1]], [[-0.2, 1]]], _Tokenizer({1: "A"})
        )


def _payload(params: dict) -> StagePayload:
    return StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params=params),
        data={
            "engine_outputs": {
                "thinker": {
                    "output_ids": [1, 2],
                    "step": 2,
                    "is_final": True,
                    "extra_model_outputs": {},
                    "finish_reason": "stop",
                    "output_token_logprobs": [[-0.1, 1], [-0.7, 2]],
                    "output_top_logprobs": TOP,
                }
            },
            "thinker_out": None,
            "prompt": {"input_ids": []},
            "stream_state": {},
        },
    )


def test_decode_result_carries_token_logprobs_when_requested() -> None:
    sched = StreamingDetokenizeScheduler(
        tokenizer=_Tokenizer({1: "A", 2: "<|im_end|>", 3: "B"}, eos_token_id=2),
        eos_token_id=2,
    )

    result = sched._build_result(
        _payload(
            {
                "stream": False,
                "return_logprob": True,
                "top_logprobs_num": 2,
                "return_token_logprobs": True,
            }
        )
    )

    assert result["text"] == "A"
    assert result["output_token_logprobs"] == [[-0.1, 1], [-0.7, 2]]
    assert [item["token"] for item in result["token_logprobs"]] == ["A", "<|im_end|>"]
    assert [
        [top["token_id"] for top in item["top_logprobs"]]
        for item in result["token_logprobs"]
    ] == [[1, 3], [2, 1]]


def test_decode_result_omits_token_logprobs_unless_asked() -> None:
    sched = StreamingDetokenizeScheduler(
        tokenizer=_Tokenizer({1: "A", 2: "<|im_end|>"}, eos_token_id=2),
        eos_token_id=2,
    )

    result = sched._build_result(_payload({"stream": False, "return_logprob": True}))

    assert result["output_token_logprobs"] == [[-0.1, 1], [-0.7, 2]]
    assert "token_logprobs" not in result
