# SPDX-License-Identifier: Apache-2.0
"""Per-sample token logprob readouts in the video and MMSU benchmarks."""

from __future__ import annotations

import asyncio
import math
from typing import Any

from benchmarks.benchmarker.data import RequestResult
from benchmarks.dataset.mmsu import MmsuSample
from benchmarks.dataset.videomme import VideoMMESample
from benchmarks.tasks.audio_understanding import (
    _build_request_payload,
    _build_result_from_response,
    build_mmsu_results,
)
from benchmarks.tasks.token_logprobs import (
    extract_token_logprobs,
    summarize_token_logprobs,
)
from benchmarks.tasks.video_understanding import (
    _apply_chat_completion_response,
    build_videomme_result_records,
    make_video_send_fn,
)


def _entry(token: str, token_id: int, logprob: float, top=()) -> dict[str, Any]:
    return {
        "token": token,
        "token_id": token_id,
        "logprob": logprob,
        "top_logprobs": [
            {"token": t, "token_id": i, "logprob": lp} for t, i, lp in top
        ],
    }


ANSWER_D = [
    _entry("Answer", 1, -0.05, [("Answer", 1, -0.05), ("The", 2, -3.1)]),
    _entry(":", 3, -0.01, [(":", 3, -0.01), (".", 4, -5.0)]),
    _entry(" D", 5, -0.4, [(" D", 5, -0.4), (" A", 6, -1.1)]),
]


def _video_sample() -> VideoMMESample:
    return VideoMMESample(
        sample_id="001-1",
        video_path="v.mp4",
        question="q",
        options=["A. x", "B. y", "C. z", "D. w"],
        answer="D",
        all_choices=["A", "B", "C", "D"],
        index2ans={"A": "x", "B": "y", "C": "z", "D": "w"},
    )


def _mmsu_sample() -> MmsuSample:
    return MmsuSample(
        sample_id="s1",
        audio_path="a.wav",
        question="q",
        choices=["x", "y", "z", "w"],
        answer_text="w",
        answer_index=3,
        task_name="t",
        category="c",
        sub_category="",
        sub_sub_category="",
        linguistics_sub_discipline="",
    )


def _body(content: str, token_logprobs: list[dict[str, Any]] | None) -> dict:
    choice: dict[str, Any] = {"message": {"content": content}, "logprobs": None}
    if token_logprobs is not None:
        choice["logprobs"] = {"content": token_logprobs}
    return {
        "choices": [choice],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


class _Response:
    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        return self._body

    async def __aenter__(self) -> "_Response":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _Session:
    def __init__(self, body: dict) -> None:
        self._body = body
        self.payloads: list[dict] = []

    def post(self, url: str, json: dict) -> _Response:
        del url
        self.payloads.append(json)
        return _Response(self._body)


def test_summarize_finds_answer_position_and_margins() -> None:
    summary = summarize_token_logprobs(ANSWER_D, "D")

    assert summary["answer_token_index"] == 2
    assert math.isclose(summary["answer_logprob"], -0.4)
    assert math.isclose(summary["answer_margin"], 0.7)
    assert summary["min_margin_index"] == 2
    assert math.isclose(summary["min_margin"], 0.7)


def test_summarize_without_top_k_has_no_margins() -> None:
    summary = summarize_token_logprobs([_entry(" D", 5, -0.4)], "D")

    assert summary == {
        "answer_token_index": 0,
        "answer_logprob": -0.4,
        "answer_margin": None,
        "min_margin": None,
        "min_margin_index": None,
    }


def test_summarize_margin_does_not_depend_on_top_order() -> None:
    unsorted = [_entry(" D", 5, -0.4, [(" A", 6, -1.1), (" D", 5, -0.4)])]

    summary = summarize_token_logprobs(unsorted, "D")

    assert math.isclose(summary["answer_margin"], 0.7)


def test_video_records_skip_answer_margin_for_parse_fallback() -> None:
    result = RequestResult(request_id="001-1")
    assert _apply_chat_completion_response(
        result,
        _body("no letter here", ANSWER_D),
        audio_output_dir=None,
        sample_id="001-1",
    )

    (record,) = build_videomme_result_records([_video_sample()], [result])

    assert record["is_mc_fallback"] is True
    assert record["answer_token_index"] is None
    assert record["answer_margin"] is None
    assert record["min_margin_index"] == 2


def test_summarize_without_answer_token() -> None:
    summary = summarize_token_logprobs(ANSWER_D, "B")

    assert summary["answer_token_index"] is None
    assert summary["answer_logprob"] is None
    assert summary["answer_margin"] is None
    assert summary["min_margin_index"] == 2


def test_extract_token_logprobs_reads_first_choice() -> None:
    assert extract_token_logprobs(_body("D", ANSWER_D)) == ANSWER_D
    assert extract_token_logprobs(_body("D", None)) is None
    assert extract_token_logprobs({}) is None


def test_video_send_fn_requests_logprobs_and_keeps_them() -> None:
    session = _Session(_body("Answer: D", ANSWER_D))
    send_fn = make_video_send_fn("m", "http://x", top_logprobs=5)

    result = asyncio.run(send_fn(session, _video_sample()))

    assert session.payloads[0]["logprobs"] is True
    assert session.payloads[0]["top_logprobs"] == 5
    assert result.is_success
    assert result.token_logprobs == ANSWER_D


def test_video_send_fn_omits_logprobs_by_default() -> None:
    session = _Session(_body("Answer: D", None))
    send_fn = make_video_send_fn("m", "http://x")

    result = asyncio.run(send_fn(session, _video_sample()))

    assert "logprobs" not in session.payloads[0]
    assert "top_logprobs" not in session.payloads[0]
    assert result.token_logprobs is None


def test_video_records_carry_margins() -> None:
    result = RequestResult(request_id="001-1")
    assert _apply_chat_completion_response(
        result, _body("Answer: D", ANSWER_D), audio_output_dir=None, sample_id="001-1"
    )

    (record,) = build_videomme_result_records([_video_sample()], [result])

    assert record["predicted"] == "D"
    assert record["is_correct"] is True
    assert record["answer_token_index"] == 2
    assert math.isclose(record["answer_logprob"], -0.4)
    assert math.isclose(record["answer_margin"], 0.7)
    assert record["min_margin_index"] == 2
    assert record["token_logprobs"] == ANSWER_D


def test_video_records_without_logprobs_keep_none_fields() -> None:
    result = RequestResult(request_id="001-1")
    assert _apply_chat_completion_response(
        result, _body("Answer: D", None), audio_output_dir=None, sample_id="001-1"
    )

    (record,) = build_videomme_result_records([_video_sample()], [result])

    assert record["predicted"] == "D"
    assert record["answer_token_index"] is None
    assert record["answer_margin"] is None
    assert record["min_margin"] is None
    assert record["token_logprobs"] is None


def test_mmsu_results_carry_margins() -> None:
    entry = _entry("D", 5, -0.2, [("D", 5, -0.2), ("B", 6, -1.7)])
    result = _build_result_from_response(
        RequestResult(request_id="s1"),
        _body("D", [entry]),
        audio_mode=False,
        sample_id="s1",
        save_audio_dir=None,
    )

    (mmsu,) = build_mmsu_results([result], [_mmsu_sample()])

    assert mmsu.predicted_choice == "D"
    assert mmsu.is_correct is True
    assert mmsu.answer_token_index == 0
    assert math.isclose(mmsu.answer_logprob, -0.2)
    assert math.isclose(mmsu.answer_margin, 1.5)
    assert mmsu.token_logprobs == [entry]


def test_mmsu_results_without_logprobs_keep_none_fields() -> None:
    result = _build_result_from_response(
        RequestResult(request_id="s1"),
        _body("D", None),
        audio_mode=False,
        sample_id="s1",
        save_audio_dir=None,
    )

    (mmsu,) = build_mmsu_results([result], [_mmsu_sample()])

    assert mmsu.predicted_choice == "D"
    assert mmsu.answer_margin is None
    assert mmsu.token_logprobs is None


def test_mmsu_payload_asks_for_logprobs_only_when_configured() -> None:
    common = dict(
        model_name="m", prompt="p", modalities=["text"], max_tokens=32, temperature=0.0
    )

    with_k = _build_request_payload(_mmsu_sample(), top_logprobs=5, **common)
    without = _build_request_payload(_mmsu_sample(), **common)

    assert with_k["logprobs"] is True
    assert with_k["top_logprobs"] == 5
    assert "logprobs" not in without
    assert "top_logprobs" not in without
