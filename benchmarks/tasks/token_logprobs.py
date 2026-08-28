# SPDX-License-Identifier: Apache-2.0
"""Per-sample readouts of chat completion token logprobs."""

from __future__ import annotations

from typing import Any


def extract_token_logprobs(body: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The logprobs content list of the first choice, or None when absent."""
    choices = body.get("choices") or [{}]
    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    return content if isinstance(content, list) else None


def _margin(item: dict[str, Any]) -> float | None:
    top = item.get("top_logprobs") or []
    if len(top) < 2:
        return None
    best, second = sorted((float(t["logprob"]) for t in top), reverse=True)[:2]
    return best - second


def summarize_token_logprobs(
    token_logprobs: list[dict[str, Any]],
    answer_letter: str,
) -> dict[str, Any]:
    """Margins of the greedy path of one completion.

    answer_token_index is the last position whose token text, stripped, is
    answer_letter. answer_logprob is the sampled logprob there and
    answer_margin the gap between the two most likely tokens there.
    min_margin and min_margin_index give the smallest such gap over every
    position, the place where the completion is closest to taking another
    path. Margins need top_logprobs of at least 2 and are None otherwise.
    """
    answer_index: int | None = None
    if answer_letter:
        for index in range(len(token_logprobs) - 1, -1, -1):
            if token_logprobs[index]["token"].strip() == answer_letter:
                answer_index = index
                break
    defined = [
        (margin, index)
        for index, item in enumerate(token_logprobs)
        if (margin := _margin(item)) is not None
    ]
    min_margin, min_index = min(defined, default=(None, None))
    answer = token_logprobs[answer_index] if answer_index is not None else None
    return {
        "answer_token_index": answer_index,
        "answer_logprob": float(answer["logprob"]) if answer is not None else None,
        "answer_margin": _margin(answer) if answer is not None else None,
        "min_margin": min_margin,
        "min_margin_index": min_index,
    }
