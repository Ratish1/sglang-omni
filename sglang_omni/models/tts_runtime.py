# SPDX-License-Identifier: Apache-2.0
"""Shared runtime helpers for TTS model components."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def run_bucketed_batch(
    items: Sequence[Any],
    *,
    bucket_key_fn: Callable[[Any], Any],
    single_fn: Callable[[Any], Any],
    batch_fn: Callable[[list[Any]], list[Any]],
    error_label: str,
) -> list[Any]:
    if not items:
        return []
    if len(items) == 1:
        return [single_fn(items[0])]

    buckets: dict[Any, list[int]] = {}
    for i, item in enumerate(items):
        buckets.setdefault(bucket_key_fn(item), []).append(i)

    results: list[Any | None] = [None] * len(items)
    for indices in buckets.values():
        if len(indices) == 1:
            results[indices[0]] = single_fn(items[indices[0]])
            continue
        batch_results = batch_fn([items[i] for i in indices])
        for idx, result in zip(indices, batch_results):
            results[idx] = result

    out: list[Any] = []
    for i, result in enumerate(results):
        if result is None:
            raise RuntimeError(f"{error_label} did not produce result for item {i}")
        out.append(result)
    return out


def require_batch_result_count(
    *,
    owner: str,
    result_label: str,
    actual: int,
    expected: int,
) -> None:
    if actual != expected:
        raise RuntimeError(
            f"{owner} returned {actual} {result_label} for {expected} requests"
        )


def build_tts_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    engine_time_s: float,
) -> dict[str, Any] | None:
    if not (prompt_tokens or completion_tokens or engine_time_s):
        return None
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if engine_time_s:
        usage["engine_time_s"] = round(engine_time_s, 6)
    return usage
