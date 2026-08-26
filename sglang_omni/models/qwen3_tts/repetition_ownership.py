# SPDX-License-Identifier: Apache-2.0
"""Diagnostic routing for Qwen3-TTS repetition-penalty ownership."""

from __future__ import annotations

import os

REPETITION_PENALTY_OWNER_ENV = "SGLANG_OMNI_QWEN3_TTS_REPETITION_PENALTY_OWNER"
REPETITION_PENALTY_OWNERS = frozenset({"sglang", "qwen", "double"})


def repetition_penalty_owner() -> str:
    """Resolve the server-wide diagnostic owner, defaulting to clean behavior."""

    owner = os.environ.get(REPETITION_PENALTY_OWNER_ENV, "sglang").strip().lower()
    if owner not in REPETITION_PENALTY_OWNERS:
        choices = ", ".join(sorted(REPETITION_PENALTY_OWNERS))
        raise ValueError(
            f"{REPETITION_PENALTY_OWNER_ENV} must be one of {choices}; got {owner!r}"
        )
    return owner


def route_repetition_penalty(
    public_penalty: float,
    *,
    owner: str | None = None,
) -> tuple[str, float, float]:
    """Return ``(owner, qwen_penalty, sglang_penalty)`` for one request."""

    resolved_owner = repetition_penalty_owner() if owner is None else owner
    if resolved_owner not in REPETITION_PENALTY_OWNERS:
        choices = ", ".join(sorted(REPETITION_PENALTY_OWNERS))
        raise ValueError(f"owner must be one of {choices}; got {resolved_owner!r}")

    penalty = float(public_penalty)
    qwen_penalty = penalty if resolved_owner in {"qwen", "double"} else 1.0
    sglang_penalty = penalty if resolved_owner in {"sglang", "double"} else 1.0
    return resolved_owner, qwen_penalty, sglang_penalty


__all__ = [
    "REPETITION_PENALTY_OWNER_ENV",
    "REPETITION_PENALTY_OWNERS",
    "repetition_penalty_owner",
    "route_repetition_penalty",
]
