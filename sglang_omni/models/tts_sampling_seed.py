# SPDX-License-Identifier: Apache-2.0
"""Shared TTS sampling seed helpers."""

from __future__ import annotations

import hashlib
import os
from typing import Any

TTS_SAMPLING_SEED_MASK = 0x7FFFFFFF


def new_tts_sampling_seed() -> int:
    return int.from_bytes(os.urandom(4), "little") & TTS_SAMPLING_SEED_MASK


def normalize_tts_sampling_seed(seed: Any, *, owner: str) -> int:
    if isinstance(seed, bool):
        raise ValueError(f"{owner} seed must be an integer")
    if isinstance(seed, float) and not seed.is_integer():
        raise ValueError(f"{owner} seed must be an integer")
    try:
        normalized = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{owner} seed must be an integer") from exc
    return normalized & TTS_SAMPLING_SEED_MASK


def derive_tts_sampling_seed(namespace: str, seed: int, *labels: str) -> int:
    seed_material = ":".join((namespace, str(int(seed)), *labels))
    digest = hashlib.blake2b(seed_material.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & TTS_SAMPLING_SEED_MASK
