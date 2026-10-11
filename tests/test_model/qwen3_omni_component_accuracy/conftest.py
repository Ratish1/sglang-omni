# SPDX-License-Identifier: Apache-2.0
"""Opt-in Qwen3-Omni component accuracy against the Hugging Face reference.

QWEN3_OMNI_ACCURACY_MODEL names the checkpoint (id or path) and enables the suite;
QWEN3_OMNI_ACCURACY_REPORT_DIR, when set, keeps every per-request number as JSON.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.test_model.qwen3_omni_component_accuracy.stage6_requests import (
    AudioQuestion,
    load_audio_questions,
)


@pytest.fixture(scope="session")
def model_path() -> str:
    path = os.environ.get("QWEN3_OMNI_ACCURACY_MODEL")
    if path is None:
        pytest.skip(
            "Set QWEN3_OMNI_ACCURACY_MODEL to run the Qwen3-Omni accuracy suite"
        )
    else:
        pass
    return path


@pytest.fixture(scope="session")
def audio_questions() -> list[AudioQuestion]:
    return load_audio_questions()


@pytest.fixture(scope="session")
def report_dir() -> Path | None:
    path = os.environ.get("QWEN3_OMNI_ACCURACY_REPORT_DIR")
    if path is None:
        return None
    else:
        report_path = Path(path)
        report_path.mkdir(parents=True, exist_ok=True)
        return report_path
