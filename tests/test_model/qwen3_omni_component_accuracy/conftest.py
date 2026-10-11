# SPDX-License-Identifier: Apache-2.0
"""Opt-in Qwen3-Omni component accuracy against the Hugging Face reference.

QWEN3_OMNI_ACCURACY_MODEL names the checkpoint (id or path) and enables the suite;
QWEN3_OMNI_ACCURACY_REPORT_DIR, when set, keeps every per-request number as JSON;
QWEN3_OMNI_ACCURACY_DEVICE is the card the component checks run on (cuda:0 by default).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeForConditionalGeneration

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
def component_device() -> str:
    return os.environ.get("QWEN3_OMNI_ACCURACY_DEVICE", "cuda:0")


@pytest.fixture(scope="session")
def checkpoint_dir(model_path: str) -> str:
    return snapshot_download(model_path)


@pytest.fixture(scope="session")
def reference_model(checkpoint_dir: str) -> Qwen3OmniMoeForConditionalGeneration:
    """The whole HF model in FP32 on the host, loaded by HF's own loader; each check copies one component to the
    card in FP32 (the truth) and in BF16 (the floor)."""
    return Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        checkpoint_dir,
        dtype=torch.float32,
        device_map="cpu",
        enable_audio_output=True,
        experts_implementation="eager",
    ).eval()


@pytest.fixture(scope="session")
def report_dir() -> Path | None:
    path = os.environ.get("QWEN3_OMNI_ACCURACY_REPORT_DIR")
    if path is None:
        return None
    else:
        report_path = Path(path)
        report_path.mkdir(parents=True, exist_ok=True)
        return report_path
