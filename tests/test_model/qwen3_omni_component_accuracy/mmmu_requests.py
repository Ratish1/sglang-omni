# SPDX-License-Identifier: Apache-2.0
"""The MMMU talker stage's requests, built by the benchmark's own code, with HF's processor outputs for each."""

from __future__ import annotations

import base64
import io

from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from benchmarks.dataset.mmmu import MMMUSample, load_mmmu_samples
from benchmarks.dataset.prepare import DATASETS
from tests.test_model.test_qwen3_omni_mmmu_talker_ci import MAX_SAMPLES, MMMU_TTS_PROMPT


def load_image_questions() -> list[MMMUSample]:
    return load_mmmu_samples(
        max_samples=MAX_SAMPLES,
        repo_id=DATASETS["mmmu-ci-50"],
        instruction_override=MMMU_TTS_PROMPT,
    )


def reference_inputs(
    processor: Qwen3OmniMoeProcessor, sample: MMMUSample
) -> BatchFeature:
    """HF's processor on the request's decoded images, placed before its text."""
    images = [
        Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")
        for uri in sample.image_data_uris
    ]
    conversation = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in images),
                {"type": "text", "text": sample.prompt},
            ],
        }
    ]
    return processor(
        text=processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        ),
        images=images,
        return_tensors="pt",
        padding=True,
    )
