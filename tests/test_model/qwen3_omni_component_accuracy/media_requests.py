# SPDX-License-Identifier: Apache-2.0
"""The MMMU and Video-MME talker stages' requests as the accuracy suite's real inputs to the served talker."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.videomme import load_videomme_samples
from tests.test_model.qwen3_omni_component_accuracy.mmmu_requests import (
    load_image_questions,
)
from tests.test_model.test_qwen3_omni_videomme_talker_ci import (
    MAX_SAMPLES as VIDEO_MAX_SAMPLES,
)
from tests.test_model.test_qwen3_omni_videomme_talker_ci import SHORT_ANSWER_PROMPT

VIDEO_FPS = 2
VIDEO_MAX_FRAMES = 128


@dataclass(frozen=True, kw_only=True)
class MediaQuestion:
    """One user turn with images or a video: the metadata fields a request carries them in, and its text."""

    sample_id: str
    question_text: str
    media: dict[str, object]


def load_image_media_questions() -> list[MediaQuestion]:
    return [
        MediaQuestion(
            sample_id=sample.sample_id,
            question_text=sample.prompt,
            media={"images": list(sample.image_data_uris)},
        )
        for sample in load_image_questions()
    ]


def load_video_media_questions() -> list[MediaQuestion]:
    return [
        MediaQuestion(
            sample_id=sample.sample_id,
            question_text=f"{sample.prompt}\n{SHORT_ANSWER_PROMPT}",
            media={
                "videos": [sample.video_path],
                "video_fps": VIDEO_FPS,
                "video_max_frames": VIDEO_MAX_FRAMES,
            },
        )
        for sample in load_videomme_samples(
            max_samples=VIDEO_MAX_SAMPLES, repo_id=DATASETS["videomme-ci-50"]
        )
    ]
