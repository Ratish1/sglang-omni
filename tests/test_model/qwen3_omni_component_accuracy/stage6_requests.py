# SPDX-License-Identifier: Apache-2.0
"""CI stage 6's MMSU requests, built by the benchmark's own code, as the accuracy suite's real inputs."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.tasks.audio_understanding import _build_question_text
from tests.test_model.test_qwen3_omni_mmsu_talker_ci import MAX_SAMPLES, MMSU_TTS_PROMPT


@dataclass(frozen=True, kw_only=True)
class AudioQuestion:
    """One user turn: an audio file and the question text the benchmark sends with it."""

    sample_id: str
    audio_path: str
    question_text: str


def load_audio_questions() -> list[AudioQuestion]:
    return [
        AudioQuestion(
            sample_id=sample.sample_id,
            audio_path=sample.audio_path,
            question_text=_build_question_text(sample, MMSU_TTS_PROMPT),
        )
        for sample in load_mmsu_samples(
            max_samples=MAX_SAMPLES, repo_id=DATASETS["mmsu-ci-2000"]
        )
    ]
