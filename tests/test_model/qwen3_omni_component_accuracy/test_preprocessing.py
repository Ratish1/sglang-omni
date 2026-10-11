# SPDX-License-Identifier: Apache-2.0
"""Omni's Qwen3-Omni preprocessing against Qwen's reference loader and HF's processor on CI stage 6's requests.

Both sides run the same HF processor, so prompt ids, mel features and masks must be identical; any difference
comes from decoding, resampling, message order or the template.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import librosa
import pytest
import soundfile
import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.test_model.qwen3_omni_component_accuracy.stage6_requests import AudioQuestion
from tests.test_model.test_qwen3_omni_mmsu_talker_ci import MAX_TOKENS

pytestmark = pytest.mark.benchmark

# note (ratish): Qwen's loader reads every clip with librosa.load(path, sr=16000), resampling with soxr_hq.
REFERENCE_SAMPLE_RATE = 16000


def test_preprocessing_matches_reference(
    model_path: str,
    audio_questions: list[AudioQuestion],
    report_dir: Path | None,
) -> None:
    preprocessor = Qwen3OmniPreprocessor(model_path)
    reference_processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    records = []
    for question in audio_questions:
        payload = StagePayload(
            request_id=question.sample_id,
            request=OmniRequest(
                inputs={
                    "messages": [{"role": "user", "content": question.question_text}],
                    "audios": [question.audio_path],
                },
                params={"max_new_tokens": MAX_TOKENS},
            ),
            data=None,
        )
        state = asyncio.run(preprocessor(payload)).data
        audio_inputs = state["encoder_inputs"]["audio_encoder"]
        waveform, _ = librosa.load(question.audio_path, sr=REFERENCE_SAMPLE_RATE)
        reference_prompt = reference_processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": question.audio_path},
                        {"type": "text", "text": question.question_text},
                    ],
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        reference = reference_processor(
            text=reference_prompt, audio=[waveform], return_tensors="pt", padding=True
        )
        omni_features = audio_inputs["input_features"].float()
        reference_features = reference["input_features"].float()
        shared_frames = min(omni_features.shape[-1], reference_features.shape[-1])
        feature_error = (
            omni_features[..., :shared_frames] - reference_features[..., :shared_frames]
        )
        records.append(
            {
                "sample_id": question.sample_id,
                "sample_rate": soundfile.info(question.audio_path).samplerate,
                "prompt_equal": state["prompt"]["prompt_text"] == reference_prompt,
                "input_ids_equal": torch.equal(
                    state["prompt"]["input_ids"], reference["input_ids"][0]
                ),
                "feature_frames": [
                    omni_features.shape[-1],
                    reference_features.shape[-1],
                ],
                "mask_equal": torch.equal(
                    audio_inputs["feature_attention_mask"],
                    reference["feature_attention_mask"],
                ),
                "features_max_abs": feature_error.abs().max().item(),
                "features_relative_l2": (
                    feature_error.norm()
                    / reference_features[..., :shared_frames].norm()
                ).item(),
            }
        )
    if report_dir is not None:
        (report_dir / "preprocessing.json").write_text(json.dumps(records, indent=1))
    else:
        pass
    different = [
        f"{record['sample_id']} ({record['sample_rate']} Hz): "
        f"features relative L2 {record['features_relative_l2']:.4f}"
        for record in records
        if not (
            record["prompt_equal"]
            and record["input_ids_equal"]
            and record["mask_equal"]
            and record["features_max_abs"] == 0.0
        )
    ]
    print(
        f"\n[preprocessing] {len(records) - len(different)} of {len(records)} identical"
    )
    assert not different, "preprocessing differs from the reference:\n" + "\n".join(
        different
    )
