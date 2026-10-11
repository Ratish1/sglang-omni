# SPDX-License-Identifier: Apache-2.0
"""Omni's image and video preprocessing against Qwen's reference loader and HF's processor on CI requests.

The model card prepares media with qwen_omni_utils.process_mm_info at its defaults: images and video frames are
resized to multiples of 28 there (fetch_image, fetch_video) before the HF processor resizes them again, and video is
sampled at the request's fps. Images come from stage 3's mmmu-ci-50 with its prompt, videos from the VideoMME stage's
videomme-ci-50 at its fps 2 and 128 frames. Prompt ids, grids and pixels are compared.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from benchmarks.dataset.mmmu import load_mmmu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.videomme import load_videomme_samples
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.test_model.test_qwen3_omni_mmmu_talker_ci import MMMU_TTS_PROMPT

pytestmark = pytest.mark.benchmark

IMAGE_SAMPLES = 20
VIDEO_SAMPLES = 6
VIDEO_FPS = 2
VIDEO_MAX_FRAMES = 128


def omni_inputs(
    preprocessor: Qwen3OmniPreprocessor, request_id: str, inputs: dict[str, object]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=inputs, params={"max_new_tokens": 256}),
        data=None,
    )
    state = asyncio.run(preprocessor(payload)).data
    return state["prompt"]["input_ids"], state["encoder_inputs"]["image_encoder"]


def compare(
    case: str,
    omni_ids: torch.Tensor,
    omni_media: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    pixels: str,
    grid: str,
) -> dict[str, object]:
    omni_pixels = omni_media[pixels].float()
    reference_pixels = reference[pixels].float()
    same_shape = omni_pixels.shape == reference_pixels.shape
    return {
        "case": case,
        "input_ids_equal": torch.equal(omni_ids, reference["input_ids"][0]),
        "tokens": [int(omni_ids.shape[0]), int(reference["input_ids"].shape[1])],
        "grid": [omni_media[grid].tolist(), reference[grid].tolist()],
        "pixels_max_abs": (
            (omni_pixels - reference_pixels).abs().max().item() if same_shape else None
        ),
        "pixels_relative_l2": (
            ((omni_pixels - reference_pixels).norm() / reference_pixels.norm()).item()
            if same_shape
            else None
        ),
    }


def test_vision_preprocessing_matches_reference(
    model_path: str, report_dir: Path | None
) -> None:
    qwen_omni_utils = pytest.importorskip("qwen_omni_utils")
    preprocessor = Qwen3OmniPreprocessor(model_path)
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    records = []
    for sample in load_mmmu_samples(
        max_samples=IMAGE_SAMPLES,
        repo_id=DATASETS["mmmu-ci-50"],
        instruction_override=MMMU_TTS_PROMPT,
    ):
        omni_ids, omni_media = omni_inputs(
            preprocessor,
            sample.sample_id,
            {
                "messages": [{"role": "user", "content": sample.prompt}],
                "images": list(sample.image_data_uris),
            },
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    *(
                        {"type": "image", "image": uri}
                        for uri in sample.image_data_uris
                    ),
                    {"type": "text", "text": sample.prompt},
                ],
            }
        ]
        _, images, _ = qwen_omni_utils.process_mm_info(
            conversation, use_audio_in_video=False
        )
        reference = processor(
            text=processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            ),
            images=images,
            return_tensors="pt",
            padding=True,
        )
        records.append(
            compare(
                f"image/{sample.sample_id}",
                omni_ids,
                omni_media,
                reference,
                "pixel_values",
                "image_grid_thw",
            )
        )
    for sample in load_videomme_samples(
        max_samples=VIDEO_SAMPLES, repo_id=DATASETS["videomme-ci-50"]
    ):
        omni_ids, omni_media = omni_inputs(
            preprocessor,
            sample.sample_id,
            {
                "messages": [{"role": "user", "content": sample.prompt}],
                "videos": [sample.video_path],
                "video_fps": VIDEO_FPS,
                "video_max_frames": VIDEO_MAX_FRAMES,
            },
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": sample.video_path,
                        "fps": VIDEO_FPS,
                        "max_frames": VIDEO_MAX_FRAMES,
                    },
                    {"type": "text", "text": sample.prompt},
                ],
            }
        ]
        _, _, videos, video_kwargs = qwen_omni_utils.process_mm_info(
            conversation, use_audio_in_video=False, return_video_kwargs=True
        )
        reference = processor(
            text=processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            ),
            videos=videos,
            fps=video_kwargs["fps"][0],
            do_sample_frames=False,
            return_tensors="pt",
            padding=True,
        )
        records.append(
            compare(
                f"video/{sample.sample_id}",
                omni_ids,
                omni_media,
                reference,
                "pixel_values_videos",
                "video_grid_thw",
            )
        )

    if report_dir is not None:
        (report_dir / "vision_preprocessing.json").write_text(
            json.dumps(records, indent=1)
        )
    else:
        pass
    print(
        "\n[vision_preprocessing] "
        + "\n".join(json.dumps(record) for record in records)
    )
    different = [
        record["case"]
        for record in records
        if not (record["input_ids_equal"] and record["pixels_max_abs"] == 0.0)
    ]
    assert (
        not different
    ), f"vision preprocessing differs from the reference: {different}"
