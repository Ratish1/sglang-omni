# SPDX-License-Identifier: Apache-2.0
"""Omni's thinker MRoPE positions against HF's get_rope_index on the same ids and grids.

Omni computes positions on the host with a vectorized port (mrope_positions.py, through compute_mrope_positions);
HF computes them in the thinker (modeling_qwen3_omni_moe.py:220-460). Both read the same processor outputs here, so
the positions and the rope delta must be equal for audio, images, video, audio inside video and a multi-turn mix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.mrope_positions import talker_can_use_linear_mrope
from sglang_omni.models.qwen3_omni.request_builders import compute_mrope_positions

pytestmark = pytest.mark.benchmark

SAMPLE_RATE = 16000
SEED = 20261011


def chat(*turns: tuple[str, str]) -> str:
    return "".join(
        f"<|im_start|>{role}\n{content}<|im_end|>\n" for role, content in turns
    ) + ("<|im_start|>assistant\n")


def talker_input_ids(input_ids: torch.Tensor, config) -> torch.Tensor:
    """The talker's prompt as HF's generate builds it: the user turns, then the assistant tail of tts_pad ids."""
    starts = (input_ids == config.im_start_token_id).nonzero().flatten().tolist()
    ends = [*starts[1:], input_ids.shape[0]]
    user_turns = [
        input_ids[start:end]
        for start, end in zip(starts, ends)
        if input_ids[start + 1] == config.user_token_id
    ]
    tail = torch.full((9,), config.tts_pad_token_id, dtype=input_ids.dtype)
    return torch.cat([*user_turns, tail])


def test_mrope_positions_match_reference(
    model_path: str,
    reference_model: Qwen3OmniMoeForConditionalGeneration,
    report_dir: Path | None,
) -> None:
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    audio = (
        processor.audio_bos_token + processor.audio_token + processor.audio_eos_token
    )
    image = (
        processor.vision_bos_token + processor.image_token + processor.vision_eos_token
    )
    video = (
        processor.vision_bos_token + processor.video_token + processor.vision_eos_token
    )
    generator = np.random.default_rng(SEED)
    waveforms = [
        generator.uniform(-0.1, 0.1, seconds * SAMPLE_RATE).astype(np.float32)
        for seconds in (3, 7)
    ]
    images = [
        generator.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for height, width in ((336, 448), (700, 1036))
    ]
    frames = generator.integers(0, 256, (16, 360, 640, 3), dtype=np.uint8)
    cases = {
        "audio": (
            chat(("user", audio + "What is said?")),
            {"audio": waveforms[:1]},
            False,
        ),
        "image": (
            chat(("user", image + "Describe it.")),
            {"images": images[:1]},
            False,
        ),
        "two_images_and_audio": (
            chat(("user", image + image + audio + "Compare.")),
            {"images": images, "audio": waveforms[1:]},
            False,
        ),
        "video": (chat(("user", video + "What happens?")), {"videos": [frames]}, False),
        "audio_in_video": (
            chat(("user", video + "What happens?")),
            {"videos": [frames], "audio": waveforms[1:]},
            True,
        ),
        "multi_turn": (
            chat(
                ("system", "Be brief."),
                ("user", image + "What is it?"),
                ("assistant", "A pattern."),
                ("user", audio + "And this?"),
            ),
            {"images": images[:1], "audio": waveforms[:1]},
            False,
        ),
    }

    records = []
    thinker_config = reference_model.config.thinker_config
    for case, (text, media, use_audio_in_video) in cases.items():
        inputs = processor(
            text=text,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
            **media,
        )
        audio_lengths = (
            inputs["feature_attention_mask"].sum(dim=1)
            if "feature_attention_mask" in inputs
            else None
        )
        reference_positions, reference_delta = reference_model.thinker.get_rope_index(
            inputs["input_ids"],
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            inputs["attention_mask"],
            use_audio_in_video,
            audio_lengths,
            inputs.get("video_second_per_grid"),
        )
        omni_positions, omni_delta = compute_mrope_positions(
            inputs["input_ids"][0],
            {
                "image_grid_thw": inputs.get("image_grid_thw"),
                "video_grid_thw": inputs.get("video_grid_thw"),
                "video_second_per_grid": inputs.get("video_second_per_grid"),
                "use_audio_in_video": use_audio_in_video,
                "audio_feature_lengths": audio_lengths,
            },
            thinker_config,
        )
        talker_ids = talker_input_ids(inputs["input_ids"][0], reference_model.config)
        has_grid = (
            inputs.get("image_grid_thw") is not None
            or inputs.get("video_grid_thw") is not None
        )
        talker_inputs = {
            "image_grid_thw": inputs.get("image_grid_thw"),
            "video_grid_thw": inputs.get("video_grid_thw"),
            "video_second_per_grid": inputs.get("video_second_per_grid"),
            "use_audio_in_video": use_audio_in_video,
            "audio_feature_lengths": audio_lengths,
        }
        # note (ratish): HF's talker reads audio_config, which its config lacks, on a prompt with a grid and audio.
        if has_grid and audio_lengths is None:
            talker_reference, talker_reference_delta = (
                reference_model.talker.get_rope_index(
                    talker_ids[None],
                    inputs.get("image_grid_thw"),
                    inputs.get("video_grid_thw"),
                    torch.ones_like(talker_ids)[None],
                    use_audio_in_video,
                    audio_lengths,
                    inputs.get("video_second_per_grid"),
                )
            )
            talker_omni, talker_omni_delta = compute_mrope_positions(
                talker_ids, talker_inputs, thinker_config
            )
            records.append(
                {
                    "case": f"talker/{case}",
                    "tokens": int(talker_ids.shape[0]),
                    "positions_equal": torch.equal(
                        talker_omni.float(), talker_reference[:, 0, :].float()
                    ),
                    "positions_max_abs": (
                        talker_omni.float() - talker_reference[:, 0, :].float()
                    )
                    .abs()
                    .max()
                    .item(),
                    "delta": [
                        int(talker_omni_delta.flatten()[0]),
                        int(talker_reference_delta.flatten()[0]),
                    ],
                }
            )
        elif not has_grid:
            assert talker_can_use_linear_mrope(
                talker_ids, talker_inputs, thinker_config
            ), f"{case}: HF's talker uses linear positions without a grid"
        else:
            pass
        reference_positions = reference_positions[:, 0, :]
        records.append(
            {
                "case": case,
                "tokens": int(inputs["input_ids"].shape[1]),
                "positions_equal": torch.equal(
                    omni_positions.float(), reference_positions.float()
                ),
                "positions_max_abs": (
                    omni_positions.float() - reference_positions.float()
                )
                .abs()
                .max()
                .item(),
                "delta": [
                    int(omni_delta.flatten()[0]),
                    int(reference_delta.flatten()[0]),
                ],
            }
        )

    if report_dir is not None:
        (report_dir / "mrope_positions.json").write_text(json.dumps(records, indent=1))
    else:
        pass
    print("\n[mrope_positions] " + json.dumps(records))
    different = [
        record
        for record in records
        if not record["positions_equal"] or record["delta"][0] != record["delta"][1]
    ]
    assert not different, f"omni's MRoPE positions differ from HF's: {different}"
