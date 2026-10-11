# SPDX-License-Identifier: Apache-2.0
"""Omni's served vision encoder against the HF vision encoder in FP32, with HF's own BF16 copy as the floor.

Omni runs the HF module with the Transformers 5.6 positional interpolation, a Linear patch embedding and the
platform's joint rotary kernel. The check covers the merged embeddings and the three deepstack features for images
of several grids and for a video, and separately measures what the 5.6 interpolation alone changes in FP32.
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

from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.vision_compat import (
    Qwen3OmniMoeVisionEncoderCompat,
)
from tests.test_model.qwen3_omni_component_accuracy.comparison import (
    CaseRecord,
    deviation,
    floor_violations,
    parameter_mismatches,
    reference_copy,
)

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

IMAGE_SIZES = ((224, 224), (336, 448), (480, 640), (700, 1036), (1080, 1920))
VIDEO_FRAMES = 8
VIDEO_SIZE = (360, 640)
SEED = 20261011


def seeded_image(height: int, width: int, generator: torch.Generator) -> np.ndarray:
    coarse = torch.rand(1, 3, height // 32 + 1, width // 32 + 1, generator=generator)
    smooth = torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="bilinear", align_corners=False
    )
    detail = 0.1 * torch.rand(1, 3, height, width, generator=generator)
    image = (smooth + detail).clamp(0, 1)[0].permute(1, 2, 0)
    return (image * 255).to(torch.uint8).numpy()


def visual_outputs(
    visual: torch.nn.Module, pixels: torch.Tensor, grid: torch.Tensor
) -> list[torch.Tensor]:
    with torch.inference_mode():
        output = visual(pixels.to(visual.dtype), grid_thw=grid)
    return [output.pooler_output, *output.deepstack_features]


def test_vision_encoder_follows_reference(
    model_path: str,
    checkpoint_dir: str,
    reference_model: Qwen3OmniMoeForConditionalGeneration,
    component_device: str,
    report_dir: Path | None,
) -> None:
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    generator = torch.Generator().manual_seed(SEED)
    cases = []
    for height, width in IMAGE_SIZES:
        processed = processor.image_processor(
            images=[seeded_image(height, width, generator)], return_tensors="pt"
        )
        cases.append(
            (
                f"image_{height}x{width}",
                "image",
                processed["pixel_values"],
                processed["image_grid_thw"],
            )
        )
    frames = np.stack(
        [seeded_image(*VIDEO_SIZE, generator) for _ in range(VIDEO_FRAMES)]
    )
    processed = processor.video_processor(videos=[frames], return_tensors="pt")
    cases.append(
        (
            f"video_{VIDEO_FRAMES}x{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}",
            "video",
            processed["pixel_values_videos"],
            processed["video_grid_thw"],
        )
    )

    truth = reference_copy(
        reference_model.thinker.visual, device=component_device, dtype=torch.float32
    )
    floor = reference_copy(
        reference_model.thinker.visual, device=component_device, dtype=torch.bfloat16
    )
    legacy_interpolation = (
        Qwen3OmniMoeVisionEncoderCompat._from_config(  # noqa: leading-underscore  # Transformers construction API
            truth.config
        )
        .to(device=component_device, dtype=torch.float32)
        .eval()
    )
    legacy_interpolation.load_state_dict(truth.state_dict())
    served = Qwen3OmniImageEncoder(checkpoint_dir, device=component_device)

    served_state = dict(served.visual.state_dict())
    linear_weight = served_state.pop("patch_embed.linear.weight")
    served_state["patch_embed.proj.bias"] = served_state.pop("patch_embed.linear.bias")
    served_state["patch_embed.proj.weight"] = linear_weight.view(
        floor.patch_embed.proj.weight.shape
    )
    mismatches = parameter_mismatches(served_state, floor.state_dict())
    assert not mismatches, f"omni vision weights differ: {mismatches[:10]}"

    records: list[CaseRecord] = []
    interpolation: list[dict[str, object]] = []
    output_names = ("merged", "deepstack0", "deepstack1", "deepstack2")
    for case, modality, pixels, grid in cases:
        pixels = pixels.to(component_device)
        grid = grid.to(component_device)
        reference = visual_outputs(truth, pixels, grid)
        floor_outputs = visual_outputs(floor, pixels, grid)
        legacy_outputs = visual_outputs(legacy_interpolation, pixels, grid)
        with torch.inference_mode():
            if modality == "image":
                encoded = served(pixel_values=pixels, image_grid_thw=grid)
                omni_outputs = [
                    encoded["image_embeds"],
                    *encoded["deepstack_visual_embeds_image"],
                ]
            else:
                encoded = served(pixel_values_videos=pixels, video_grid_thw=grid)
                omni_outputs = [
                    encoded["video_embeds"],
                    *encoded["deepstack_visual_embeds_video"],
                ]
        for name, omni_output, floor_output, legacy_output, truth_output in zip(
            output_names, omni_outputs, floor_outputs, legacy_outputs, reference
        ):
            records.append(
                CaseRecord(
                    case=f"{case}/{name}",
                    omni=deviation(omni_output, truth_output),
                    floor=deviation(floor_output, truth_output),
                )
            )
            interpolation.append(
                {
                    "case": f"{case}/{name}",
                    "legacy_interpolation_fp32_relative_l2": deviation(
                        legacy_output, truth_output
                    ).relative_l2,
                }
            )

    if report_dir is not None:
        (report_dir / "vision_encoder.json").write_text(
            json.dumps(
                {
                    "records": [record.as_json() for record in records],
                    "interpolation": interpolation,
                },
                indent=1,
            )
        )
    else:
        pass
    print(
        "\n[vision_encoder] floor ratio per case: "
        + ", ".join(f"{record.case} {record.floor_ratio:.2f}" for record in records)
        + "\n[vision_encoder] 5.6 interpolation alone in FP32, relative L2: "
        + ", ".join(
            f"{entry['case']} {entry['legacy_interpolation_fp32_relative_l2']:.2e}"
            for entry in interpolation
        )
    )
    violations = floor_violations(records)
    assert not violations, "vision encoder strays past the BF16 floor:\n" + "\n".join(
        violations
    )
