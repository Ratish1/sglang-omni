# SPDX-License-Identifier: Apache-2.0
"""Omni's served audio encoder against the HF audio encoder in FP32, with HF's own BF16 copy as the floor.

Omni runs the HF module with host-side attention splits and, as served, replays the layer stack from CUDA graphs on
packed attention. The check covers the graph path and the eager fallback, single requests and the executor's padded
batches, on stage 6's clips and on lengths at the chunk, window and graph bucket edges.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import pytest
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration
from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import (
    Qwen3OmniMoeProcessor,
)

from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from tests.test_model.qwen3_omni_component_accuracy.comparison import (
    CaseRecord,
    deviation,
    floor_violations,
    parameter_mismatches,
    reference_copy,
)
from tests.test_model.qwen3_omni_component_accuracy.stage6_requests import AudioQuestion

pytestmark = [pytest.mark.benchmark, pytest.mark.accelerator]

SAMPLE_RATE = 16000
EDGE_FRAMES = (1, 7, 99, 100, 101, 799, 800, 801, 1637, 3000, 6000, 33000)
BATCH_SIZE = 8


def reference_embeds(
    tower: torch.nn.Module, features: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """HF's get_audio_features (modeling_qwen3_omni_moe.py:1976-1983) on the tower alone."""
    lengths = mask.sum(dim=1)
    packed = features.permute(0, 2, 1)[mask.bool()].permute(1, 0)
    with torch.inference_mode():
        return tower(packed.to(tower.dtype), feature_lens=lengths).last_hidden_state


def omni_embeds(
    encoder: Qwen3OmniAudioEncoder, features: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    with torch.inference_mode():
        return encoder(
            input_features=features,
            feature_attention_mask=mask,
            audio_feature_lengths=mask.sum(dim=1),
        )["audio_embeds"]


def padded_batch(
    items: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    longest = max(features.shape[-1] for features, _ in items)
    features = torch.cat(
        [
            torch.nn.functional.pad(
                item_features, (0, longest - item_features.shape[-1])
            )
            for item_features, _ in items
        ]
    )
    mask = torch.cat(
        [
            torch.nn.functional.pad(item_mask, (0, longest - item_mask.shape[-1]))
            for _, item_mask in items
        ]
    )
    return features, mask


def test_audio_encoder_follows_reference(
    model_path: str,
    checkpoint_dir: str,
    audio_questions: list[AudioQuestion],
    reference_model: Qwen3OmniMoeForConditionalGeneration,
    component_device: str,
    report_dir: Path | None,
) -> None:
    extractor = Qwen3OmniMoeProcessor.from_pretrained(model_path).feature_extractor
    clips = []
    for question in audio_questions:
        waveform, _ = librosa.load(question.audio_path, sr=SAMPLE_RATE)
        extracted = extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        clips.append(
            (
                question.sample_id,
                extracted["input_features"],
                extracted["attention_mask"],
            )
        )
    joined = torch.cat([features[0] for _, features, _ in clips], dim=-1)
    edges = [
        (
            f"edge_{frames}_frames",
            joined[:, :frames].unsqueeze(0),
            torch.ones(1, frames, dtype=torch.long),
        )
        for frames in EDGE_FRAMES
    ]

    truth = reference_copy(
        reference_model.thinker.audio_tower,
        device=component_device,
        dtype=torch.float32,
    )
    floor = reference_copy(
        reference_model.thinker.audio_tower,
        device=component_device,
        dtype=torch.bfloat16,
    )
    served = Qwen3OmniAudioEncoder(
        checkpoint_dir, device=component_device, enable_layer_cuda_graph=True
    )
    eager = Qwen3OmniAudioEncoder(
        checkpoint_dir, device=component_device, enable_layer_cuda_graph=False
    )
    assert (
        served.layer_graph_runner is not None and served.layer_graph_runner.has_graphs
    )
    served_state = {
        name.replace("layers.layers.", "layers."): tensor
        for name, tensor in served.audio_tower.state_dict().items()
    }
    mismatches = parameter_mismatches(served_state, floor.state_dict())
    assert not mismatches, f"omni audio tower weights differ: {mismatches[:10]}"

    records: list[CaseRecord] = []
    for path_name, encoder in (("graph", served), ("eager", eager)):
        for case, features, mask in [*clips, *edges]:
            reference = reference_embeds(
                truth, features.to(component_device), mask.to(component_device)
            )
            records.append(
                CaseRecord(
                    case=f"{path_name}/{case}",
                    omni=deviation(omni_embeds(encoder, features, mask), reference),
                    floor=deviation(
                        reference_embeds(
                            floor,
                            features.to(component_device),
                            mask.to(component_device),
                        ),
                        reference,
                    ),
                )
            )
        for start in range(0, len(clips), BATCH_SIZE):
            group = clips[start : start + BATCH_SIZE]
            features, mask = padded_batch(
                [(item_features, item_mask) for _, item_features, item_mask in group]
            )
            batched = omni_embeds(encoder, features, mask)
            lengths = served.downsample_lengths(mask.sum(dim=1)).tolist()
            for (case, item_features, item_mask), rows in zip(
                group, batched.split(lengths)
            ):
                reference = reference_embeds(
                    truth,
                    item_features.to(component_device),
                    item_mask.to(component_device),
                )
                records.append(
                    CaseRecord(
                        case=f"{path_name}/batch{start // BATCH_SIZE}/{case}",
                        omni=deviation(rows, reference),
                        floor=deviation(
                            reference_embeds(
                                floor,
                                item_features.to(component_device),
                                item_mask.to(component_device),
                            ),
                            reference,
                        ),
                    )
                )

    if report_dir is not None:
        (report_dir / "audio_encoder.json").write_text(
            json.dumps([record.as_json() for record in records], indent=1)
        )
    else:
        pass
    ratios = sorted(record.floor_ratio for record in records)
    print(
        f"\n[audio_encoder] {len(records)} cases, floor ratio median {ratios[len(ratios) // 2]:.2f} "
        f"max {ratios[-1]:.2f}"
    )
    violations = floor_violations(records)
    assert not violations, "audio encoder strays past the BF16 floor:\n" + "\n".join(
        violations
    )
