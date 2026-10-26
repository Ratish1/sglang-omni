"""Compare the merged position interpolation with its preceding arithmetic on CUDA."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)
from transformers.vision_utils import get_vision_interpolation_indices_and_weights

from sglang_omni.models.qwen3_omni.components.vision_compat import (
    Qwen3OmniMoeVisionEncoderCompat,
)
from tests.unit_test.qwen3_omni.test_vision_compat import (
    transformers_5_6_pos_embed_interpolate,
)


class LegacyVisionEncoder(Qwen3OmniMoeVisionEncoderCompat):
    def interpolate_position_embeddings(
        self, grid_thw: list[list[int]]
    ) -> torch.Tensor:
        return transformers_5_6_pos_embed_interpolate(self, grid_thw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    grid_cases = [[[1, height, 14]] for height in range(2, 130, 2)]
    grid_cases.extend([[[64, 28, 44]], [[1, 16, 24], [1, 32, 12]], [[1, 64, 98]]])
    failures: list[dict[str, int | str | list[list[int]] | float]] = []
    comparisons = 0
    hf_changed_elements = 0
    for dtype in (torch.bfloat16, torch.float32):
        encoder = Qwen3OmniMoeVisionEncoderCompat.__new__(
            Qwen3OmniMoeVisionEncoderCompat
        )
        torch.nn.Module.__init__(encoder)
        encoder.spatial_merge_size = 2
        encoder.num_grid_per_side = 48
        encoder.pos_embed = torch.nn.Embedding(
            48 * 48, 1152, device="cuda", dtype=dtype
        )
        for grid_thw in grid_cases:
            with torch.inference_mode():
                expected = transformers_5_6_pos_embed_interpolate(encoder, grid_thw)
                actual = encoder.interpolate_position_embeddings(grid_thw)
                different_elements = int((actual != expected).sum())
                comparisons += 1
                if different_elements:
                    failures.append(
                        {
                            "dtype": str(dtype),
                            "grid": grid_thw,
                            "different_elements": different_elements,
                            "max_absolute_error": float(
                                (actual - expected).abs().max()
                            ),
                        }
                    )
                else:
                    pass
                if dtype == torch.bfloat16 and grid_thw == [[1, 64, 98]]:
                    indices, weights = get_vision_interpolation_indices_and_weights(
                        torch.tensor(grid_thw, device="cuda"),
                        num_grid_per_side=48,
                        align_corners=True,
                        spatial_merge_size=2,
                    )
                    hf_output = (
                        (encoder.pos_embed(indices) * weights[:, :, None])
                        .sum(1)
                        .to(dtype)
                    )
                    hf_changed_elements = int((actual != hf_output).sum())
                else:
                    pass
        timings: dict[str, list[float]] = {"legacy": [], "merged": []}
        for repetition in range(12):
            for method in ("legacy", "merged"):
                torch.cuda.synchronize()
                start_seconds = time.perf_counter()
                with torch.inference_mode():
                    for grid_thw in grid_cases[-3:]:
                        if method == "legacy":
                            transformers_5_6_pos_embed_interpolate(encoder, grid_thw)
                        else:
                            encoder.interpolate_position_embeddings(grid_thw)
                torch.cuda.synchronize()
                if repetition > 1:
                    timings[method].append((time.perf_counter() - start_seconds) * 1000)
                else:
                    pass
        print(
            json.dumps(
                {
                    "dtype": str(dtype),
                    "mixed_grid_median_ms": {
                        method: statistics.median(milliseconds)
                        for method, milliseconds in timings.items()
                    },
                }
            ),
            flush=True,
        )

    config = Qwen3OmniMoeVisionEncoderConfig(
        hidden_size=64,
        intermediate_size=128,
        out_hidden_size=64,
        num_heads=4,
        depth=2,
        deepstack_visual_indexes=[0, 1],
        num_position_embeddings=2304,
    )
    encoder = (
        Qwen3OmniMoeVisionEncoderCompat(config)
        .eval()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    legacy_encoder = (
        LegacyVisionEncoder(config).eval().to(device="cuda", dtype=torch.bfloat16)
    )
    legacy_encoder.load_state_dict(encoder.state_dict())
    grid_thw = torch.tensor([[1, 16, 24], [2, 8, 12]], dtype=torch.long)
    patch_count = int(grid_thw.prod(1).sum())
    pixels = torch.randn(
        patch_count,
        config.in_channels * config.temporal_patch_size * config.patch_size**2,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.inference_mode():
        actual_output = encoder(pixels, grid_thw)
        expected_output = legacy_encoder(pixels, grid_thw)
    encoder_exact = torch.equal(
        actual_output.pooler_output, expected_output.pooler_output
    ) and all(
        torch.equal(actual, expected)
        for actual, expected in zip(
            actual_output.deepstack_features, expected_output.deepstack_features
        )
    )
    results = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "interpolation_comparisons": comparisons,
        "interpolation_failures": failures,
        "full_encoder_and_deepstack_bit_exact": encoder_exact,
        "hf_5_19_changed_bf16_elements": hf_changed_elements,
    }
    arguments.output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
