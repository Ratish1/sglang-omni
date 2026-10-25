"""Compare actual CUDA interpolation with the old paths and a float64 oracle."""

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from base_vision_compat import Qwen3OmniMoeVisionEncoderCompat as BaseEncoder
from port_vision_compat import Qwen3OmniMoeVisionEncoderCompat as PortEncoder

from sglang_omni.models.qwen3_omni.components.vision_compat import (
    Qwen3OmniMoeVisionEncoderCompat,
)


@torch.inference_mode()
def main() -> None:
    torch.manual_seed(3175)
    cases = {
        "minimum": [[1, 2, 2]],
        "identity": [[1, 48, 48]],
        "rectangular-down": [[1, 16, 24]],
        "large-up": [[1, 64, 98]],
        "skinny": [[1, 2, 158]],
        "mixed-video-image": [[3, 12, 16], [1, 20, 28], [2, 8, 8]],
        "many": [[1, 24, 24]] * 8,
    }
    records = []
    for channels in (16, 1152):
        source = torch.randn(48 * 48, channels)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            encoder = Qwen3OmniMoeVisionEncoderCompat.__new__(
                Qwen3OmniMoeVisionEncoderCompat
            )
            torch.nn.Module.__init__(encoder)
            encoder.config = type("Config", (), {"spatial_merge_size": 2})()
            encoder.spatial_merge_size = 2
            encoder.num_grid_per_side = 48
            encoder.pos_embed = torch.nn.Embedding.from_pretrained(
                source.to(device="cuda", dtype=dtype)
            )
            table = (
                encoder.pos_embed.weight.detach()
                .cpu()
                .double()
                .T.reshape(1, channels, 48, 48)
            )
            for name, grid in cases.items():
                expected = torch.cat(
                    [
                        F.interpolate(
                            table, size=(h, w), mode="bilinear", align_corners=True
                        )[0]
                        .reshape(channels, h // 2, 2, w // 2, 2)
                        .permute(1, 3, 2, 4, 0)
                        .reshape(h * w, channels)
                        .repeat(t, 1)
                        for t, h, w in grid
                    ]
                )
                actual = encoder.interpolate_position_embeddings(grid)
                assert actual.device.type == "cuda" and actual.dtype == dtype
                assert actual.shape == expected.shape
                error = actual.cpu().double() - expected
                maximum = error.abs().max().item()
                bound = (
                    torch.finfo(dtype).eps + 512 * torch.finfo(torch.float32).eps
                ) * expected.abs().max().item()
                assert maximum <= bound, (channels, dtype, name, maximum, bound)
                if name in ("minimum", "identity"):
                    assert torch.equal(actual.cpu(), expected.to(dtype))
                record = {
                    "channels": channels,
                    "dtype": str(dtype),
                    "case": name,
                    "shape": list(actual.shape),
                    "max_abs_error": maximum,
                    "relative_l2": (error.norm() / expected.norm()).item(),
                }
                records.append(record)
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode("error")
            try:
                encoder.interpolate_position_embeddings(cases["mixed-video-image"])
            finally:
                torch.cuda.set_sync_debug_mode("default")
    encoder.pos_embed = torch.nn.Embedding.from_pretrained(
        source.to(device="cuda", dtype=torch.bfloat16)
    )
    timing = []
    for name in ("rectangular-down", "large-up", "mixed-video-image", "many"):
        grid = cases[name]
        host_grid = torch.tensor(grid)
        port_method = (
            PortEncoder.fast_pos_embed_interpolate_vectorized
            if len(grid) >= 6
            else PortEncoder.fast_pos_embed_interpolate_from_list
        )
        methods = (
            lambda: BaseEncoder.legacy_pos_embed_interpolate(encoder, host_grid),
            lambda: port_method(encoder, grid),
            lambda: encoder.interpolate_position_embeddings(grid),
        )
        for method in methods:
            method()
        samples = [[], [], []]
        for repeat in range(7):
            for arm in (0, 1, 2) if repeat % 2 == 0 else (2, 1, 0):
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(5):
                    methods[arm]()
                torch.cuda.synchronize()
                samples[arm].append((time.perf_counter() - started) * 1000 / 5)
        outputs = [method().cpu().double() for method in methods]
        table = (
            encoder.pos_embed.weight.detach().cpu().double().T.reshape(1, 1152, 48, 48)
        )
        expected = torch.cat(
            [
                F.interpolate(table, size=(h, w), mode="bilinear", align_corners=True)[
                    0
                ]
                .reshape(1152, h // 2, 2, w // 2, 2)
                .permute(1, 3, 2, 4, 0)
                .reshape(h * w, 1152)
                .repeat(t, 1)
                for t, h, w in grid
            ]
        )
        timing.append(
            {
                "case": name,
                "order": ["base", "described_port", "current"],
                "median_wall_ms": [statistics.median(s) for s in samples],
                "samples_ms": samples,
                "relative_l2": [
                    ((o - expected).norm() / expected.norm()).item() for o in outputs
                ],
                "current_equals_described_port": torch.equal(outputs[1], outputs[2]),
            }
        )
    result = {
        "cuda_cases": records,
        "no_host_wait": True,
        "timing": timing,
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
    }
    Path("/data/review3175oct25/interpolation.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps({"cases_passed": len(records), "timing": timing}), flush=True)


if __name__ == "__main__":
    main()
