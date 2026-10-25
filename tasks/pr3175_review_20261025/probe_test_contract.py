"""Run the existing oracle with old arithmetic and with tighter CUDA precision."""

import json
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from base_vision_compat import Qwen3OmniMoeVisionEncoderCompat as BaseEncoder

import tests.unit_test.qwen3_omni.test_vision_compat as contracts


def main() -> None:
    original_factory = contracts.make_encoder
    records = []

    def legacy(self, grid):
        return BaseEncoder.legacy_pos_embed_interpolate(self, torch.tensor(grid))

    def old_factory(dtype):
        encoder = original_factory(dtype)
        encoder.config = SimpleNamespace(spatial_merge_size=2)
        encoder.interpolate_position_embeddings = types.MethodType(legacy, encoder)
        return encoder

    contracts.make_encoder = old_factory
    for dtype in contracts.BILINEAR_TOLERANCES:
        for name, grid in contracts.GRID_CASES.items():
            contracts.test_position_interpolation_matches_float64_bilinear(grid, dtype)
            records.append(
                {
                    "arm": "old_current_bound",
                    "dtype": str(dtype),
                    "case": name,
                    "passed": True,
                }
            )
    contracts.BILINEAR_TOLERANCES[torch.bfloat16] = 2e-3
    for name, grid in contracts.GRID_CASES.items():
        try:
            contracts.test_position_interpolation_matches_float64_bilinear(
                grid, torch.bfloat16
            )
        except AssertionError:
            records.append({"arm": "old_tighter_bound", "case": name, "rejected": True})
        else:
            raise AssertionError(
                "the stronger BF16 oracle must reject the old arithmetic"
            )

    devices = []

    def cuda_factory(dtype):
        encoder = original_factory(dtype).cuda()
        devices.append(encoder.pos_embed.weight.device.type)
        return encoder

    contracts.make_encoder = cuda_factory
    for dtype in contracts.BILINEAR_TOLERANCES:
        for name, grid in contracts.GRID_CASES.items():
            contracts.test_position_interpolation_matches_float64_bilinear(grid, dtype)
            records.append(
                {
                    "arm": "cuda_tighter_bound",
                    "dtype": str(dtype),
                    "case": name,
                    "passed": True,
                }
            )
    assert devices == ["cuda"] * 12
    Path("/data/review3175oct25/stronger_contract.json").write_text(
        json.dumps(records, indent=2)
    )
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
