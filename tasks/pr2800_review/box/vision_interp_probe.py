"""How long the Qwen3-Omni vision encoder's position interpolation takes on MMMU-50's 51 images, against the device
form SGLang runs, and how far apart their outputs are.

Run on the box from main's tree (cwd and PYTHONPATH the tree):
    python3 vision_interp_probe.py --inputs /data/pr2800/run309/probe/inputs.pt
Builds the served vision encoder (build_visual, the checkpoint's dtype, card 0). Per image grid it times the encoder's
legacy_pos_embed_interpolate (index and weight lists built on the host) and SGLang 0.5.22's
Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list on the same pos_embed table (on the device), each
synchronized per call, one warmup sweep then three timed sweeps; the whole encoder forward over the same images is
timed the same way for scale. Prints the sweep times and the largest difference and relative L2 between the two
interpolations' outputs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from huggingface_hub import snapshot_download
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel

from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.qwen3_omni.components.image_encoder import build_visual

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
TIMED_SWEEPS = 3


def sweep_ms(call, arguments: list) -> float:
    total = 0.0
    for argument in arguments:
        torch.cuda.synchronize()
        start = time.perf_counter()
        call(argument)
        torch.cuda.synchronize()
        total += time.perf_counter() - start
    return total * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    path = snapshot_download(MODEL)
    visual = build_visual(path, thinker_cfg=load_thinker_config(path), torch_dtype=None, device="cuda:0").eval()
    device = visual.pos_embed.weight.device
    device_form = SimpleNamespace(
        num_grid_per_side=visual.num_grid_per_side,
        spatial_merge_size=visual.spatial_merge_size,
        pos_embed=visual.pos_embed,
        device=device,
        dtype=visual.pos_embed.weight.dtype,
    )
    inputs = torch.load(args.inputs)
    grids = [grid_thw.cpu() for _, grid_thw in inputs]
    images = [(pixel_values.to(device), grid_thw.to(device)) for pixel_values, grid_thw in inputs]
    print(f"{len(grids)} images, patches per image {[int(grid.prod()) for grid in grids]}")

    def host(grid: torch.Tensor) -> torch.Tensor:
        return visual.legacy_pos_embed_interpolate(grid)

    def on_device(grid: torch.Tensor) -> torch.Tensor:
        return Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list(device_form, grid.tolist())

    def encoder(image: tuple[torch.Tensor, torch.Tensor]) -> None:
        visual(image[0], grid_thw=image[1])

    with torch.no_grad():
        largest = 0.0
        relative = []
        for grid in grids:
            first, second = host(grid).float(), on_device(grid).float()
            largest = max(largest, float((first - second).abs().max()))
            relative.append(float((first - second).norm() / first.norm()))
        print(f"host against device: largest difference {largest:.4g}, relative L2 largest {max(relative):.4g}, "
              f"outputs equal on {sum(value == 0 for value in relative)} of {len(relative)}")
        for name, call, arguments in (("host interpolation", host, grids), ("device interpolation", on_device, grids),
                                      ("encoder forward", encoder, images)):
            sweep_ms(call, arguments)
            times = [sweep_ms(call, arguments) for _ in range(TIMED_SWEEPS)]
            print(f"{name}: {', '.join(f'{value:.1f}' for value in times)} ms over {len(arguments)} images")


if __name__ == "__main__":
    main()
