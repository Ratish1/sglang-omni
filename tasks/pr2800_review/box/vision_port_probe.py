"""The branch's vision position interpolation against SGLang 0.5.22's, and its unit case's float64 distances.

Run on the box from the branch's tree (cwd and PYTHONPATH the tree):
    python3 vision_port_probe.py --calls <name>=<dir with grids_*.jsonl> [<name>=<dir> ...]
1. The unit case (test_vision_compat.py's make_encoder table and GRID_CASES) on the CPU, bf16 and fp32: per grid case
   the largest absolute difference of fast_pos_embed_interpolate_from_list to F.interpolate(bilinear,
   align_corners=True) in float64, and whether the vectorized form equals the loop.
2. The served encoder (build_visual, the checkpoint's dtype, card 0) on every recorded interpolation of each --calls
   set (videoamme_pair.py --encoder-grids): the branch's loop, vectorized and dispatched forms (the forward's choice at
   VECTORIZED_POS_EMBED_MIN_IMAGES rows) against SGLang's Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list on
   a stub holding the same table, counted bit-equal; then each form's summed time, one warmup and three timed sweeps,
   each interpolation synchronized before and after.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from sglang.srt.models import qwen3_vl
from vision_interp_probe import MODEL, TIMED_SWEEPS, load_interpolations

from sglang_omni.models.qwen3_omni.components import vision_compat
from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.qwen3_omni.components.image_encoder import build_visual
from tests.unit_test.qwen3_omni.test_vision_compat import GRID_CASES, make_encoder


def unit_case_distances() -> None:
    for dtype in (torch.bfloat16, torch.float32):
        for name, grid_thw in GRID_CASES.items():
            encoder = make_encoder(dtype)
            table = encoder.pos_embed.weight.detach().double().T.reshape(1, -1, 48, 48)
            expected = torch.cat([
                F.interpolate(table, size=(h, w), mode="bilinear", align_corners=True)[0]
                .reshape(-1, h // 2, 2, w // 2, 2).permute(1, 3, 2, 4, 0).reshape(h * w, -1).repeat(t, 1)
                for t, h, w in grid_thw
            ])
            with torch.no_grad():
                looped = encoder.fast_pos_embed_interpolate_from_list(grid_thw)
                vectorized = encoder.fast_pos_embed_interpolate_vectorized(grid_thw)
            difference = (looped.double() - expected).abs()
            print(f"unit {str(dtype):15s} {name:17s} largest {float(difference.max()):.3e} mean "
                  f"{float(difference.mean()):.3e}; vectorized equals loop {torch.equal(vectorized, looped)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", nargs="+", required=True, help="name=directory")
    args = parser.parse_args()
    unit_case_distances()
    path = snapshot_download(MODEL)
    visual = build_visual(path, thinker_cfg=load_thinker_config(path), torch_dtype=None, device="cuda:0").eval()
    stub = SimpleNamespace(
        num_grid_per_side=visual.num_grid_per_side,
        spatial_merge_size=visual.spatial_merge_size,
        pos_embed=visual.pos_embed,
        device=visual.pos_embed.weight.device,
        dtype=visual.pos_embed.weight.dtype,
    )
    threshold = vision_compat.VECTORIZED_POS_EMBED_MIN_IMAGES
    sglang_threshold = qwen3_vl._VECTORIZED_VL_POS_EMBED_MIN_IMAGES  # noqa: leading-underscore  # SGLang's constant
    print(f"served table {tuple(visual.pos_embed.weight.shape)} {visual.pos_embed.weight.dtype}; encoder device "
          f"{visual.device} dtype {visual.dtype}; threshold {threshold} (SGLang {sglang_threshold})", flush=True)
    forms = {
        "sglang_loop": lambda grid: qwen3_vl.Qwen3VLMoeVisionModel.fast_pos_embed_interpolate_from_list(stub, grid),
        "loop": visual.fast_pos_embed_interpolate_from_list,
        "vectorized": visual.fast_pos_embed_interpolate_vectorized,
        "dispatch": lambda grid: (
            visual.fast_pos_embed_interpolate_vectorized(grid)
            if len(grid) >= threshold
            else visual.fast_pos_embed_interpolate_from_list(grid)
        ),
    }
    with torch.no_grad():
        for entry in args.calls:
            name, directory = entry.split("=", 1)
            calls, interpolations = load_interpolations(Path(directory))
            equal = dict.fromkeys(forms, 0)
            for grid in interpolations:
                reference = forms["sglang_loop"](grid)
                for form, call in forms.items():
                    equal[form] += int(torch.equal(call(grid), reference))
            print(f"== {name}: {calls} calls, {len(interpolations)} interpolations, at or over {threshold}: "
                  f"{sum(len(grid) >= threshold for grid in interpolations)}", flush=True)
            for form, call in forms.items():
                times = []
                for sweep in range(TIMED_SWEEPS + 1):
                    total = 0.0
                    for grid in interpolations:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        call(grid)
                        torch.cuda.synchronize()
                        total += time.perf_counter() - start
                    if sweep > 0:
                        times.append(total * 1000)
                    else:
                        pass
                print(f"{form:11s} equal to SGLang's loop {equal[form]} of {len(interpolations)}; time "
                      f"{', '.join(f'{value:.1f}' for value in times)} ms", flush=True)


if __name__ == "__main__":
    main()
