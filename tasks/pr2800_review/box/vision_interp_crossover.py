"""Where the device interpolation's all-rows form overtakes its per-row loop on this GPU, by grid rows per call.

Run on the box from the device interpolation branch's tree (cwd and PYTHONPATH the tree):
    python3 vision_interp_crossover.py --grids <dir with grids_*.jsonl> [<dir> ...]
Builds the served vision encoder (Qwen3-Omni Instruct, bf16, card 0) and takes every distinct image grid row the
recorded MMMU calls hold. For 1 to 16 rows per call it times fast_pos_embed_interpolate_from_list and
fast_pos_embed_interpolate_vectorized on two call shapes: "same", N copies of the median-size row, and "mixed", the
first N distinct rows in recorded order (cycled). Each time is the median of 30 calls after 5 warmup calls, every call
synchronized before and after. Prints the branch's threshold (VECTORIZED_POS_EMBED_MIN_IMAGES) next to the measured
crossover, the smallest N from which the all-rows form is faster at every larger N.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from sglang_omni.models.qwen3_omni.components import vision_compat
from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.qwen3_omni.components.image_encoder import build_visual

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
WARMUP_CALLS = 5
TIMED_CALLS = 30
MAX_ROWS = 16


def median_ms(call, grid: list[list[int]]) -> float:
    times = []
    for index in range(WARMUP_CALLS + TIMED_CALLS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call(grid)
        torch.cuda.synchronize()
        if index >= WARMUP_CALLS:
            times.append((time.perf_counter() - start) * 1000)
        else:
            pass
    return statistics.median(times)


def crossover(rows: list[tuple[int, float, float]]) -> int | None:
    for index, (count, _, _) in enumerate(rows):
        if all(vectorized < loop for _, loop, vectorized in rows[index:]):
            return count
        else:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    distinct: list[list[int]] = []
    for directory in args.grids:
        for path in sorted(directory.glob("grids_*.jsonl")):
            for line in path.read_text().splitlines():
                for row in json.loads(line)["image"] or []:
                    if row not in distinct:
                        distinct.append(row)
                    else:
                        pass
    path = snapshot_download(MODEL)
    visual = build_visual(path, thinker_cfg=load_thinker_config(path), torch_dtype=torch.bfloat16, device="cuda:0")
    by_patches = sorted(distinct, key=lambda row: row[0] * row[1] * row[2])
    median_row = by_patches[len(by_patches) // 2]
    print(f"{len(distinct)} distinct image rows; median row {median_row}; branch threshold "
          f"{vision_compat.VECTORIZED_POS_EMBED_MIN_IMAGES}; {torch.cuda.get_device_name(0)}", flush=True)
    with torch.no_grad():
        for shape in ("same", "mixed"):
            rows = []
            for count in range(1, MAX_ROWS + 1):
                grid = (
                    [median_row] * count
                    if shape == "same"
                    else [distinct[index % len(distinct)] for index in range(count)]
                )
                loop = median_ms(visual.fast_pos_embed_interpolate_from_list, grid)
                vectorized = median_ms(visual.fast_pos_embed_interpolate_vectorized, grid)
                rows.append((count, loop, vectorized))
                print(f"{shape:5s} rows {count:2d}: loop {loop:.3f} ms, vectorized {vectorized:.3f} ms, "
                      f"vectorized/loop {vectorized / loop:.2f}", flush=True)
            print(f"{shape:5s} crossover (vectorized faster from here on): {crossover(rows)}", flush=True)


if __name__ == "__main__":
    main()
