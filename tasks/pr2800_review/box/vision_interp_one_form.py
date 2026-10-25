"""The vision position interpolation forms of one tree: output hashes and time, on grid sweeps and served calls.

Run on the box from an arm's tree (cwd and PYTHONPATH the tree):
    python3 vision_interp_one_form.py run --forms <name>=<method> [...] --calls <name>=<dir> [...] --out <json>
    python3 vision_interp_one_form.py compare <json> <json>
The served vision encoder is built (Qwen3-Omni Instruct, bf16, card 0); each form is a method of it, called with a grid
list. "dispatch" names the tree's own choice: the vectorized method at VECTORIZED_POS_EMBED_MIN_IMAGES grids or more;
"legacy" is main's legacy_pos_embed_interpolate, given the grid list as the host tensor its forward passes.
Inputs: "same" N copies of the median recorded image grid and "mixed" the first N distinct recorded image grids, for N
in SWEEP_COUNTS; then every interpolation of each served call set (grids_*.jsonl from encoder_grids/sitecustomize.py).
Per input and form: the sha256 of the bf16 output and of the fp32 output (the encoder cast to fp32), the bf16 output's
relative L2 to the table's bilinear resampling in float64 (median and largest over interpolations), and two times,
each the median of TIMED_CALLS after WARMUP_CALLS: alone (synchronized before and after the call) and in forward (the
patch embedding of that many random patches, the interpolation, the add, synchronized after, as the encoder's forward
runs them). compare prints, per input, whether every form's hashes agree across both files, then the time tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from sglang_omni.models.qwen3_omni.components import vision_compat
from sglang_omni.models.qwen3_omni.components.common import load_thinker_config
from sglang_omni.models.qwen3_omni.components.image_encoder import build_visual

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
WARMUP_CALLS = 5
TIMED_CALLS = 30
SWEEP_COUNTS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32)


def load_interpolations(directory: Path) -> list[list[list[int]]]:
    interpolations = []
    for path in sorted(directory.glob("grids_*.jsonl")):
        for line in path.read_text().splitlines():
            call = json.loads(line)
            interpolations.extend(call[key] for key in ("image", "video") if call[key])
    return interpolations


def sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()[:16]


def float64_truth(
    table: torch.Tensor, grid: list[list[int]], merge_size: int
) -> torch.Tensor:
    hidden_size = table.shape[1]
    pieces = []
    for frames, height, width in grid:
        resampled = F.interpolate(
            table, size=(height, width), mode="bilinear", align_corners=True
        )[0]
        merged = (
            resampled.permute(1, 2, 0)
            .reshape(
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
                hidden_size,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, hidden_size)
        )
        pieces.append(merged.repeat(frames, 1))
    return torch.cat(pieces)


def median_ms(run, warmup: int = WARMUP_CALLS, timed: int = TIMED_CALLS) -> float:
    times = []
    for index in range(warmup + timed):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        if index >= warmup:
            times.append((time.perf_counter() - start) * 1000)
        else:
            pass
    return statistics.median(times)


def run(args: argparse.Namespace) -> None:
    path = snapshot_download(MODEL)
    visual = build_visual(
        path,
        thinker_cfg=load_thinker_config(path),
        torch_dtype=torch.bfloat16,
        device="cuda:0",
    )
    visual = visual.eval()
    config = visual.config
    patch_features = (
        config.in_channels * config.temporal_patch_size * config.patch_size**2
    )
    forms = {}
    for entry in args.forms:
        name, method = entry.split("=", 1)
        if method == "dispatch":
            threshold = vision_compat.VECTORIZED_POS_EMBED_MIN_IMAGES
            forms[name] = lambda grid, threshold=threshold: (
                visual.fast_pos_embed_interpolate_vectorized(grid)
                if len(grid) >= threshold
                else visual.fast_pos_embed_interpolate_from_list(grid)
            )
        elif method == "legacy":
            forms[name] = lambda grid: visual.legacy_pos_embed_interpolate(
                torch.tensor(grid)
            )
        else:
            forms[name] = getattr(visual, method)
    calls = {}
    distinct: list[list[int]] = []
    for entry in args.calls:
        name, directory = entry.split("=", 1)
        calls[name] = load_interpolations(Path(directory))
        for grid in calls[name]:
            for row in grid:
                if row[0] == 1 and row not in distinct:
                    distinct.append(row)
                else:
                    pass
    median_row = sorted(distinct, key=lambda row: row[1] * row[2])[len(distinct) // 2]
    inputs: dict[str, list[list[list[int]]]] = {}
    for count in SWEEP_COUNTS:
        inputs[f"same {count}"] = [[median_row] * count]
        inputs[f"mixed {count}"] = [
            [distinct[index % len(distinct)] for index in range(count)]
        ]
    inputs.update(calls)
    print(
        f"{torch.cuda.get_device_name(0)}; {len(distinct)} distinct image grids, median {median_row}; forms "
        f"{list(forms)}",
        flush=True,
    )
    results: dict[str, dict[str, dict[str, object]]] = {}
    side = visual.num_grid_per_side
    table = visual.pos_embed.weight.detach().double().T.reshape(1, -1, side, side)
    with torch.no_grad():
        for name, interpolations in inputs.items():
            results[name] = {}
            for form, call in forms.items():
                patches = [
                    torch.randn(
                        sum(t * h * w for t, h, w in grid),
                        patch_features,
                        device="cuda:0",
                        dtype=torch.bfloat16,
                    )
                    for grid in interpolations
                ]

                def alone() -> None:
                    for grid in interpolations:
                        call(grid)

                def in_forward() -> None:
                    for grid, patch in zip(interpolations, patches):
                        hidden_states = visual.patch_embed(patch)
                        hidden_states = hidden_states + call(grid).to(
                            hidden_states.dtype
                        )

                single = len(interpolations) == 1
                distances = []
                for grid in interpolations:
                    truth = float64_truth(table, grid, visual.spatial_merge_size)
                    distances.append(
                        float((call(grid).double() - truth).norm() / truth.norm())
                    )
                    del truth
                results[name][form] = {
                    "bf16": [sha256(call(grid)) for grid in interpolations],
                    "distance": distances,
                    "alone_ms": median_ms(alone, *(() if single else (1, 3))),
                    "in_forward_ms": median_ms(in_forward, *(() if single else (1, 3))),
                }
                print(
                    f"{name:14s} {form:10s} alone {results[name][form]['alone_ms']:8.3f} ms, in forward "
                    f"{results[name][form]['in_forward_ms']:8.3f} ms, relative L2 to float64 median "
                    f"{statistics.median(distances):.3e} largest {max(distances):.3e}",
                    flush=True,
                )
                del patches
        visual = visual.float()
        for name, interpolations in inputs.items():
            for form, call in forms.items():
                results[name][form]["fp32"] = [
                    sha256(call(grid)) for grid in interpolations
                ]
    Path(args.out).write_text(json.dumps(results, indent=1))


def compare(args: argparse.Namespace) -> None:
    first, second = (
        json.loads(Path(path).read_text()) for path in (args.first, args.second)
    )
    print("== hashes: every form of both files, per input")
    for name in first:
        hashes = {
            dtype: {
                tuple(entry[dtype])
                for results in (first, second)
                for entry in results[name].values()
            }
            for dtype in ("bf16", "fp32")
        }
        print(
            f"{name:14s} bf16 {'equal' if len(hashes['bf16']) == 1 else 'DIFFER'}, fp32 "
            f"{'equal' if len(hashes['fp32']) == 1 else 'DIFFER'}"
        )
    for metric in ("alone_ms", "in_forward_ms"):
        print(f"== {metric}")
        for name in first:
            cells = [
                f"{form} {entry[metric]:.3f}"
                for results in (first, second)
                for form, entry in results[name].items()
            ]
            print(f"{name:14s} " + ", ".join(cells))
    print("== relative L2 to float64, median / largest")
    for name in first:
        cells = [
            f"{form} {statistics.median(entry['distance']):.3e} / {max(entry['distance']):.3e}"
            for results in (first, second)
            for form, entry in results[name].items()
            if "distance" in entry
        ]
        print(f"{name:14s} " + ", ".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--forms", nargs="+", required=True, help="name=method or name=dispatch"
    )
    run_parser.add_argument("--calls", nargs="+", required=True, help="name=directory")
    run_parser.add_argument("--out", required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("first")
    compare_parser.add_argument("second")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
