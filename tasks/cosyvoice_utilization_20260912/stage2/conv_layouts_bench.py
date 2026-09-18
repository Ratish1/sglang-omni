"""Three layouts for the DiT's causal conv position embed, the conv alone.

  padded  rows scattered to (rows, widest), what main runs
  long    one sequence, 30 zero frames before each row (the first packed attempt)
  tiled   that same sequence cut into as many equal tiles as there are rows,
          each tile carrying the 30 frames before it, so the conv stays a
          batched call and its work is total frames, not rows x widest

Each is timed in bfloat16 autocast under inference mode, and its output is
compared with the float32 padded conv on the same activations, so a layout that
makes cuDNN pick a lossier algorithm shows up as a lower SNR than padded has.

  python conv_layouts_bench.py --model .../snapshots/master
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedRows,
    gather_rows,
    pack_rows,
    scatter_rows,
)
from sglang_omni.models.fun_cosyvoice3.stages import load_cosyvoice3_flow_hift

ITERATIONS = 50

# mel frames per request row; every shape is doubled for CFG as the solver does
SHAPES: dict[str, tuple[int, ...]] = {
    "1 row of 550": (550,),
    "2 rows, equal 550": (550, 550),
    "4 rows, 400 to 950": (400, 550, 750, 950),
    "8 rows, 400 to 1100": tuple(400 + 100 * index for index in range(8)),
    "16 rows, 400 to 1150": tuple(400 + 50 * index for index in range(16)),
    "16 rows, equal 550": (550,) * 16,
    "16 rows, 500 to 650": tuple(500 + 10 * index for index in range(16)),
    "8 rows, 500 to 640": tuple(500 + 20 * index for index in range(8)),
    "4 rows, 500 to 560": (500, 520, 540, 560),
    "16 rows, one runaway of 4300": (4300,) + tuple(400 + 50 * i for i in range(15)),
    "8 rows, one runaway of 4300": (4300,) + tuple(400 + 100 * i for i in range(7)),
}


def padded(embed, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    return gather_rows(embed(scatter_rows(h, rows, rows.width)), rows)


def spaced_layout(h: torch.Tensor, rows: PackedRows, gap: int):
    total = h.shape[1]
    outputs = torch.arange(total, device=h.device) + gap * rows.row_ids
    return outputs, outputs + gap, total + gap * len(rows.lengths)


def long(embed, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    gap = embed.kernel_size - 1
    outputs, inputs, length = spaced_layout(h, rows, gap)
    spaced = h.new_zeros(h.shape[2], length)
    spaced.T[inputs] = h[0]
    first = embed.conv1(spaced.unsqueeze(0))[0]
    spaced = torch.zeros_like(spaced)
    spaced.T[inputs] = first.T[outputs]
    return embed.conv2(spaced.unsqueeze(0))[0].T[outputs].unsqueeze(0)


def tiled_conv(conv, spaced: torch.Tensor, tiles: int, gap: int) -> torch.Tensor:
    """spaced: (channels, length), length a multiple of tiles. The causal conv
    of the whole sequence, computed as one batched call over equal tiles."""
    channels, length = spaced.shape
    size = length // tiles
    windows = F.pad(spaced, (gap, 0)).unfold(1, size + gap, size)
    out = conv(windows.transpose(0, 1).contiguous())
    return out.transpose(0, 1).reshape(channels, length)


def tiled(embed, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    gap = embed.kernel_size - 1
    tiles = len(rows.lengths)
    total = h.shape[1]
    positions = torch.arange(total, device=h.device) + gap * (rows.row_ids + 1)
    length = -(-(total + gap * tiles) // tiles) * tiles
    spaced = h.new_zeros(h.shape[2], length)
    spaced.T[positions] = h[0]
    first = tiled_conv(embed.conv1, spaced, tiles, gap)
    spaced = torch.zeros_like(spaced)
    spaced.T[positions] = first.T[positions]
    return tiled_conv(embed.conv2, spaced, tiles, gap).T[positions].unsqueeze(0)


# The same two layouts with everything that does not change between the ten
# Euler steps computed once per Flow call and kept on the rows, which is how a
# per call object would hold it. Cached on the PackedRows instance by id.
STEP_INVARIANT: dict[tuple[int, str], torch.Tensor] = {}


def padded_once(embed, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    key = (id(rows), "padded")
    if key not in STEP_INVARIANT:
        STEP_INVARIANT[key] = rows.row_ids * rows.width + rows.positions
    index = STEP_INVARIANT[key]
    count, width, channels = len(rows.lengths), rows.width, h.shape[2]
    flat = h.new_zeros(count * width, channels)
    flat[index] = h[0]
    out = embed(flat.view(count, width, channels))
    return out.reshape(count * width, channels)[index].unsqueeze(0)


def tiled_once(embed, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    gap = embed.kernel_size - 1
    tiles = len(rows.lengths)
    key = (id(rows), "tiled")
    if key not in STEP_INVARIANT:
        STEP_INVARIANT[key] = torch.arange(h.shape[1], device=h.device) + gap * (
            rows.row_ids + 1
        )
    positions = STEP_INVARIANT[key]
    length = -(-(h.shape[1] + gap * tiles) // tiles) * tiles
    spaced = h.new_zeros(h.shape[2], length)
    spaced.T[positions] = h[0]
    first = tiled_conv(embed.conv1, spaced, tiles, gap)
    spaced = torch.zeros_like(spaced)
    spaced.T[positions] = first.T[positions]
    return tiled_conv(embed.conv2, spaced, tiles, gap).T[positions].unsqueeze(0)


def fixed_tiles(width: int):
    """Tiles of one fixed width, the way SGLang's audio encoders feed their conv
    stems fixed windows: cuDNN then sees one width and only the tile count
    varies."""

    def layout(embed, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
        gap = embed.kernel_size - 1
        key = (id(rows), "fixed")
        if key not in STEP_INVARIANT:
            STEP_INVARIANT[key] = torch.arange(h.shape[1], device=h.device) + gap * (
                rows.row_ids + 1
            )
        positions = STEP_INVARIANT[key]
        tiles = -(-(h.shape[1] + gap * len(rows.lengths)) // width)
        spaced = h.new_zeros(h.shape[2], tiles * width)
        spaced.T[positions] = h[0]
        first = tiled_conv(embed.conv1, spaced, tiles, gap)
        spaced = torch.zeros_like(spaced)
        spaced.T[positions] = first.T[positions]
        return tiled_conv(embed.conv2, spaced, tiles, gap).T[positions].unsqueeze(0)

    return layout


def snr_db(reference: torch.Tensor, other: torch.Tensor) -> float:
    noise = (reference.double() - other.double()).pow(2).sum()
    if noise == 0:
        return float("inf")
    return round(float(10 * torch.log10(reference.double().pow(2).sum() / noise)), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    flow, _ = load_cosyvoice3_flow_hift(args.model, device="cuda:0")
    embed = flow.decoder.estimator.input_embed.conv_pos_embed
    parser_layouts = {
        "padded_once": padded_once,
        "tiled_once": tiled_once,
        "fixed_250": fixed_tiles(250),
        "fixed_500": fixed_tiles(500),
        "fixed_1000": fixed_tiles(1000),
        "fixed_2000": fixed_tiles(2000),
    }
    layouts = dict(parser_layouts)
    # The same layouts again with cuDNN choosing its algorithm by measurement
    # instead of by heuristic; the five warmup calls absorb the search.
    for key, layout in parser_layouts.items():

        def searched(embed, h, rows, layout=layout):
            with torch.backends.cudnn.flags(benchmark=True):
                return layout(embed, h, rows)

        layouts[f"{key}+search"] = searched

    report = []
    for name, lengths in SHAPES.items():
        STEP_INVARIANT.clear()
        rows = pack_rows(lengths * 2, torch.device("cuda"))
        torch.manual_seed(0)
        h = torch.randn(1, rows.total, 1024, device="cuda")
        with torch.inference_mode():
            truth = padded(embed, h, rows)
        row = {
            "shape": name,
            "rows_x_widest": len(rows.lengths) * rows.width,
            "total_frames": rows.total,
        }
        half = h.to(torch.bfloat16)
        for key, layout in layouts.items():

            def run(layout=layout):
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    return layout(embed, half, rows)

            for _ in range(5):
                out = run()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(ITERATIONS):
                run()
            end.record()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            run()
            torch.cuda.synchronize()
            row[key] = {
                "ms": round(start.elapsed_time(end) / ITERATIONS, 3),
                "peak_mib": round(
                    (torch.cuda.max_memory_allocated() - before) / (1 << 20), 1
                ),
                "snr_db_vs_float32": snr_db(truth, out),
                "dtype": str(out.dtype),
            }
        with torch.inference_mode():
            row["float32_long_snr_db"] = snr_db(truth, long(embed, h, rows))
            row["float32_tiled_snr_db"] = snr_db(truth, tiled(embed, h, rows))
        report.append(row)

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "rows": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
