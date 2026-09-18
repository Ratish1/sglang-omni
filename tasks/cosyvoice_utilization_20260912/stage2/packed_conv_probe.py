"""The conv position embed on the packed sequence against the padded one.

Run from a tree whose PackedDiT._conv_pos_embed is the packed conv. The padded
conv it replaced is kept here as the reference. One process, the real Flow,
the serving entry points (hop_batch and leftover_batch with their autocast and
inference mode), no profiler. For each batch shape, with each conv:

  wall time of the whole Flow call, interleaved rounds
  peak torch memory of the call
  the mel against the reference: bit identity, max abs difference, SNR in dB

and the conv alone on one activation tensor of that shape: time and peak memory.

  python packed_conv_probe.py --model .../snapshots/master
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    PackedDiT,
    PackedRows,
    gather_rows,
    pack_rows,
    scatter_rows,
)
from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    FlowBatchInput,
    load_cosyvoice3_flow_hift,
    patch_chunk_mask,
)

WARMUP = 2
ITERATIONS = 8
ROUNDS = 3

# name, prompt tokens, generated tokens of each row
SHAPES: dict[str, tuple[int, tuple[int, ...]]] = {
    "1 row": (150, (128,)),
    "2 rows, equal": (150, (128, 128)),
    "4 rows, ragged": (150, (53, 128, 228, 328)),
    "8 rows, ragged": (150, tuple(53 + 50 * index for index in range(8))),
    "16 rows, ragged": (150, tuple(53 + 25 * index for index in range(16))),
    "16 rows, one runaway of 2000": (
        150,
        (2000,) + tuple(53 + 25 * i for i in range(15)),
    ),
}


def padded_conv(self: PackedDiT, h: torch.Tensor, rows: PackedRows) -> torch.Tensor:
    padded = scatter_rows(h, rows, rows.width)
    return gather_rows(self.dit.input_embed.conv_pos_embed(padded), rows)


def items(prompt: int, tokens: tuple[int, ...], flow) -> list[FlowBatchInput]:
    generator = torch.Generator().manual_seed(len(tokens) * 1000 + sum(tokens))
    return [
        FlowBatchInput(
            token=torch.randint(
                0, 6000, (1, count), generator=generator, dtype=torch.int32
            ),
            prompt_token=torch.randint(
                0, 6000, (1, prompt), generator=generator, dtype=torch.int32
            ),
            prompt_feat=torch.randn(
                1, prompt * 2, flow.output_size, generator=generator
            ),
            embedding=torch.randn(
                1, flow.spk_embed_affine_layer.in_features, generator=generator
            ),
        )
        for count in tokens
    ]


def peak_mib(call) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    call()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / (1 << 20)


def snr_db(reference: torch.Tensor, other: torch.Tensor) -> float:
    noise = (reference.double() - other.double()).pow(2).sum()
    if noise == 0:
        return float("inf")
    return float(10 * torch.log10(reference.double().pow(2).sum() / noise))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    flow, hift = load_cosyvoice3_flow_hift(args.model, device="cuda:0")
    patch_chunk_mask()
    vocoder = CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    estimator = flow.packed_estimator
    convs = {"padded": padded_conv, "packed": PackedDiT._conv_pos_embed}

    report = []
    for name, (prompt, tokens) in SHAPES.items():
        batch = items(prompt, tokens, flow)
        row = {"shape": name}
        for kind, call in (
            ("hop", lambda: vocoder.hop_batch(batch)),
            ("final", lambda: vocoder.leftover_batch(batch)),
        ):
            mels, walls, peaks = {}, {key: [] for key in convs}, {}
            for key, conv in convs.items():
                PackedDiT._conv_pos_embed = conv
                for _ in range(WARMUP):
                    call()
                mels[key] = [mel.clone() for mel in call()]
                peaks[key] = peak_mib(call)
            for _ in range(ROUNDS):
                for key, conv in convs.items():
                    PackedDiT._conv_pos_embed = conv
                    for _ in range(ITERATIONS):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        call()
                        torch.cuda.synchronize()
                        walls[key].append((time.perf_counter() - started) * 1000)
            reference = torch.cat([mel.flatten() for mel in mels["padded"]])
            candidate = torch.cat([mel.flatten() for mel in mels["packed"]])
            row[kind] = {
                "padded_ms": round(statistics.median(walls["padded"]), 2),
                "packed_ms": round(statistics.median(walls["packed"]), 2),
                "padded_min_ms": round(min(walls["padded"]), 2),
                "packed_min_ms": round(min(walls["packed"]), 2),
                "padded_peak_mib": round(peaks["padded"], 1),
                "packed_peak_mib": round(peaks["packed"], 1),
                "bit_identical": bool(torch.equal(reference, candidate)),
                "max_abs_diff": float((reference - candidate).abs().max()),
                "snr_db": round(snr_db(reference, candidate), 2),
                "mel_abs_max": float(reference.abs().max()),
            }

        # The conv alone, on one bfloat16 activation of this shape, CFG rows doubled.
        lengths = tuple((prompt + count - 3) * 2 for count in tokens) * 2
        rows = pack_rows(lengths, torch.device("cuda"))
        h = torch.randn(1, rows.total, 1024, device="cuda", dtype=torch.bfloat16)
        alone = {"rows_x_widest": len(lengths) * rows.width, "total_frames": rows.total}
        for key, conv in convs.items():

            def run(conv=conv):
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    return conv(estimator, h, rows)

            for _ in range(5):
                run()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(30):
                run()
            end.record()
            torch.cuda.synchronize()
            alone[f"{key}_ms"] = round(start.elapsed_time(end) / 30, 3)
            alone[f"{key}_peak_mib"] = round(peak_mib(run), 1)
        row["conv_alone"] = alone
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
