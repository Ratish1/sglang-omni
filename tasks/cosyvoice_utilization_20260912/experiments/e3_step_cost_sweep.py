#!/usr/bin/env python3
"""E3: wall time and peak memory of one vocoder step over batch and padded length.

Run on the H100 venv at the branch head, alone on the GPU, no server:

  python tasks/cosyvoice_utilization_20260912/experiments/e3_step_cost_sweep.py \
      --checkpoint /path/to/Fun-CosyVoice3-0.5B-2512 --device cuda:0 --json e3.json

One real SeedTTS clip supplies the prompt; the generated rows are synthetic
speech tokens of the swept length, since time and memory depend on shapes,
not on token values. Every row of a call has the same length, so the padded
length is the row length. The sweep runs the packed causal hop call, the
packed non-streaming leftover call, and HiFT over a growing mel history, and
reports per point the median wall time, the transient peak above the resident
allocation, and the allocator's reserved footprint after the call. The
reserved footprint is what a warmup at that shape leaves resident before the
KV pool is sized.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import replace

import torch
from e2_flow_path_exactness import PROMPT_TOKENS, Stream, build_streams, flow_input

from sglang_omni.models.fun_cosyvoice3.sglang_model import VOCAB_SIZE
from sglang_omni.models.fun_cosyvoice3.stages import (
    AUTOCAST_DTYPES,
    CosyVoice3Vocoder,
    _patch_chunk_mask,
    load_cosyvoice3_flow_hift,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_MEL_RATIO,
)

# note(ratish): the hop ladder's windows (28, 78, 178, then 103 per hop at the
# 100 token cap), then the AR ceiling of 2048 tokens plus its lookahead.
TOKENS = (28, 78, 178, 384, 796, 1620, 2051)
BATCHES = (1, 2, 4, 8, 16)
HISTORY_FRAMES = (50, 150, 350, 750, 1550, 3150, 4200)
REPEATS = 3
GIB = 1024**3


def synthetic_rows(stream: Stream, tokens: int, batch: int, seed: int) -> list[Stream]:
    generator = torch.Generator().manual_seed(seed)
    return [
        replace(
            stream,
            sample_id=f"{stream.sample_id}#{index}",
            tokens=torch.randint(
                0, VOCAB_SIZE, (1, tokens), dtype=torch.int32, generator=generator
            ),
        )
        for index in range(batch)
    ]


def timed(call: Callable[[], object], device: torch.device) -> tuple[float, int, int]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    call()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1e3
    peak = torch.cuda.max_memory_allocated(device) - resident
    return wall_ms, peak, torch.cuda.memory_reserved(device)


def measure(call: Callable[[], object], device: torch.device) -> dict[str, float]:
    call()
    samples = [timed(call, device) for _ in range(REPEATS)]
    return {
        "wall_ms": statistics.median(s[0] for s in samples),
        "peak_gib": max(s[1] for s in samples) / GIB,
        "reserved_gib": max(s[2] for s in samples) / GIB,
    }


def sweep_flow(
    vocoder: CosyVoice3Vocoder, stream: Stream, path: str, device: torch.device
) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for tokens in TOKENS:
        lookahead = PRE_LOOKAHEAD_LEN if path == "hop" else 0
        frames = (PROMPT_TOKENS + tokens - lookahead) * TOKEN_MEL_RATIO
        for batch in BATCHES:
            items = [
                flow_input(row, tokens)
                for row in synthetic_rows(stream, tokens, batch, seed=tokens)
            ]
            call = vocoder.hop_batch if path == "hop" else vocoder.leftover_batch
            point: dict[str, float | int | str | bool] = {
                "path": path,
                "batch": batch,
                "tokens": tokens,
                "frames": frames,
            }
            try:
                point.update(measure(lambda: call(items), device))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                point["oom"] = True
                rows.append(point)
                print(f"{path} batch={batch} tokens={tokens} frames={frames} oom")
                break
            point["ms_per_row"] = point["wall_ms"] / batch
            point["us_per_row_frame"] = point["wall_ms"] * 1e3 / (batch * frames)
            rows.append(point)
            print(
                f"{path} batch={batch:2d} tokens={tokens:4d} frames={frames:5d} "
                f"wall {point['wall_ms']:8.1f} ms  per row {point['ms_per_row']:7.1f} ms  "
                f"per row frame {point['us_per_row_frame']:6.1f} us  "
                f"peak {point['peak_gib']:6.2f} GiB  reserved {point['reserved_gib']:6.2f} GiB"
            )
    return rows


def sweep_hift(
    vocoder: CosyVoice3Vocoder, stream: Stream, device: torch.device
) -> list[dict[str, float | int | str]]:
    hop_frames = HISTORY_FRAMES[0]
    hop_mel = vocoder.hop_batch([flow_input(stream, 28)])[0][:, :, :hop_frames]
    samples_per_frame = int(vocoder.hift.istft_params["hop_len"]) * math.prod(
        vocoder.hift.upsample_rates
    )
    rows: list[dict[str, float | int | str]] = []
    for history in HISTORY_FRAMES:
        held = history - hop_frames
        hift_mel = hop_mel.repeat(1, 1, held // hop_frames) if held else None
        call = lambda: vocoder.hift_delta(
            hop_mel,
            hift_mel=hift_mel,
            speech_offset=held * samples_per_frame,
            finalize=False,
        )
        point: dict[str, float | int | str] = {
            "path": "hift",
            "history_frames": history,
        }
        point.update(measure(call, device))
        rows.append(point)
        print(
            f"hift history={history:5d} frames wall {point['wall_ms']:7.1f} ms  "
            f"peak {point['peak_gib']:6.2f} GiB  reserved {point['reserved_gib']:6.2f} GiB"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--flow-dtype", choices=sorted(AUTOCAST_DTYPES), default="bfloat16"
    )
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--json")
    args = parser.parse_args()

    device = torch.device(args.device)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(f"checkpoint {args.checkpoint}")
    print(f"branch head {head}")
    print(f"torch {torch.__version__}")
    print(f"device {args.device} {torch.cuda.get_device_name(device)}")
    print(f"flow autocast {args.flow_dtype}, hift float32, eager, no graph runner")

    flow, hift = load_cosyvoice3_flow_hift(args.checkpoint, args.device)
    _patch_chunk_mask()
    vocoder = CosyVoice3Vocoder(
        flow, hift, autocast_dtype=AUTOCAST_DTYPES[args.flow_dtype]
    )
    stream = build_streams(args.checkpoint, args.device, args.samples)[0]
    print(f"prompt {stream.sample_id}, {PROMPT_TOKENS} tokens")
    print(f"resident after load {torch.cuda.memory_allocated(device) / GIB:.2f} GiB")

    rows: list[dict] = []
    with torch.inference_mode():
        print("\n1. packed causal hop call")
        rows.extend(sweep_flow(vocoder, stream, "hop", device))
        print("\n2. packed non-streaming leftover call")
        rows.extend(sweep_flow(vocoder, stream, "leftover", device))
        print("\n3. hift over the accumulated mel history")
        rows.extend(sweep_hift(vocoder, stream, device))

    if args.json:
        with open(args.json, "w") as out:
            json.dump(
                {"head": head, "flow_dtype": args.flow_dtype, "rows": rows},
                out,
                indent=1,
            )


if __name__ == "__main__":
    main()
