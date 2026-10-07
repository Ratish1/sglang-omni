# SPDX-License-Identifier: Apache-2.0
"""Final step graph replay against the eager final solve, per frame tier of a candidate
ladder, in the serving configuration (DiT compile on, final graphs on; buffered and
prefix graphs off, they do not touch this path). For each tier: the replay wall time with a
device sync per call (the replay's GPU time plus its copy in), at one row filling the tier
and at the most rows, and the eager solve's wall time with and without a sync per call at
the same rows. The eager call without a sync is its host launch time. Also the capture time
and memory of the whole ladder, from the runner's log line.

    cd <finals tree> && python final_tiers.py --out <json> [--tiers 48 64 ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.packed_dit import (
    pack_rows,
    solve_flow_euler_packed,
)


def sglang_prefill_ladder(max_frames: int) -> list[int]:
    return [
        frames
        for frames in (
            list(range(48, 257, 16))
            + list(range(288, 513, 32))
            + list(range(576, 1025, 64))
            + list(range(1280, 4097, 256))
            + list(range(4608, max_frames + 1, 512))
        )
        if frames <= max_frames
    ]


def split_rows(frames: int, rows: int) -> tuple[int, ...]:
    base = frames // rows
    return tuple(
        base + (1 if index < frames - base * rows else 0) for index in range(rows)
    )


def timed(call, *, calls: int, warmup: int, synchronize: bool) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    walls = []
    for _ in range(calls):
        started = time.perf_counter()
        call()
        if synchronize:
            torch.cuda.synchronize()
        else:
            pass
        walls.append(time.perf_counter() - started)
    torch.cuda.synchronize()
    return 1000 * statistics.median(walls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--tiers", nargs="+", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=8192)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--calls", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    tiers = args.tiers or sglang_prefill_ladder(args.max_frames)
    stages.FINAL_TIER_FRAMES = tuple(tiers)
    started = time.perf_counter()
    scheduler = stages.create_vocoder_executor(
        args.model,
        device="cuda",
        gpu_id=0,
        max_batch_size=args.max_rows,
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=False,
        enable_flow_final_cuda_graph=True,
        flow_prefix_cache_gb=0.0,
    )
    startup_s = time.perf_counter() - started
    vocoder = scheduler.vocoder
    runner = vocoder.flow.final_cuda_graph_runner
    assert runner is not None and list(runner.tier_frames) == sorted(tiers)
    estimator = vocoder.flow.packed_estimator
    device = runner.device
    generator = torch.Generator(device=device).manual_seed(0)
    unit = torch.linspace(0, 1, 11, device=device)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).to(runner.frame_dtype)
    results = []
    for tier in runner.tier_frames:
        row = {"tier": tier}
        for label, rows in (("1", 1), ("max", min(args.max_rows, tier // 25))):
            lengths = split_rows(tier, rows)
            shape = (1, tier, runner.mel_channels)
            inputs = dict(
                noise=torch.randn(shape, device=device, generator=generator).to(
                    runner.frame_dtype
                ),
                time_span=time_span,
                mu=torch.randn(shape, device=device, generator=generator).to(
                    runner.frame_dtype
                ),
                speaker_embeddings=torch.randn(
                    rows, runner.speaker_channels, device=device, generator=generator
                ).to(runner.speaker_dtype),
                mel_conditioning=torch.randn(
                    shape, device=device, generator=generator
                ).to(runner.frame_dtype),
            )
            twin_rows = pack_rows(lengths * 2, device)

            def replay():
                return runner.run(**inputs, lengths=lengths)

            def eager():
                with torch.autocast("cuda", dtype=vocoder.autocast_dtype):
                    return solve_flow_euler_packed(
                        estimator,
                        inputs["noise"],
                        inputs["time_span"],
                        inputs["mu"],
                        inputs["speaker_embeddings"],
                        inputs["mel_conditioning"],
                        twin_rows,
                        estimator.row_attention(
                            twin_rows, streaming=False, dtype=runner.speaker_dtype
                        ),
                        estimator.rope_angles(twin_rows.width),
                        cfg_rate=runner.cfg_rate,
                    )

            with torch.inference_mode(), vocoder.stream_context:
                replayed = replay()
                expected = eager()
                row[f"rows_{label}"] = rows
                row[f"max_abs_diff_{label}"] = float((replayed - expected).abs().max())
                row[f"replay_ms_{label}"] = timed(
                    replay,
                    calls=args.calls,
                    warmup=args.warmup,
                    synchronize=True,
                )
                row[f"eager_ms_{label}"] = timed(
                    eager, calls=args.calls, warmup=args.warmup, synchronize=True
                )
                row[f"eager_host_ms_{label}"] = timed(
                    eager, calls=args.calls, warmup=args.warmup, synchronize=False
                )
        print(json.dumps(row), flush=True)
        results.append(row)
    with open(args.out, "w") as handle:
        json.dump({"startup_s": startup_s, "tiers": results}, handle, indent=1)


if __name__ == "__main__":
    main()
