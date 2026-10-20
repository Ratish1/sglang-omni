# SPDX-License-Identifier: Apache-2.0
"""Per step cost of the ways to run a stream step past a solve graph runner's largest tier, on
the real model in the serving configuration (DiT compile on, prefix and whole history graphs on,
the default 24 GB prefix pool): eager (before the split replay), the split replay (now), and one
replay of a graph whose tier holds the whole step. The shapes are steps that missed in serving,
read from graph_hits probe logs. Wall time per call with a device sync (the step's latency),
median of the calls, plus the eager call without a sync (its host launch time). Prefix rows get
their logged cached history first and are reset before every call.

    cd <tree> && PYTHONPATH=<tree> python3 step_strategies.py --whole-log <serve.log> \
        --prefix-log <serve.log> --out <json>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time

import torch
from sglang.srt.arg_groups.cuda_graph_hook import generate_prefill_cuda_graph_batch_sizes

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.packed_dit import pack_rows, solve_flow_euler_packed
from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    PrefixCacheRow,
    grow_rows,
    release_rows,
    solve_flow_euler_prefix,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import PrefixCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.streaming import TOKEN_HOP_LEN, TOKEN_MAX_HOP_LEN
from sglang_omni.models.fun_cosyvoice3.whole_history_cuda_graph import (
    WholeHistoryCudaGraphRunner,
)
from sglang_omni.platforms import current_platform


def missed_steps(log: str, kind: str) -> list[dict]:
    steps = []
    for line in open(log, errors="replace"):
        match = re.search(rf"graph\.{kind} t=\S+ (.*)", line)
        if not match:
            continue
        fields = dict(item.split("=") for item in match.group(1).split())
        if fields["hit"] == "1":
            continue
        step = {"lengths": [int(v) for v in fields["lengths"].split(",")]}
        if "committed" in fields:
            step["committed"] = [int(v) for v in fields["committed"].split(",")]
        steps.append(step)
    return steps


def sample(steps: list[dict], count: int) -> list[dict]:
    unique = {json.dumps(step, sort_keys=True): step for step in steps}
    ordered = sorted(unique.values(), key=lambda step: sum(step["lengths"]))
    if len(ordered) <= count:
        return ordered
    else:
        return [ordered[round(i * (len(ordered) - 1) / (count - 1))] for i in range(count)]


def groups(lengths: list[int], top: int, max_rows: int) -> int:
    count, frames, rows = 1, 0, 0
    for length in lengths:
        if frames + length > top or rows == max_rows:
            count, frames, rows = count + 1, 0, 0
        frames, rows = frames + length, rows + 1
    return count


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
    parser.add_argument("--whole-log", required=True)
    parser.add_argument("--prefix-log", required=True)
    parser.add_argument("--shapes", type=int, default=8)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--whole-cap", type=int, default=10240)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scheduler = stages.create_vocoder_executor(
        args.model,
        device="cuda",
        gpu_id=0,
        max_batch_size=16,
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=True,
        enable_flow_whole_history_cuda_graph=True,
        flow_prefix_cache_gb=24.0,
    )
    vocoder = scheduler.vocoder
    flow = vocoder.flow
    estimator = flow.packed_estimator
    pool = flow.prefix_pool
    whole = flow.whole_history_cuda_graph_runner
    prefix = flow.prefix_cuda_graph_runner
    device = whole.device
    autocast = vocoder.autocast_dtype
    backend = current_platform.get_device_graph_backend(device)
    whole_single = WholeHistoryCudaGraphRunner(
        estimator,
        graphs=SolveGraphCapture(backend, device=device, autocast_dtype=autocast),
        frame_dtype=whole.frame_dtype,
        speaker_dtype=whole.speaker_dtype,
        cfg_rate=whole.cfg_rate,
        euler_steps=whole.euler_steps,
        mel_channels=whole.mel_channels,
        speaker_channels=whole.speaker_channels,
        max_rows=whole.max_rows,
        tier_frames=[
            tier
            for tier in generate_prefill_cuda_graph_batch_sizes(args.whole_cap)
            if tier > whole.tier_frames[-1]
        ],
    )
    ratio = flow.token_mel_ratio
    prefix_single = PrefixCudaGraphRunner(
        estimator,
        pool,
        graphs=SolveGraphCapture(backend, device=device, autocast_dtype=autocast),
        frame_dtype=prefix.frame_dtype,
        speaker_dtype=prefix.speaker_dtype,
        cfg_rate=prefix.cfg_rate,
        mel_channels=prefix.mel_channels,
        speaker_channels=prefix.speaker_channels,
        max_rows=2 * prefix.max_rows,
        hop_frames=TOKEN_MAX_HOP_LEN * ratio,
        min_hop_frames=TOKEN_HOP_LEN * ratio,
        max_frames=flow.decoder.rand_noise.shape[2],
    )
    with torch.inference_mode(), vocoder.stream_context:
        whole_single.capture()
        prefix_single.capture()
    generator = torch.Generator(device=device).manual_seed(0)

    def frames_input(total: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(
            1, total, whole.mel_channels, device=device, generator=generator
        ).to(dtype)

    results = []
    for step in sample(missed_steps(args.whole_log, "whole"), args.shapes):
        lengths = tuple(step["lengths"])
        total = sum(lengths)
        unit = torch.linspace(0, 1, whole.euler_steps + 1, device=device)
        inputs = dict(
            noise=frames_input(total, whole.frame_dtype),
            time_span=(1 - torch.cos(unit * 0.5 * torch.pi)).to(whole.frame_dtype),
            mu=frames_input(total, whole.frame_dtype),
            speaker_embeddings=torch.randn(
                len(lengths), whole.speaker_channels, device=device, generator=generator
            ).to(whole.speaker_dtype),
            mel_conditioning=frames_input(total, whole.frame_dtype),
        )

        def eager():
            twin_rows = pack_rows(list(lengths) * 2, device)
            return solve_flow_euler_packed(
                estimator,
                inputs["noise"],
                inputs["time_span"],
                inputs["mu"],
                inputs["speaker_embeddings"],
                inputs["mel_conditioning"],
                twin_rows,
                estimator.row_attention(
                    twin_rows, streaming=False, dtype=whole.speaker_dtype
                ),
                estimator.rope_angles(twin_rows.width),
                cfg_rate=whole.cfg_rate,
            )

        def split():
            return whole.run(**inputs, lengths=lengths)

        def single():
            return whole_single.run(**inputs, lengths=lengths)

        with torch.inference_mode(), vocoder.stream_context, torch.autocast(
            "cuda", dtype=autocast
        ):
            reference = eager()
            row = {
                "kind": "whole",
                "rows": len(lengths),
                "frames": total,
                "groups": groups(list(lengths), whole.tier_frames[-1], whole.max_rows),
                "split_max_abs_diff": float((split() - reference).abs().max()),
                "single_max_abs_diff": float((single() - reference).abs().max()),
                "eager_ms": timed(eager, calls=args.calls, warmup=args.warmup, synchronize=True),
                "eager_host_ms": timed(
                    eager, calls=args.calls, warmup=args.warmup, synchronize=False
                ),
                "split_ms": timed(split, calls=args.calls, warmup=args.warmup, synchronize=True),
                "single_ms": timed(single, calls=args.calls, warmup=args.warmup, synchronize=True),
            }
        print(json.dumps(row), flush=True)
        results.append(row)

    for step in sample(missed_steps(args.prefix_log, "prefix"), args.shapes):
        lengths = list(step["lengths"])
        committed = list(step["committed"])
        pairs = [(PrefixCacheRow(), PrefixCacheRow()) for _ in lengths]
        rows = [row for pair in pairs for row in pair]
        unit = torch.linspace(0, 1, prefix.euler_steps + 1, device=device)
        time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).to(prefix.frame_dtype)
        with torch.inference_mode(), vocoder.stream_context, torch.autocast(
            "cuda", dtype=autocast
        ):
            held = [index for index, frames in enumerate(committed) if frames > 0]
            if held:
                for index in held:
                    assert grow_rows(pool, list(pairs[index]), [committed[index]] * 2)
                held_frames = [committed[index] for index in held]
                solve_flow_euler_prefix(
                    estimator,
                    pool,
                    frames_input(sum(held_frames), prefix.frame_dtype),
                    time_span,
                    frames_input(sum(held_frames), prefix.frame_dtype),
                    torch.randn(
                        len(held), prefix.speaker_channels, device=device, generator=generator
                    ).to(prefix.speaker_dtype),
                    frames_input(sum(held_frames), prefix.frame_dtype),
                    held_frames,
                    [pairs[index] for index in held],
                    cfg_rate=prefix.cfg_rate,
                )
            else:
                pass
            assert [pair[0].committed_frames for pair in pairs] == committed
            for pair, held_frames, new in zip(pairs, committed, lengths, strict=True):
                assert grow_rows(pool, list(pair), [held_frames + new] * 2)
            saved = [(row.committed_frames, row.conv_context) for row in rows]
            total = sum(lengths)
            inputs = dict(
                noise=frames_input(total, prefix.frame_dtype),
                time_span=time_span,
                mu=frames_input(total, prefix.frame_dtype),
                speaker_embeddings=torch.randn(
                    len(lengths), prefix.speaker_channels, device=device, generator=generator
                ).to(prefix.speaker_dtype),
                mel_conditioning=frames_input(total, prefix.frame_dtype),
            )

            def restore():
                for row, (frames, context) in zip(rows, saved, strict=True):
                    row.committed_frames = frames
                    row.conv_context = context

            def eager():
                restore()
                return solve_flow_euler_prefix(
                    estimator,
                    pool,
                    **inputs,
                    new_frames=lengths,
                    caches=pairs,
                    cfg_rate=prefix.cfg_rate,
                )

            def split():
                restore()
                return prefix.run(**inputs, new_frames=lengths, caches=pairs)

            def single():
                restore()
                return prefix_single.run(**inputs, new_frames=lengths, caches=pairs)

            reference = eager()
            row = {
                "kind": "prefix",
                "rows": len(lengths),
                "frames": total,
                "first_hops": sum(frames == 0 for frames in committed),
                "groups": groups(lengths, prefix.tier_frames[-1], prefix.max_rows),
                "split_max_abs_diff": float((split() - reference).abs().max()),
                "single_max_abs_diff": float((single() - reference).abs().max()),
                "eager_ms": timed(eager, calls=args.calls, warmup=args.warmup, synchronize=True),
                "eager_host_ms": timed(
                    eager, calls=args.calls, warmup=args.warmup, synchronize=False
                ),
                "split_ms": timed(split, calls=args.calls, warmup=args.warmup, synchronize=True),
                "single_ms": timed(single, calls=args.calls, warmup=args.warmup, synchronize=True),
            }
            restore()
            release_rows(pool, rows)
        print(json.dumps(row), flush=True)
        results.append(row)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
