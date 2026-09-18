#!/usr/bin/env python3
"""S1 gate: the shipped hop K/V cache against the hop main serves.

Drives the branch's own code, built by its own factory
(`create_vocoder_executor(..., flow_kv_cache_bytes=...)`), over a staggered
multi row schedule whose hops grow the way serving grows them. Every hop runs:

  truth       `flow.inference_causal` in float32, for the first --truth-steps
  production  `vocoder.hop_batch`, the whole prefix, bfloat16 autocast
  cached      `vocoder.hop_batch_cached`, the new frames only

Reported per hop and row: cached against production (bit identity is expected
on a stream's first hop, where both run the same rows through the same
kernels), each against the truth, and per step the synchronized wall time and
the peak allocated memory of both calls.

  python s1_hop_cache_gate.py --out .tmp/s1-gate --streams 8 --steps 6
  python s1_hop_cache_gate.py --out .tmp/s1-long --streams 2 --steps 16 --truth-steps 4
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage0"))
from common import (  # noqa: E402
    MODEL_ID,
    build_streams,
    compare,
    extend_tokens,
    provenance,
)

from sglang_omni.models.fun_cosyvoice3.stages import (  # noqa: E402
    FlowBatchInput,
    create_vocoder_executor,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (  # noqa: E402
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MEL_RATIO,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint  # noqa: E402

PROMPT_TOKENS = (40, 55, 63, 78, 91, 110, 127, 144)
BYTES_PER_FRAME = 1_802_240


def measured(call, repeats: int, before_repeat=None):
    """The call's result, its median synchronized wall in ms, and the peak
    allocated MiB above what was allocated when it started."""
    samples = []
    peak = 0
    for _ in range(repeats):
        if before_repeat is not None:
            before_repeat()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        started = time.perf_counter()
        result = call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1e3)
        peak = max(peak, torch.cuda.max_memory_allocated() - base)
    return result, statistics.median(samples), peak / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--stagger", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--truth-steps", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1088)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    truth_steps = args.steps if args.truth_steps is None else args.truth_steps

    os.makedirs(args.out, exist_ok=True)
    info = provenance(args.device)
    checkpoint = resolve_checkpoint(args.model)

    prompts = tuple(
        PROMPT_TOKENS[index % len(PROMPT_TOKENS)] for index in range(args.streams)
    )
    hop, tokens = TOKEN_HOP_LEN, 0
    for _ in range(args.steps):
        tokens += hop
        hop = next_stream_hop_len(hop)
    references = build_streams(
        checkpoint,
        args.device,
        count=4 * args.streams,
        prompt_tokens=prompts * 4,
        min_generated=TOKEN_HOP_LEN + PRE_LOOKAHEAD_LEN,
        samples=args.samples,
    )
    streams = extend_tokens(references, args.streams, tokens + PRE_LOOKAHEAD_LEN)
    prompts_padded = [
        pad_flow_prompt_to_hop(
            stream.prompt_token, stream.prompt_feat, hop_len=TOKEN_HOP_LEN
        )
        for stream in streams
    ]

    # The schedule: row index joins at step index % stagger, then hops grow.
    plan: list[list[tuple[int, int, int]]] = []
    state = [(0, TOKEN_HOP_LEN) for _ in streams]
    final_frames = [0 for _ in streams]
    for step in range(args.steps):
        participants = []
        for index, stream in enumerate(streams):
            offset, hop = state[index]
            if step < index % args.stagger:
                continue
            if offset + hop + PRE_LOOKAHEAD_LEN > int(stream.tokens.shape[1]):
                continue
            participants.append((index, offset, hop))
            prompt_len = int(prompts_padded[index][0].shape[1])
            final_frames[index] = (prompt_len + offset + hop) * TOKEN_MEL_RATIO
            state[index] = (offset + hop, next_stream_hop_len(hop))
        plan.append(participants)

    budget = int(sum(final_frames) * BYTES_PER_FRAME * 1.1)
    scheduler = create_vocoder_executor(
        args.model,
        device=args.device,
        enable_flow_cuda_graph=False,
        flow_kv_cache_bytes=budget,
    )
    vocoder = scheduler.vocoder
    cache = vocoder.flow_hop_cache
    flow = vocoder.flow
    info["sglang_omni"] = inspect.getsourcefile(type(vocoder))
    info["budget_gib"] = budget / 2**30
    info["slots"] = cache.slots
    print(f"sglang_omni {info['sglang_omni']}")
    print(f"budget {budget / 2**30:.2f} GiB, {cache.slots} slots")

    handles = {}
    hops: list[dict] = []
    steps_detail: list[dict] = []
    for step, participants in enumerate(plan):
        if not participants:
            continue
        items = [
            FlowBatchInput(
                token=streams[index].tokens[:, : offset + hop + PRE_LOOKAHEAD_LEN],
                prompt_token=prompts_padded[index][0],
                prompt_feat=prompts_padded[index][1],
                embedding=streams[index].embedding,
            )
            for index, offset, hop in participants
        ]
        truth = None
        if step < truth_steps:
            with torch.autocast(device_type="cuda", enabled=False):
                truth = flow.inference_causal(items)
        production, production_ms, production_mib = measured(
            lambda: vocoder.hop_batch(items), args.repeats
        )

        for index, _, _ in participants:
            if index not in handles:
                handles[index] = cache.open_stream()
        row_handles = [handles[index] for index, _, _ in participants]
        starts = [handle.frames for handle in row_handles]
        for handle, (index, offset, hop) in zip(row_handles, participants, strict=True):
            prompt_len = int(prompts_padded[index][0].shape[1])
            if not cache.reserve(handle, (prompt_len + offset + hop) * TOKEN_MEL_RATIO):
                raise RuntimeError("the probe's pool is too small for its schedule")
        lanes = [lane for handle in row_handles for lane in handle.lanes]
        tails = cache.conv_tails[:, :, lanes]

        def rewind():
            # A repeat rewrites the same slots with the same K and V; only the
            # frame cursor and the conv tails have to go back.
            for handle, start in zip(row_handles, starts, strict=True):
                handle.frames = start
            cache.conv_tails[:, :, lanes] = tails

        cached, cached_ms, cached_mib = measured(
            lambda: vocoder.hop_batch_cached(items, row_handles), args.repeats, rewind
        )

        for row, (index, offset, hop) in enumerate(participants):
            emitted = production[row][:, :, offset * TOKEN_MEL_RATIO :]
            record = {
                "step": step,
                "rows": len(participants),
                "row": index,
                "sample_id": streams[index].sample_id,
                "hop": hop,
                "token_offset": offset,
                "first_hop": starts[row] == 0,
                "new_frames": row_handles[row].frames - starts[row],
                "window_frames": row_handles[row].frames,
                "finite": bool(torch.isfinite(cached[row]).all()),
                "cached_vs_production": compare(
                    cached[row].float().cpu(), emitted.float().cpu()
                ),
            }
            if truth is not None:
                reference = truth[row][:, :, offset * TOKEN_MEL_RATIO :].float().cpu()
                record["production_vs_truth_db"] = compare(
                    emitted.float().cpu(), reference
                )["snr_db"]
                record["cached_vs_truth_db"] = compare(
                    cached[row].float().cpu(), reference
                )["snr_db"]
            hops.append(record)
        steps_detail.append(
            {
                "step": step,
                "rows": len(participants),
                "new_frames": sum(
                    handle.frames - start
                    for handle, start in zip(row_handles, starts, strict=True)
                ),
                "window_frames": sum(handle.frames for handle in row_handles),
                "production_ms": production_ms,
                "cached_ms": cached_ms,
                "production_peak_mib": production_mib,
                "cached_peak_mib": cached_mib,
            }
        )
        print(
            f"step {step}: {len(participants)} rows, "
            f"{steps_detail[-1]['new_frames']} new of "
            f"{steps_detail[-1]['window_frames']} frames, "
            f"production {production_ms:.1f} ms {production_mib:.0f} MiB, "
            f"cached {cached_ms:.1f} ms {cached_mib:.0f} MiB"
        )

    free_before = cache.allocator.available_size()
    for handle in handles.values():
        cache.release(handle)
    first = [record for record in hops if record["first_hop"]]
    later = [record for record in hops if not record["first_hop"]]
    with_truth = [record for record in hops if "cached_vs_truth_db" in record]
    report = {
        "provenance": info,
        "args": vars(args),
        "first_hops": len(first),
        "first_hops_bit_identical": sum(
            record["cached_vs_production"]["equal"] for record in first
        ),
        "later_hops": len(later),
        "later_hops_min_snr_db_vs_production": min(
            (record["cached_vs_production"]["snr_db"] for record in later),
            default=None,
        ),
        "all_finite": all(record["finite"] for record in hops),
        "truth_hops": len(with_truth),
        "production_vs_truth_median_db": statistics.median(
            record["production_vs_truth_db"] for record in with_truth
        ),
        "cached_vs_truth_median_db": statistics.median(
            record["cached_vs_truth_db"] for record in with_truth
        ),
        "production_vs_truth_min_db": min(
            record["production_vs_truth_db"] for record in with_truth
        ),
        "cached_vs_truth_min_db": min(
            record["cached_vs_truth_db"] for record in with_truth
        ),
        "slots_used_at_end": cache.slots - free_before,
        "slots_free_after_release": cache.allocator.available_size(),
        "rows_free_after_release": cache.rows.available_size(),
        "production_ms_total": sum(step["production_ms"] for step in steps_detail),
        "cached_ms_total": sum(step["cached_ms"] for step in steps_detail),
        "steps": steps_detail,
        "hops": hops,
    }
    with open(os.path.join(args.out, "s1_gate.json"), "w") as out:
        json.dump(report, out, indent=1)
    for key, value in report.items():
        if key not in ("hops", "steps", "provenance", "args"):
            print(f"{key} {value}")


if __name__ == "__main__":
    main()
