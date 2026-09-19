#!/usr/bin/env python3
"""S3 gate: the cached hop replayed from captured steps against the eager one.

One process, the branch's own factory and its own startup capture. The cached
estimator decides per hop whether a captured step size holds it, so the eager
reference is the same estimator with its sizes taken away for the call. Over
the staggered schedule of the S1 gate, per step:

  eager      vocoder.hop_batch_cached with no step sizes
  replayed   the same call through the captured steps
  poisoned   the same again with every padded frame set to 1e3 and not zero

Reported: replayed against eager (bit identity where the step fills its size,
SNR where it is padded), poisoned against replayed (bit identity is the claim
that padded frames are inert), the time until the call returns and until the
device is done for both, the same two times with a Python thread spinning
beside the call, launches per call from the profiler, and what the startup
capture cost in seconds and MiB.

  python s3_step_graph_gate.py --out .tmp/s3 --streams 8 --steps 6
  python s3_step_graph_gate.py --out .tmp/s3 --streams 16 --steps 2 --stagger 1
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import sys
import threading
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage0"))
from common import MODEL_ID, build_streams, extend_tokens, provenance  # noqa: E402

from sglang_omni.models.fun_cosyvoice3 import flow_hop_cache  # noqa: E402
from sglang_omni.models.fun_cosyvoice3.flow_hop_cache import CachedDiT  # noqa: E402
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


def timed(call, repeats: int, before_repeat):
    """The call's result, and the medians in ms of the time until it returns
    and the time until the device is done."""
    returned, done = [], []
    for _ in range(repeats):
        before_repeat()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = call()
        returned.append((time.perf_counter() - started) * 1e3)
        torch.cuda.synchronize()
        done.append((time.perf_counter() - started) * 1e3)
    return result, statistics.median(returned), statistics.median(done)


def snr_db(value: torch.Tensor, reference: torch.Tensor) -> float:
    error = (value.float() - reference.float()).norm()
    if error == 0:
        return float("inf")
    return float(20 * torch.log10(reference.float().norm() / error))


def launches(call, before) -> dict[str, int]:
    """CUDA runtime calls of one call, by name."""
    before()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        call()
        torch.cuda.synchronize()
    counts: dict[str, int] = {}
    for event in profile.key_averages():
        if event.key.startswith("cuda") and "Synchronize" not in event.key:
            counts[event.key] = event.count
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--stagger", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1088)
    parser.add_argument("--spin-step", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

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

    # The buffered solver graphs are off so the capture below is the only one
    # and its memory can be read off the card.
    budget = int(sum(final_frames) * BYTES_PER_FRAME * 1.1)
    device_type, _, card = args.device.partition(":")
    scheduler = create_vocoder_executor(
        args.model,
        device=device_type,
        gpu_id=int(card or 0),
        enable_flow_cuda_graph=False,
        flow_kv_cache_bytes=budget,
    )
    vocoder = scheduler.vocoder
    cache = vocoder.flow_hop_cache
    flow = vocoder.flow
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_before, _ = torch.cuda.mem_get_info()
    started = time.perf_counter()
    flow.cached_estimator = CachedDiT(
        flow.decoder.estimator, cache, device=args.device, capture_steps=True
    )
    # The scheduler's own capture loop, one size at a time, so the card's free
    # memory can be read after each.
    estimator = flow.cached_estimator
    smallest = FlowBatchInput(
        token=torch.zeros(1, TOKEN_HOP_LEN + PRE_LOOKAHEAD_LEN, dtype=torch.int32),
        prompt_token=torch.zeros(1, 0, dtype=torch.int32),
        prompt_feat=torch.zeros(1, 0, flow.output_size),
        embedding=torch.zeros(1, flow.spk_embed_affine_layer.in_features),
    )
    info["capture_mib_by_size"] = {}
    for size in (*reversed(estimator.sizes), None):
        estimator.capture_size = size
        free_at, _ = torch.cuda.mem_get_info()
        handle = cache.open_stream()
        cache.reserve(handle, TOKEN_HOP_LEN * TOKEN_MEL_RATIO)
        vocoder.hop_batch_cached([smallest], [handle])
        cache.release(handle)
        torch.cuda.synchronize()
        if size is not None:
            info["capture_mib_by_size"][size] = (
                free_at - torch.cuda.mem_get_info()[0]
            ) / 2**20
    info["capture_s"] = time.perf_counter() - started
    info["capture_mib"] = (free_before - torch.cuda.mem_get_info()[0]) / 2**20
    info["sizes"] = list(estimator.sizes)
    info["captured"] = len(estimator.graphs.entries)
    info["segments"] = sorted(
        {entry.num_segments for entry in estimator.graphs.entries.values()}
    )
    info["sglang_omni"] = inspect.getsourcefile(type(vocoder))
    info["budget_gib"] = budget / 2**30
    info["slots"] = cache.slots
    print(json.dumps(info, indent=1))

    sizes = estimator.sizes
    zero_pad = flow_hop_cache.pad_frames

    def poisoned_pad(packed: torch.Tensor, frames: int) -> torch.Tensor:
        if packed.shape[1] == frames:
            return packed
        return F.pad(packed, (0, 0, 0, frames - packed.shape[1]), value=1e3)

    stop = threading.Event()

    def spin() -> None:
        count = 0
        while not stop.is_set():
            count += 1

    handles = {}
    rows_out: list[dict] = []
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
        tails = cache.conv_tails[:, :, lanes].clone()

        def rewind():
            # A repeat rewrites the same slots with the same K and V; only the
            # frame cursor and the conv tails have to go back.
            for handle, start in zip(row_handles, starts, strict=True):
                handle.frames = start
            cache.conv_tails[:, :, lanes] = tails

        def call():
            return vocoder.hop_batch_cached(items, row_handles)

        new_frames = sum(
            handle.reserved - start for handle, start in zip(row_handles, starts)
        )
        packed = 2 * new_frames
        size = next((size for size in sizes if size >= packed), None)
        row = {
            "step": step,
            "rows": len(items),
            "packed_frames": packed,
            "step_size": size,
            "window_frames": sum(handle.reserved for handle in row_handles),
        }

        estimator.sizes = ()
        eager, row["eager_returned_ms"], row["eager_done_ms"] = timed(
            call, args.repeats, rewind
        )
        row["eager_launches"] = launches(call, rewind)
        estimator.sizes = sizes
        replayed, row["replayed_returned_ms"], row["replayed_done_ms"] = timed(
            call, args.repeats, rewind
        )
        row["replayed_launches"] = launches(call, rewind)
        flow_hop_cache.pad_frames = poisoned_pad
        rewind()
        poisoned = call()
        flow_hop_cache.pad_frames = zero_pad

        if step == args.spin_step:
            # One repeat: an eager call beside a spinning thread takes minutes.
            stop.clear()
            spinner = threading.Thread(target=spin, daemon=True)
            spinner.start()
            estimator.sizes = ()
            _, row["eager_spin_returned_ms"], row["eager_spin_done_ms"] = timed(
                call, 1, rewind
            )
            estimator.sizes = sizes
            _, row["replayed_spin_returned_ms"], row["replayed_spin_done_ms"] = timed(
                call, 1, rewind
            )
            stop.set()
            spinner.join()

        row["replayed_equals_eager"] = all(
            torch.equal(a, b) for a, b in zip(replayed, eager, strict=True)
        )
        row["replayed_snr_db_min"] = min(
            snr_db(a, b) for a, b in zip(replayed, eager, strict=True)
        )
        row["poisoned_equals_replayed"] = all(
            torch.equal(a, b) for a, b in zip(poisoned, replayed, strict=True)
        )
        row["finite"] = all(bool(torch.isfinite(mel).all()) for mel in replayed)
        rows_out.append(row)
        brief = {
            key: (round(value, 1) if isinstance(value, float) else value)
            for key, value in row.items()
            if not key.endswith("launches")
        }
        brief["eager_launch_count"] = sum(row["eager_launches"].values())
        brief["replayed_launch_count"] = sum(row["replayed_launches"].values())
        print(json.dumps(brief))

    with open(os.path.join(args.out, "s3_gate.json"), "w") as out:
        json.dump({"info": info, "steps": rows_out}, out, indent=1)


if __name__ == "__main__":
    main()
