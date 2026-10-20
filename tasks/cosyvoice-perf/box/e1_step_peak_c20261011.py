# SPDX-License-Identifier: Apache-2.0
"""E1: does item 3 v3's largest step measurement, run under force_eager, bound what serving's compiled
path allocates for the same step?

The vocoder is built by create_vocoder_executor as serving builds it (16 rows, compile on, buffered and
whole history graphs), without the prefix pool (flow_prefix_cache_gb=0), so the card has room. Then for
16 rows of N tokens, the body of measure_peak_step_bytes (uncached hops, then finals over the whole
history, each followed by HiFT over the last hop) runs once under force_eager and once compiled. Per run:
peak bytes beyond the resident set, wall time, and Dynamo's counters before and after (a new graph means
serving would compile at that size).

    PYTHONPATH=/workspace/t_p123v3 python e1_step_peak_c20261011.py --out /data/c20261011/e1/e1.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time

import torch
from torch._dynamo.utils import counters

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.models.fun_cosyvoice3.stages import HiftStepRow

ROWS = 16
TOKENS = (1024, 2048, 4096)


def dynamo_state() -> dict[str, int]:
    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "frames_ok": int(counters["frames"]["ok"]),
        "recompiles": sum(int(v) for v in counters["recompiles"].values()),
    }


def run_step(scheduler, device: torch.device, tokens: int, *, eager: bool) -> dict:
    vocoder = scheduler.vocoder
    hop_frames = scheduler.token_max_hop_len * vocoder.flow.token_mel_ratio
    samples_per_frame = vocoder.hift_samples_per_mel_frame
    items = [
        scheduler.make_warmup_flow_input(tokens - scheduler.token_hop_len)
        for _ in range(ROWS)
    ]
    torch.cuda.synchronize(device)
    resident_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    before = dynamo_state()
    stance = torch.compiler.set_stance("force_eager") if eager else contextlib.nullcontext()
    started = time.monotonic()
    with vocoder.stream_context, stance:
        for decode, is_final in ((vocoder.hop_batch, False), (vocoder.leftover_batch, True)):
            mels = decode(items)
            vocoder.hift_step(
                [
                    HiftStepRow(
                        history=mel,
                        emitted_samples=max(int(mel.shape[2]) - hop_frames, 0)
                        * samples_per_frame,
                        is_final=is_final,
                    )
                    for mel in mels
                ]
            )
    torch.cuda.synchronize(device)
    wall_s = time.monotonic() - started
    peak_bytes = torch.cuda.max_memory_allocated(device) - resident_bytes
    after = dynamo_state()
    return {
        "peak_gib": round(peak_bytes / 2**30, 3),
        "wall_s": round(wall_s, 2),
        "dynamo_before": before,
        "dynamo_after": after,
        "reserved_gib": round(torch.cuda.memory_reserved(device) / 2**30, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda", 0)
    # note(ratish): the vocoder stage's factory arguments in config.py, the pool off.
    scheduler = stages.create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        dtype="bfloat16",
        max_batch_size=ROWS,
        max_batch_wait_ms=30,
        flow_batch_admission_frames=8000,
        flow_merge_max_gap_frames=384,
        flow_merge_pad_budget_percent=25.0,
        enable_flow_cuda_graph=True,
        enable_flow_prefix_cuda_graph=True,
        enable_flow_whole_history_cuda_graph=True,
        flow_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
        enable_flow_estimator_trt=False,
        token_hop_len=25,
        token_max_hop_len=100,
        disable_hop_growth=False,
        flow_prefix_cache_gb=0,
        flow_prefix_cache_reserve_gb=0.0,
    )
    print(f"tree={stages.__file__} context={FunCosyVoice3EngineBuilder.context_length}", flush=True)
    report: dict[str, dict] = {}
    with torch.inference_mode():
        for tokens in TOKENS:
            for eager in (True, False):
                name = f"rows={ROWS} tokens={tokens} {'eager' if eager else 'compiled'}"
                report[name] = run_step(scheduler, device, tokens, eager=eager)
                print(name, report[name], flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
