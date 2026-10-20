# SPDX-License-Identifier: Apache-2.0
"""E1b: how much device memory the largest admitted step maps, against the bytes it holds live.

E2 showed the vocoder stream mapping 9.98 GiB during item 3 v3's startup measurement whose live peak was
6.44 GiB: with expandable segments a step needs its mapped pages, not its live bytes. The vocoder is built
as serving builds it (compile on, buffered and whole history graphs, ROWS rows), without the prefix pool.
For ROWS rows of N tokens the body of measure_peak_step_bytes runs compiled, then under force_eager, each
from a cache emptied before it: per run the peak of torch's mapped bytes above the start, the peak of live
bytes above the start, the wall time and Dynamo's counters before and after.

    PYTHONPATH=/workspace/t_p123v3 python e1b_step_mapped_c20261011.py --rows 16 --out e1b.json
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

TOKENS = (1024, 2048, 4096)


def dynamo_state() -> dict[str, int]:
    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "frames_ok": int(counters["frames"]["ok"]),
        "recompiles": sum(int(v) for v in counters["recompiles"].values()),
    }


def run_step(
    scheduler, device: torch.device, rows: int, tokens: int, *, eager: bool
) -> dict:
    vocoder = scheduler.vocoder
    hop_frames = scheduler.token_max_hop_len * vocoder.flow.token_mel_ratio
    samples_per_frame = vocoder.hift_samples_per_mel_frame
    items = [
        scheduler.make_warmup_flow_input(tokens - scheduler.token_hop_len)
        for _ in range(rows)
    ]
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    resident_bytes = torch.cuda.memory_allocated(device)
    mapped_bytes = torch.cuda.memory_reserved(device)
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
    peak_mapped_bytes = torch.cuda.max_memory_reserved(device) - mapped_bytes
    after = dynamo_state()
    return {
        "peak_live_gib": round(peak_bytes / 2**30, 3),
        "peak_mapped_gib": round(peak_mapped_bytes / 2**30, 3),
        "wall_s": round(wall_s, 2),
        "dynamo_before": before,
        "dynamo_after": after,
        "reserved_gib": round(torch.cuda.memory_reserved(device) / 2**30, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
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
        max_batch_size=args.rows,
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
            for eager in (False, True):
                name = f"rows={args.rows} tokens={tokens} {'eager' if eager else 'compiled'}"
                report[name] = run_step(scheduler, device, args.rows, tokens, eager=eager)
                print(name, report[name], flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
