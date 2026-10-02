# SPDX-License-Identifier: Apache-2.0
"""Host cost of one vocoder Flow call in the serving configuration (DiT compile on, Flow
graph off), measured without a profiler: per call thread CPU time and wall time over
back to back calls with no sync between them, and the Python GC collections that ran
inside the calls. Reuses the trace pair cases, so the work is the traced work.

    cd <tree> && python host_bench.py --out <json> [--disable-gc]
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_trace_pair import case_body as prefix_case_body  # noqa: E402
from vocoder_trace_pair import case_body as vocoder_case_body  # noqa: E402

from sglang_omni.models.fun_cosyvoice3 import stages  # noqa: E402

PREFIX_CASES = (
    "prefix_first",
    "prefix_hop2",
    "prefix_hop3",
    "prefix_hop2_r1",
    "prefix_hop2_r4",
)
VOCODER_CASES = (
    "hop1",
    "hop16",
    "final1",
    "final16",
    "buffered_small",
    "buffered_large",
    "buffered_miss8",
    "buffered_miss2",
    "hiftstep_hop1",
    "hiftstep_hop16",
    "hiftstep_final1",
    "hiftstep_final16",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--disable-gc", action="store_true")
    parser.add_argument(
        "--cases", nargs="+", default=list(PREFIX_CASES + VOCODER_CASES)
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    scheduler = stages.create_vocoder_executor(
        args.model,
        device="cuda",
        gpu_id=0,
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=False,
        flow_prefix_cache_gb=24.0,
    )
    vocoder = scheduler.vocoder
    collections: list[float] = []
    started_at: list[float] = []

    def on_gc(phase: str, info: dict) -> None:
        if phase == "start":
            started_at.append(time.perf_counter())
        else:
            collections.append((time.perf_counter() - started_at.pop()) * 1e3)

    gc.callbacks.append(on_gc)
    results = {}
    for case in args.cases:
        caches: list = []
        if case in PREFIX_CASES:
            body, caches = prefix_case_body(vocoder, case)
        else:
            body = vocoder_case_body(vocoder, case)
        for _ in range(args.warmup):
            body()
        torch.cuda.synchronize()
        if args.disable_gc:
            gc.disable()
        else:
            pass
        collections.clear()
        cpu, wall = [], []
        for _ in range(args.calls):
            cpu_start, wall_start = time.thread_time(), time.perf_counter()
            body()
            cpu.append((time.thread_time() - cpu_start) * 1e3)
            wall.append((time.perf_counter() - wall_start) * 1e3)
        torch.cuda.synchronize()
        gc.enable()
        results[case] = dict(
            cpu_ms_median=statistics.median(cpu),
            cpu_ms_mean=statistics.fmean(cpu),
            wall_ms_median=statistics.median(wall),
            gc_collections=len(collections),
            gc_ms=sum(collections),
        )
        print(case, results[case], flush=True)
        for cache in caches:
            vocoder.release_prefix_cache(cache)
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
