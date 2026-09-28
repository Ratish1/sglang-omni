# SPDX-License-Identifier: Apache-2.0
"""Startup cost of the Fun-CosyVoice3 vocoder, phase by phase.

Builds the vocoder exactly as the serving factory does (create_vocoder_executor
with the shipped config) in this process, times each startup phase, and prints
Dynamo's compile time table and the Inductor and AOTAutograd cache counters.
Run it twice against one TORCHINDUCTOR_CACHE_DIR to see what a warm cache
removes.

    OMP_NUM_THREADS=1 python c1_compile_startup.py --out /data/c1/cold.json
    python c1_compile_startup.py --no-compile --out /data/c1/eager.json
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import time

import torch
import torch._dynamo.utils as dynamo_utils

from sglang_omni.models.fun_cosyvoice3 import stages, streaming_vocoder
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
)

PHASES: dict[str, float] = {}


def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            PHASES[name] = PHASES.get(name, 0.0) + time.perf_counter() - start

    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    stages.load_cosyvoice3_flow_hift = timed("load", stages.load_cosyvoice3_flow_hift)
    stages.compile_dit_backbone = timed("native_compile", stages.compile_dit_backbone)
    stages.FlowCudaGraphRunner.capture = timed(
        "graph_capture", stages.FlowCudaGraphRunner.capture
    )
    scheduler_cls = streaming_vocoder.FunCosyVoice3StreamingVocoderScheduler
    scheduler_cls.warmup_packed_dit_compile = timed(
        "packed_compile", scheduler_cls.warmup_packed_dit_compile
    )
    scheduler_cls.warmup_now = timed("warmup", scheduler_cls.warmup_now)

    start = time.perf_counter()
    stages.create_vocoder_executor(
        args.model,
        device="cuda",
        gpu_id=0,
        enable_dit_torch_compile=not args.no_compile,
        flow_cuda_graph_capture_shapes=FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
    )
    total = time.perf_counter() - start

    counters = {
        group: dict(values)
        for group, values in dynamo_utils.counters.items()
        if group in ("inductor", "aot_autograd", "stats", "graph_break", "recompiles")
    }
    report = {
        "compile": not args.no_compile,
        "total_s": round(total, 2),
        "phases_s": {name: round(value, 2) for name, value in PHASES.items()},
        "torch_num_threads": torch.get_num_threads(),
        "inductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "counters": counters,
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
    }
    print(json.dumps(report, indent=1))
    print(dynamo_utils.compile_times(repr="str", aggregate=True))
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
