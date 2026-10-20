# SPDX-License-Identifier: Apache-2.0
"""E3: can the prefix graphs' memory be measured before the prefix pool exists, as vLLM measures its
graphs on a throwaway pool before its KV cache (gpu_model_runner.py profile_cudagraph_memory)?

The vocoder is built as serving builds it at ROWS rows without the pool. Then:
1. dry: a small temporary pool, the prefix forward compiled and warmed through it, the prefix graphs
   captured into a fresh graph pool of their own; device memory taken by the capture (free memory read
   with an emptied cache before and after, SGLang's get_available_gpu_memory, as the capture log does).
2. the dry graphs, their pool and the temporary pool released; free memory read again.
3. real: a 20 GB pool, the same compiled forward warmed through it, the prefix graphs captured into the
   vocoder's shared solve graph pool, as the factory does; memory taken by the capture.
Dynamo's counters after each phase tell whether the real pool recompiles the prefix forward.

    PYTHONPATH=/workspace/t_p123v3 python e3_prefix_dry_capture_c20261011.py --rows 16 --out e3.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time

import torch
from sglang.srt.utils.common import get_available_gpu_memory
from torch._dynamo.utils import counters

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import PrefixCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.streaming import PRE_LOOKAHEAD_LEN
from sglang_omni.platforms import current_platform

DRY_POOL_GB = 0.25
REAL_POOL_GB = 20.0


def dynamo_state() -> dict[str, int]:
    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "recompiles": sum(int(v) for v in counters["recompiles"].values()),
    }


def free_gib(device: torch.device) -> float:
    return get_available_gpu_memory(device.type, device.index)


def prefix_runner(scheduler, pool, graphs, rows: int) -> PrefixCudaGraphRunner:
    """The runner create_vocoder_executor builds, its dtypes from the whole history runner."""
    flow = scheduler.vocoder.flow
    whole_history = flow.whole_history_cuda_graph_runner
    return PrefixCudaGraphRunner(
        flow.packed_estimator,
        pool,
        graphs=graphs,
        frame_dtype=whole_history.frame_dtype,
        speaker_dtype=whole_history.speaker_dtype,
        cfg_rate=flow.decoder.inference_cfg_rate,
        mel_channels=flow.output_size,
        speaker_channels=flow.spk_embed_affine_layer.out_features,
        max_rows=rows,
        hop_frames=scheduler.token_max_hop_len * flow.token_mel_ratio,
        min_hop_frames=scheduler.token_hop_len * flow.token_mel_ratio,
        max_frames=flow.decoder.rand_noise.shape[2],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda", 0)
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
    vocoder = scheduler.vocoder
    flow = vocoder.flow
    dtype = vocoder.autocast_dtype
    report: dict[str, object] = {"rows": args.rows, "start": dynamo_state()}

    started = time.monotonic()
    flow.prefix_pool = stages.build_prefix_pool(
        flow, device, dtype, size_gb=DRY_POOL_GB, reserve_bytes=0
    )
    scheduler.warmup_prefix_compile()
    compiled_forward = flow.prefix_pool.forward
    report["dry_warmup"] = {"s": round(time.monotonic() - started, 2), **dynamo_state()}
    dry_graphs = SolveGraphCapture(
        current_platform.get_device_graph_backend(device),
        device=device,
        autocast_dtype=dtype,
    )
    before = free_gib(device)
    started = time.monotonic()
    dry_runner = prefix_runner(scheduler, flow.prefix_pool, dry_graphs, args.rows)
    dry_runner.capture()
    report["dry_capture"] = {
        "gib": round(before - free_gib(device), 3),
        "s": round(time.monotonic() - started, 2),
        **dynamo_state(),
    }

    del dry_runner, dry_graphs
    flow.prefix_pool = None
    gc.collect()
    report["after_release"] = {"free_gib": round(free_gib(device), 3), "before_dry": round(before, 3)}

    flow.prefix_pool = stages.build_prefix_pool(
        flow, device, dtype, size_gb=REAL_POOL_GB, reserve_bytes=0
    )
    flow.prefix_pool.forward = compiled_forward
    started = time.monotonic()
    with vocoder.stream_context:
        scheduler.warmup_prefix_hops(
            scheduler.make_warmup_flow_input(scheduler.token_hop_len + PRE_LOOKAHEAD_LEN)
        )
    report["real_warmup"] = {"s": round(time.monotonic() - started, 2), **dynamo_state()}
    before = free_gib(device)
    started = time.monotonic()
    real_runner = prefix_runner(
        scheduler, flow.prefix_pool, flow.whole_history_cuda_graph_runner.graphs, args.rows
    )
    real_runner.capture()
    report["real_capture"] = {
        "gib": round(before - free_gib(device), 3),
        "s": round(time.monotonic() - started, 2),
        **dynamo_state(),
    }
    print(json.dumps(report, indent=1), flush=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
