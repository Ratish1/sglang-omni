"""Separate live CUDA graph memory from temporary convolution workspace caching."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3.hift_cuda_graph import HiftDecodeCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture
from sglang_omni.models.fun_cosyvoice3.stages import create_vocoder_executor
from sglang_omni.platforms import current_platform


def memory_snapshot() -> dict[str, int]:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "free": torch.cuda.mem_get_info()[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parsed = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(3029)
    torch.backends.cudnn.allow_tf32 = False
    scheduler = create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        dtype="bfloat16",
        max_batch_size=16,
        flow_prefix_cache_gb=0.0,
        enable_dit_torch_compile=False,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=False,
        enable_flow_whole_history_cuda_graph=False,
        enable_hift_cuda_graph=False,
    )
    device = torch.device("cuda", 0)
    backend = current_platform.get_device_graph_backend(device)
    assert backend is not None
    before = memory_snapshot()
    runner = HiftDecodeCudaGraphRunner(
        scheduler.vocoder.hift,
        graphs=SolveGraphCapture(backend, device=device, autocast_dtype=torch.bfloat16),
        mel_channels=80,
        samples_per_frame=480,
        row_tiers=[1, 4, 16],
        frame_tiers=[64, 100, 128],
    )
    with torch.inference_mode(), scheduler.vocoder.stream_context:
        runner.capture()
    live = memory_snapshot()
    del runner
    released = memory_snapshot()
    scheduler.stop()
    summary = {
        "label": parsed.label,
        "before": before,
        "graphs_live": live,
        "graphs_released": released,
    }
    Path(f"/data/review3029/{parsed.label}_memory.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
