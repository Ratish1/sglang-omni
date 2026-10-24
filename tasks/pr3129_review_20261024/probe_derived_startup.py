"""Exercise the production compiled/graph vocoder on its own GPU (no AR process)."""

import json
import logging
import time

import torch

from sglang_omni.models.fun_cosyvoice3 import stages


def main():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(1337)
    start = time.monotonic()
    scheduler = stages.create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        dtype="bfloat16",
        max_batch_size=16,
        flow_prefix_cache_gb=None,
        enable_dit_torch_compile=True,
        enable_flow_cuda_graph=True,
        enable_flow_prefix_cuda_graph=True,
        enable_flow_whole_history_cuda_graph=True,
        enable_hift_cuda_graph=True,
    )
    torch.cuda.synchronize()
    print(json.dumps(dict(case="derived_startup", seconds=time.monotonic()-start,
                         reserved_bytes=torch.cuda.memory_reserved(),
                         free_bytes=torch.cuda.mem_get_info()[0],
                         cache_frames=scheduler.vocoder.flow.prefix_pool.free_frames)), flush=True)
    try:
        reserve = scheduler.measure_peak_step_bytes(torch.device("cuda:0"), row_count=16, tokens_per_row=4096)
    except torch.OutOfMemoryError as error:
        print(json.dumps(dict(case="post_pool_worst_step", outcome=type(error).__name__, error=str(error))), flush=True)
    else:
        print(json.dumps(dict(case="post_pool_worst_step", outcome="passed", peak_bytes=reserve,
                             free_bytes=torch.cuda.mem_get_info()[0])), flush=True)


if __name__ == "__main__":
    main()
