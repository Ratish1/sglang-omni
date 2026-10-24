"""#3129 review F1 on the fixed head: a vocoder without the engine in its process (runtime headroom from its own
pre-load memory), then on the derived pool the largest step again, a 16 row cross-lingual final (4050 generated
tokens after a 750 token reference), 48 prefix-cached streams kept live, and the largest step beside them.
Adapted from the reviewer's probe_derived_startup.py (tasks/pr3129_review_20261024)."""

import json
import logging
import time

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import HiftStepRow


def free_gib() -> float:
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0] / 2**30


def emit(**fields) -> None:
    print(json.dumps(fields), flush=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(1337)
    started = time.monotonic()
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
    pool = scheduler.vocoder.flow.prefix_pool
    emit(
        case="startup",
        seconds=round(time.monotonic() - started, 1),
        reserved_gib=torch.cuda.memory_reserved() / 2**30,
        free_gib=free_gib(),
        cache_frames=None if pool is None else pool.free_frames,
    )
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.reset_peak_memory_stats()
    try:
        step_bytes = scheduler.measure_peak_step_bytes(
            device, row_count=16, generated_tokens_per_row=4096
        )
    except torch.OutOfMemoryError as error:
        emit(case="post_pool_largest_step", outcome="OutOfMemoryError", error=str(error)[:300])
    else:
        emit(case="post_pool_largest_step", outcome="passed", step_gib=step_bytes / 2**30, free_gib=free_gib())
    long_items = [
        stages.FlowBatchInput(
            token=torch.zeros(1, 4050, dtype=torch.int32),
            prompt_token=torch.zeros(1, 750, dtype=torch.int32),
            prompt_feat=torch.zeros(1, 1500, 80),
            embedding=torch.zeros(1, 192),
        )
        for _ in range(16)
    ]
    try:
        with torch.inference_mode(), scheduler.vocoder.stream_context:
            mels = scheduler.vocoder.leftover_batch(long_items)
            scheduler.vocoder.hift_step(
                [
                    HiftStepRow(
                        history=mel,
                        emitted_samples=max(mel.shape[2] - 200, 0) * 480,
                        is_final=True,
                    )
                    for mel in mels
                ]
            ).wait()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as error:
        emit(case="crosslingual_final_16x4800", outcome="OutOfMemoryError", error=str(error)[:300])
    else:
        emit(case="crosslingual_final_16x4800", outcome="passed", free_gib=free_gib())
        del mels
    caches = []
    with torch.inference_mode(), scheduler.vocoder.stream_context:
        for _ in range(3):
            items = [scheduler.make_warmup_flow_input(28) for _ in range(16)]
            pairs = [scheduler.vocoder.prefix_cache_rows(100) for _ in items]
            assert all(pair is not None for pair in pairs)
            scheduler.vocoder.hop_batch_prefix(items, pairs)
            caches.extend(pairs)
    torch.cuda.synchronize()
    context_bytes = sum(
        row.conv_context.numel() * row.conv_context.element_size() for pair in caches for row in pair
    )
    emit(case="retained_streams", streams=len(caches), context_mib=context_bytes / 2**20, free_gib=free_gib())
    try:
        step_bytes = scheduler.measure_peak_step_bytes(
            device, row_count=16, generated_tokens_per_row=4096
        )
    except torch.OutOfMemoryError as error:
        emit(case="largest_step_beside_live_streams", outcome="OutOfMemoryError", error=str(error)[:300])
    else:
        emit(
            case="largest_step_beside_live_streams",
            outcome="passed",
            step_gib=step_bytes / 2**30,
            free_gib=free_gib(),
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            total_gib=torch.cuda.get_device_properties(device).total_memory / 2**30,
        )


if __name__ == "__main__":
    main()
