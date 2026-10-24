"""#3129's non-streaming reserve on a vocoder without the engine in its process, at max_batch_size 4 where the
non-streaming batch should outgrow the streaming step: the startup log gives both peaks, then the real
non-streaming decode (decode_batch) of the longest request alone and of a 4 request batch at the admission budget
whose rows merge into one Flow and one HiFT group, each against the reserve the vocoder kept."""

import asyncio
import json
import logging
import time

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State


def emit(**fields) -> None:
    print(json.dumps(fields), flush=True)


def request(prompt_tokens: int, code_tokens: int) -> tuple[FunCosyVoice3State, torch.Tensor]:
    state = FunCosyVoice3State(
        flow_prompt_speech_token=torch.zeros(1, prompt_tokens, dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 2 * prompt_tokens, 80),
        flow_embedding=torch.zeros(1, 192),
    )
    return state, torch.zeros(code_tokens, dtype=torch.long)


def decode_peak(vocoder, items) -> dict:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    resident = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    asyncio.run(vocoder.decode_batch(items))
    torch.cuda.synchronize()
    return dict(
        peak_gib=(torch.cuda.max_memory_reserved() - resident) / 2**30,
        free_gib=torch.cuda.mem_get_info()[0] / 2**30,
        seconds=round(time.monotonic() - started, 2),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    started = time.monotonic()
    scheduler = stages.create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        dtype="bfloat16",
        max_batch_size=4,
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
        free_gib=torch.cuda.mem_get_info()[0] / 2**30,
        cache_frames=None if pool is None else pool.free_frames,
    )
    vocoder = scheduler.vocoder
    try:
        emit(case="longest_request_alone", **decode_peak(vocoder, [request(750, 4096)]))
    except torch.OutOfMemoryError as error:
        emit(case="longest_request_alone", outcome="OutOfMemoryError", error=str(error)[:300])
    try:
        batch = [request(0, 1100), request(0, 967), request(0, 967), request(0, 966)]
        emit(case="batch_at_admission_budget", **decode_peak(vocoder, batch))
    except torch.OutOfMemoryError as error:
        emit(case="batch_at_admission_budget", outcome="OutOfMemoryError", error=str(error)[:300])


if __name__ == "__main__":
    main()
