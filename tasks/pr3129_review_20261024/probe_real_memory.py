"""Compare the new streaming reserve with the real full-history HiFT path."""

import gc
import json
import logging
import time

import torch

from sglang_omni.models.fun_cosyvoice3 import stages


def main():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(1337)
    scheduler = stages.create_vocoder_executor(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda:0",
        dtype="bfloat16",
        max_batch_size=16,
        flow_prefix_cache_gb=0,
        enable_dit_torch_compile=False,
        enable_flow_cuda_graph=False,
        enable_flow_prefix_cuda_graph=False,
        enable_flow_whole_history_cuda_graph=False,
        enable_hift_cuda_graph=False,
    )
    reserve = scheduler.measure_peak_step_bytes(
        torch.device("cuda:0"), row_count=16, tokens_per_row=4096
    )
    print(json.dumps(dict(case="streaming_reserve", bytes=reserve)), flush=True)
    for row_count, frames in ((1, 8192), (2, 4000), (16, 500)):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        resident = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with torch.inference_mode(), scheduler.vocoder.stream_context:
            mels = [torch.randn(1, 80, frames, device="cuda") for _ in range(row_count)]
            audio = scheduler.vocoder.mel2wav_batch(mels)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() - resident
        print(
            json.dumps(
                dict(
                    case="nonstreaming_hift",
                    rows=row_count,
                    frames=frames,
                    peak_bytes=peak,
                    reserve_bytes=reserve,
                    audio_samples=[wave.shape[-1] for wave in audio],
                    elapsed_seconds=time.monotonic() - start,
                )
            ),
            flush=True,
        )
        del mels, audio


if __name__ == "__main__":
    main()
