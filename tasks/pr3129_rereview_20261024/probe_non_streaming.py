"""Check the measured reserve against real non-streaming decode at four rows."""

import asyncio
import gc
import json
import logging

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State


def main():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(1337)
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
    reserve = scheduler.measure_peak_step_bytes(
        torch.device("cuda", 0), row_count=4, generated_tokens_per_row=4096
    )
    for case, generated_counts, prompt_count in (
        ("singleton_long_reference", [4096], 750),
        ("four_row_admission_budget", [900, 950, 1000, 1050], 25),
    ):
        items = []
        for generated_count in generated_counts:
            warmup = scheduler.make_warmup_flow_input(
                generated_count, prompt_token_count=prompt_count
            )
            state = FunCosyVoice3State(
                flow_prompt_speech_token=warmup.prompt_token,
                flow_prompt_speech_feat=warmup.prompt_feat,
                flow_embedding=warmup.embedding,
            )
            items.append((state, warmup.token.reshape(-1).long()))
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        resident = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            outputs = asyncio.run(scheduler.vocoder.decode_batch(items))
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() - resident
        assert len(outputs) == len(items)
        assert all(torch.isfinite(waveform).all() for waveform, _ in outputs)
        assert peak <= reserve, (case, peak, reserve)
        print(
            json.dumps(
                dict(
                    case=case,
                    rows=len(items),
                    admitted_frames=sum(
                        2 * (count + prompt_count) for count in generated_counts
                    ),
                    peak_bytes=peak,
                    reserve_bytes=reserve,
                    samples=[waveform.numel() for waveform, _ in outputs],
                    outcome="passed",
                )
            ),
            flush=True,
        )
        del items, outputs


if __name__ == "__main__":
    main()
