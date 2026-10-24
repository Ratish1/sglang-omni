"""Exercise the production compiled/graph vocoder on its own GPU (no AR process)."""

import json
import logging
import time

import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.utils import CosyVoice3Tokenizer
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import HiftStepRow


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
    tokenizer = CosyVoice3Tokenizer(stages.resolve_checkpoint("FunAudioLLM/Fun-CosyVoice3-0.5B-2512") + "/CosyVoice-BlankEN")
    ar_prompt = len(tokenizer.encode("You are a helpful assistant.<|endofprompt|>")) + len(tokenizer.encode("Hi")) + 2
    print(json.dumps(dict(case="crosslingual_lengths", ar_prompt_tokens=ar_prompt,
                         generated_tokens=4050, ar_total_tokens=ar_prompt+4050,
                         flow_reference_tokens=750, flow_total_tokens=4800)), flush=True)
    long_items = [stages.FlowBatchInput(
        token=torch.zeros(1, 4050, dtype=torch.int32),
        prompt_token=torch.zeros(1, 750, dtype=torch.int32),
        prompt_feat=torch.zeros(1, 1500, 80),
        embedding=torch.zeros(1, 192),
    ) for _ in range(16)]
    try:
        with torch.inference_mode(), scheduler.vocoder.stream_context:
            for decode, final in ((scheduler.vocoder.hop_batch, False), (scheduler.vocoder.leftover_batch, True)):
                mels = decode(long_items)
                scheduler.vocoder.hift_step([HiftStepRow(
                    history=mel, emitted_samples=max(mel.shape[2]-200, 0)*480,
                    is_final=final) for mel in mels]).wait()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as error:
        print(json.dumps(dict(case="admitted_long_reference", outcome=type(error).__name__, error=str(error))), flush=True)
    else:
        print(json.dumps(dict(case="admitted_long_reference", outcome="passed")), flush=True)
        del mels
    caches = []
    with torch.inference_mode(), scheduler.vocoder.stream_context:
        for group in range(3):
            items = [scheduler.make_warmup_flow_input(28) for _ in range(16)]
            pairs = [scheduler.vocoder.prefix_cache_rows(100) for _ in items]
            assert all(pair is not None for pair in pairs)
            scheduler.vocoder.hop_batch_prefix(items, pairs)
            caches.extend(pairs)
    torch.cuda.synchronize()
    context_bytes = sum(row.conv_context.numel()*row.conv_context.element_size()
                        for pair in caches for row in pair)
    print(json.dumps(dict(case="retained_stream_state", streams=len(caches), context_bytes=context_bytes)), flush=True)
    try:
        reserve = scheduler.measure_peak_step_bytes(torch.device("cuda:0"), row_count=16, tokens_per_row=4096)
    except torch.OutOfMemoryError as error:
        print(json.dumps(dict(case="worst_step_with_live_streams", outcome=type(error).__name__, error=str(error))), flush=True)
    else:
        print(json.dumps(dict(case="worst_step_with_live_streams", outcome="passed", peak_bytes=reserve)), flush=True)


if __name__ == "__main__":
    main()
