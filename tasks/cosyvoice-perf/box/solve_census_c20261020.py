"""The Flow's Euler solve, kernel by kernel (10-20, L7 of DESIGN_PR_STACK_20261020.md 5): a replay of either graph kind
costs about 9 ms before any frame, so the census counts what one solve launches. The vocoder is built as serving
builds it (config.py's vocoder arguments) with the prefix and whole history graphs off and an 8 GB prefix pool (8 rows
of the longest hop on both CFG twins take 5,120 frames), so the
steps run the compiled forward the graphs capture, kernel by kernel under torch.profiler: an 8 row follow-up hop of
the longest hop over filled prefixes (c64's typical prefix replay) and a 4 row whole history final. Per step: kernels,
their device time, kernels under 5 us and their time, and the top kernels by count and by time.
usage: CUDA_VISIBLE_DEVICES=<card> PYTHONPATH=<tree>:<cosyvoice> python3 solve_census_c20261020.py <out txt>"""

import collections
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import FunCosyVoice3PipelineConfig
from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput
from sglang_omni.models.fun_cosyvoice3.streaming import PRE_LOOKAHEAD_LEN
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import causal_hop_frames

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"


def census(name: str, step, out) -> None:
    for _ in range(3):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        step()
        torch.cuda.synchronize()
    kernels = [event for event in prof.events() if event.device_type.name == "CUDA"]
    by_name = collections.defaultdict(lambda: [0, 0.0])
    for event in kernels:
        by_name[event.name][0] += 1
        by_name[event.name][1] += event.device_time
    total_us = sum(event.device_time for event in kernels)
    short = [event for event in kernels if event.device_time < 5]
    print(
        f"{name}: kernels {len(kernels)}, device {total_us / 1000:.2f} ms; under 5 us: {len(short)} kernels, "
        f"{sum(event.device_time for event in short) / 1000:.2f} ms",
        file=out,
    )
    for label, key in (("by count", lambda kv: -kv[1][0]), ("by time", lambda kv: -kv[1][1])):
        print(f"  top {label}:", file=out)
        for kernel, (count, micros) in sorted(by_name.items(), key=key)[:15]:
            print(f"    {count:5d}  {micros / 1000:7.2f} ms  {kernel[:110]}", file=out)
    out.flush()


def main() -> None:
    factory = FunCosyVoice3PipelineConfig(model_path=MODEL).stage_named("vocoder").factory
    arguments = {**factory.model_dump(exclude_none=True), **(factory.model_extra or {})}
    arguments.update(
        enable_flow_prefix_cuda_graph=False,
        enable_flow_whole_history_cuda_graph=False,
        flow_prefix_cache_gb=8.0,
        flow_prefix_cache_reserve_gb=0.0,
    )
    scheduler = stages.create_vocoder_executor(MODEL, device="cuda", gpu_id=0, **arguments)
    vocoder = scheduler.vocoder
    ratio = vocoder.flow.token_mel_ratio
    first = scheduler.make_warmup_flow_input(scheduler.token_hop_len + PRE_LOOKAHEAD_LEN)
    caches = []
    for _ in range(8):
        cache = vocoder.prefix_cache_rows(causal_hop_frames(first, token_mel_ratio=ratio))
        with vocoder.stream_context:
            vocoder.hop_batch_prefix([first], [cache])
        caches.append(cache)
    longer = FlowBatchInput(
        token=torch.zeros(1, first.token.shape[1] + scheduler.token_max_hop_len, dtype=torch.int32),
        prompt_token=first.prompt_token,
        prompt_feat=first.prompt_feat,
        embedding=first.embedding,
    )
    for cache in caches:
        assert vocoder.grow_prefix_cache(cache, causal_hop_frames(longer, token_mel_ratio=ratio))

    def hop_step() -> None:
        with vocoder.stream_context:
            for cache in caches:
                for row in cache:
                    row.committed_frames = causal_hop_frames(first, token_mel_ratio=ratio)
            vocoder.hop_batch_prefix([longer] * 8, caches)

    finals = [scheduler.make_warmup_flow_input(400) for _ in range(4)]

    def final_step() -> None:
        with vocoder.stream_context:
            vocoder.leftover_batch(finals)

    with open(sys.argv[1], "w") as out:
        census("prefix hop, 8 rows of the longest hop", hop_step, out)
        census("whole history final, 4 rows of 400 tokens", final_step, out)
    print(open(sys.argv[1]).read())


if __name__ == "__main__":
    main()
