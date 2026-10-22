"""The Qwen3-TTS vocoder's captured decodes as serving builds them, on the tree on PYTHONPATH (one arm per process): the
trained speech tokenizer in bf16 with SnakeBeta fused, the slot arena, warm graphs at widths 1, 4 and 8 (8 compiled,
as served) and a tail graph at width 64, each at the served warm batch buckets. Per graph: the replay's device time
from CUDA events between back-to-back replays (median, p10, p90) and the kernels a replay launches.

usage: PYTHONPATH=<tree> python3 vocoder_decode_bench.py [--card 3] [--replays 100]
"""

import argparse
import statistics

import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    IncrementalCodecGraphKey,
    IncrementalCodecGraphPool,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
BATCHES = (1, 4, 8)
WARM_WIDTHS = (1, 4, 8)
TAIL_WIDTH = 64


def time_graph(runner, codes, slots, valid_frames, replays: int, device) -> tuple[float, float, float, float]:
    assert runner.decode_slots(codes, slots, valid_frames=valid_frames) is not None
    width = next(frames for frames in runner.fresh_frames if frames >= codes.shape[2])
    bucket = next(size for size in runner.batch_sizes if size >= len(slots))
    graph = runner.graphs[IncrementalCodecGraphKey(width, bucket)].graph
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize(device)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(replays + 1)]
    events[0].record()
    for index in range(replays):
        graph.replay()
        events[index + 1].record()
    torch.cuda.synchronize(device)
    times_us = sorted(1000 * events[i].elapsed_time(events[i + 1]) for i in range(replays))
    quantiles = statistics.quantiles(times_us, n=10)
    profiled = 10
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        for _ in range(profiled):
            graph.replay()
        torch.cuda.synchronize(device)
    kernels = sum(1 for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA)
    return statistics.median(times_us), quantiles[0], quantiles[-1], kernels / profiled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    dtype = torch.bfloat16
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    decoder = tokenizer.model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=4 * max(BATCHES), device=device, dtype=dtype)
    config = decoder.config
    warm = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=dtype,
        num_quantizers=config.num_quantizers,
        mode="warm",
        fresh_frames=WARM_WIDTHS,
        batch_sizes=BATCHES,
        min_free_gb=0.0,
        compile_fresh_frames=(8,),
        arena=arena,
        graph_pool=IncrementalCodecGraphPool(stream_priority=0),
    )
    tail = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=dtype,
        num_quantizers=config.num_quantizers,
        mode="tail",
        fresh_frames=(TAIL_WIDTH,),
        batch_sizes=BATCHES,
        min_free_gb=0.0,
        arena=arena,
        graph_pool=IncrementalCodecGraphPool(stream_priority=0),
        emit_frames=1,
    )
    with torch.inference_mode():
        warm.capture()
        tail.capture()
        generator = torch.Generator(device="cpu").manual_seed(0)
        for runner, widths in ((warm, WARM_WIDTHS), (tail, (TAIL_WIDTH,))):
            for width in widths:
                for batch in BATCHES:
                    slots = [arena.acquire() for _ in range(batch)]
                    codes = torch.randint(
                        config.codebook_size, (batch, config.num_quantizers, width), generator=generator
                    ).to(device)
                    valid_frames = [width] * batch if runner is tail else None
                    median, p10, p90, kernels = time_graph(runner, codes, slots, valid_frames, args.replays, device)
                    print(
                        f"{runner.mode} width {width:3d} batch {batch}: replay median {median:8.1f} us "
                        f"p10 {p10:8.1f} p90 {p90:8.1f}, {kernels:4.0f} kernels"
                    )
                    for slot in slots:
                        arena.release(slot)


if __name__ == "__main__":
    main()
