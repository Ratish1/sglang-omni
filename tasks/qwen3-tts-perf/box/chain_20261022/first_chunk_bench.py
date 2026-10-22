"""The first chunk of a stream behind an 88 s reference (1,100 reference frames and 1 generated) as the Qwen3-TTS
vocoder decodes it, on the tree on PYTHONPATH: today's path, the whole width through window graphs of up to 64 frames
replayed one after another on the stream's slot, against the capped path, the last first_chunk_reference_frames(1) + 1
frames in one tail replay from a slot started there (skipped on a tree without the cap). Per batch (1 and 8 rows):
the device time of the chunk from CUDA events and its host time (medians of 5), and the tail graphs' capture memory.

usage: PYTHONPATH=<tree> python3 first_chunk_bench.py [--card 3]
"""

import argparse
import statistics
import time

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    IncrementalCodecGraphPool,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REFERENCE_FRAMES = 1100
BATCHES = (1, 8)
WINDOW_WIDTHS = (1, 2, 4, 8, 16, 32, 64)


def timed(run, device) -> tuple[float, float]:
    device_ms, host_ms = [], []
    for _ in range(6):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        host_start = time.perf_counter()
        start.record()
        run()
        end.record()
        host_ms.append(1000 * (time.perf_counter() - host_start))
        torch.cuda.synchronize(device)
        device_ms.append(start.elapsed_time(end))
    return statistics.median(device_ms[1:]), statistics.median(host_ms[1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
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
    pool = IncrementalCodecGraphPool(stream_priority=0)
    window = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=dtype,
        num_quantizers=config.num_quantizers,
        mode="window",
        fresh_frames=WINDOW_WIDTHS,
        batch_sizes=BATCHES,
        min_free_gb=0.0,
        arena=arena,
        graph_pool=pool,
    )
    capped = "first_chunk_reference_frames" in dir(incremental)
    tail_width = incremental.first_chunk_reference_frames(1) + 1 if capped else 0
    with torch.inference_mode():
        window.capture()
        tail = None
        if capped:
            tail = Qwen3TTSIncrementalCodecCudaGraphRunner(
                incremental,
                device=device,
                dtype=dtype,
                num_quantizers=config.num_quantizers,
                mode="tail",
                fresh_frames=(tail_width,),
                batch_sizes=BATCHES,
                min_free_gb=0.0,
                arena=arena,
                graph_pool=pool,
                emit_frames=1,
            )
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_reserved(device)
            tail.capture()
            torch.cuda.synchronize(device)
            print(
                f"tail width {tail_width} captured at buckets {BATCHES}: "
                f"reserved +{(torch.cuda.memory_reserved(device) - before) / 2**20:.0f} MiB, "
                f"graph pool {tail.stats()['memory']}"
            )
        else:
            pass
        generator = torch.Generator(device="cpu").manual_seed(0)
        total = REFERENCE_FRAMES + 1
        split = window.split_frames(total)
        for batch in BATCHES:
            codes = torch.randint(
                config.codebook_size, (batch, config.num_quantizers, total), generator=generator
            ).to(device)
            slots = [arena.acquire() for _ in range(batch)]

            def windows() -> None:
                for slot in slots:
                    arena.zero_slot(slot)
                offset = 0
                for width in split:
                    assert window.decode_slots(codes[..., offset : offset + width], slots) is not None
                    offset += width

            device_ms, host_ms = timed(windows, device)
            print(f"batch {batch} windows {split}: device {device_ms:.2f} ms, host {host_ms:.2f} ms")
            if tail is not None:
                start = total - tail_width

                def capped_tail() -> None:
                    for slot in slots:
                        arena.zero_slot(slot)
                        arena.start_slot(slot, start)
                    assert (
                        tail.decode_slots(codes[..., start:], slots, valid_frames=[tail_width] * batch) is not None
                    )

                device_ms, host_ms = timed(capped_tail, device)
                print(f"batch {batch} capped tail {tail_width}: device {device_ms:.2f} ms, host {host_ms:.2f} ms")
            else:
                pass
            for slot in slots:
                arena.release(slot)


if __name__ == "__main__":
    main()
