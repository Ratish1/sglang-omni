"""The first chunk of a stream behind a long reference (default 1,100 reference frames, about 88 s, and 1 generated)
as the Qwen3-TTS vocoder decodes it, on the tree on PYTHONPATH: today's path, the whole width through window graphs of
up to 64 frames replayed one after another on the stream's slot, against the path the scheduler plans with the cap,
one replay of the first_chunk_reference_frames(1) + 1 wide tail graph from the reach when the reference is longer and
from frame 0 otherwise (skipped on a tree without the cap). Per reference length and batch (1 and 8 rows): the device
time of the chunk from CUDA events and its host time (medians of 5), the tail graph's capture memory, and the two
paths' emitted samples and kept state against each other.

usage: PYTHONPATH=<tree> python3 first_chunk_bench.py [--card 3] [--reference-frames 300 450 579 1100]
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
SAMPLES_PER_FRAME = 1920


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
    parser.add_argument("--reference-frames", type=int, nargs="+", default=[REFERENCE_FRAMES])
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
        for reference_frames, batch in [(frames, batch) for frames in args.reference_frames for batch in BATCHES]:
            total = reference_frames + 1
            split = window.split_frames(total)
            print(f"== reference {reference_frames} frames, batch {batch}", flush=True)
            codes = torch.randint(
                config.codebook_size, (batch, config.num_quantizers, total), generator=generator
            ).to(device)
            slots = [arena.acquire() for _ in range(batch)]

            def windows() -> torch.Tensor:
                for slot in slots:
                    arena.zero_slot(slot)
                offset = 0
                for width in split:
                    waveform = window.decode_slots(codes[..., offset : offset + width], slots)
                    assert waveform is not None
                    offset += width
                return waveform[..., -SAMPLES_PER_FRAME:]

            device_ms, host_ms = timed(windows, device)
            print(f"batch {batch} windows {split}: device {device_ms:.2f} ms, host {host_ms:.2f} ms")
            if tail is not None:
                # as the scheduler plans it: from the reach when the reference is longer, else from frame 0,
                # padded into the reach-wide tail graph
                start = max(0, total - tail_width)
                valid_width = total - start

                def capped_tail() -> torch.Tensor:
                    for slot in slots:
                        arena.zero_slot(slot)
                        arena.start_slot(slot, start)
                    waveform = tail.decode_slots(codes[..., start:], slots, valid_frames=[valid_width] * batch)
                    assert waveform is not None
                    return waveform[..., -SAMPLES_PER_FRAME:]

                device_ms, host_ms = timed(capped_tail, device)
                print(
                    f"batch {batch} tail from frame {start}, {valid_width} frames in the {tail_width}-wide graph: "
                    f"device {device_ms:.2f} ms, host {host_ms:.2f} ms"
                )
                index = torch.tensor(slots, device=device)
                outputs = {}
                for name, run in (("whole", windows), ("capped", capped_tail)):
                    emitted = run().clone()
                    gathered = arena.gather(slots)
                    # main's gather returns the cohort's rows, the slot conv stack's the whole pools
                    conv_rows = [
                        pool.clone() if pool.shape[0] == batch else pool.index_select(0, index)
                        for pool in (*gathered.conv_histories.values(), *gathered.transconv_overlaps.values())
                    ]
                    outputs[name] = [
                        emitted,
                        *(value.clone() for value in gathered.transformer_keys.values()),
                        *(value.clone() for value in gathered.transformer_values.values()),
                        *conv_rows,
                    ]
                errors = [
                    ((capped.float() - whole.float()).norm() / whole.float().norm().clamp_min(1e-30)).item()
                    for whole, capped in zip(outputs["whole"], outputs["capped"])
                ]
                print(
                    f"batch {batch} capped against whole: emitted samples relative error {errors[0]:.3e}; "
                    f"retained state max {max(errors[1:]):.3e}, median {statistics.median(errors[1:]):.3e} "
                    f"over {len(errors) - 1} tensors"
                )
            else:
                pass
            for slot in slots:
                arena.release(slot)


if __name__ == "__main__":
    main()
