"""Which of the Qwen3-TTS vocoder's convs run on a slow cuDNN engine, on the tree on PYTHONPATH: eager incremental
decodes of the trained speech tokenizer (bf16, SnakeBeta fused, arena state) at the served warm widths and tail
widths, under the profiler with input shapes; per conv call its input and weight shapes, the kernels it launched and
their device time, the slowest first. An eager decode picks the same cuDNN engines as the graph that captures it.

usage: PYTHONPATH=<tree> python3 conv_engine_probe.py [--card 3]
"""

import argparse
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CASES = (("warm", 1, 1), ("warm", 8, 1), ("warm", 8, 8), ("tail", 64, 1), ("tail", 256, 1), ("tail", 581, 8))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    decoder = tokenizer.model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=16, device=device, dtype=torch.bfloat16)
    config = decoder.config
    generator = torch.Generator(device="cpu").manual_seed(0)
    with torch.inference_mode():
        for mode, width, batch in CASES:
            slots = [arena.acquire() for _ in range(batch)]
            codes = torch.randint(config.codebook_size, (batch, config.num_quantizers, width), generator=generator)
            codes = codes.to(device)

            def decode() -> None:
                state = arena.gather(slots)
                if mode == "tail":
                    incremental.decode_tail(codes, torch.full((batch,), width, device=device), state, 1)
                else:
                    incremental.decode(codes, state)
                arena.scatter(slots, state)

            for _ in range(3):
                decode()
            torch.cuda.synchronize(device)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as profiler:
                decode()
                torch.cuda.synchronize(device)
            calls = defaultdict(float)
            for event in profiler.events():
                if event.name in ("aten::cudnn_convolution", "aten::cudnn_convolution_transpose"):
                    kernels = "; ".join(sorted({kernel.name[:70] for kernel in event.kernels}))
                    device_us = sum(kernel.duration for kernel in event.kernels)
                    calls[(event.name, str(event.input_shapes[:2]), kernels)] += device_us
                else:
                    pass
            total = sum(calls.values())
            print(f"== {mode} width {width} batch {batch}: {len(calls)} distinct conv calls, {total:.0f} us of conv kernels")
            for (name, shapes, kernels), device_us in sorted(calls.items(), key=lambda item: -item[1])[:8]:
                print(f"  {device_us:8.1f} us  {name.split('::')[1]:28s} {shapes:70s} {kernels}")
            for slot in slots:
                arena.release(slot)


if __name__ == "__main__":
    main()
