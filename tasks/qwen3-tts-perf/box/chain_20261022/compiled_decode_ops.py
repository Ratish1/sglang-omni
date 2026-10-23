"""Which SnakeBeta the served compiled decode runs on the tree on PYTHONPATH: compiles the width-8 decode as the warm
runner does (bf16 decoder fused as served, arena state), then profiles one compiled call and counts the kernels by
name: the fused SnakeBeta Triton kernels (main's custom op, the slot conv stack's fused bias kernel) against
Inductor-generated kernels whose names carry sin.

usage: PYTHONPATH=<tree> python3 compiled_decode_ops.py [--card 3]
"""

import argparse
from collections import Counter

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WIDTH = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    decoder = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None).model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=2, device=device, dtype=torch.bfloat16)
    slot = arena.acquire()
    codes = torch.zeros((1, 16, WIDTH), dtype=torch.long, device=device)
    with torch.inference_mode():
        incremental.precompile(codes, arena.gather([slot]))
        for compiled in (False, True):
            state = arena.gather([slot])
            torch.cuda.synchronize(device)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
                incremental.decode(codes, state, compiled=compiled)
                torch.cuda.synchronize(device)
            names = Counter(event.name for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA)
            snake = {name: count for name, count in names.items() if "snake" in name.lower()}
            sine = {name: count for name, count in names.items() if "sin" in name.lower() and "snake" not in name.lower()}
            print(f"{'compiled' if compiled else 'eager'}: {sum(names.values())} kernels; snake kernels {snake}; "
                  f"other kernels with sin {sine}", flush=True)


if __name__ == "__main__":
    main()
