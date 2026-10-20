"""Seconds the Qwen3-TTS vocoder's startup spends per served shape, on the tree on PYTHONPATH: the first eager decode
of the shape, then Qwen3TTSIncrementalDecoder.precompile on an arena-gathered state, as the codec graph runner calls
it in warmup_capture_shape.

usage: TORCHINDUCTOR_CACHE_DIR=<dir> PYTHONPATH=<tree> python3 vocoder_precompile_bench.py [--widths 1 2 8]
       [--buckets 1 2 4 8]

The defaults are the shapes a server compiles: the cold widths 1 and 2 and the follow-up stride 8, at each batch
bucket. Run it twice on one cache dir: the first process pays a cold compile, the second the warm one a second boot
pays. Dynamo's own per-phase totals follow the per-shape seconds.
"""

import argparse
import time

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--buckets", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    device = torch.device("cuda")
    decoder = load_qwen3_tts_tokenizer(MODEL, device="cuda", dtype="bfloat16", attn_implementation=None).model.decoder
    num_quantizers = int(decoder.config.num_quantizers)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=max(args.buckets), device=device, dtype=torch.bfloat16)
    eager_total = 0.0
    compile_total = 0.0
    for width in args.widths:
        for bucket in args.buckets:
            codes = torch.randint(0, 2048, (bucket, num_quantizers, width), device=device)
            with torch.inference_mode():
                torch.cuda.synchronize()
                start = time.perf_counter()
                incremental.decode(codes, arena.gather_by_index(torch.arange(bucket, device=device)))
                torch.cuda.synchronize()
                eager = time.perf_counter() - start
                state = arena.gather_by_index(torch.arange(bucket, device=device))
                start = time.perf_counter()
                incremental.precompile(codes, state)
                torch.cuda.synchronize()
                compiled = time.perf_counter() - start
            eager_total += eager
            compile_total += compiled
            print(f"width {width} bucket {bucket}: first eager {eager:.2f} s precompile {compiled:.2f} s", flush=True)
    shapes = len(args.widths) * len(args.buckets)
    print(f"total over {shapes} shapes: first eager {eager_total:.1f} s precompile {compile_total:.1f} s", flush=True)
    print(torch._dynamo.utils.compile_times(repr="str", aggregate=True), flush=True)  # noqa: leading-underscore


if __name__ == "__main__":
    main()
