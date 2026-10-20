"""One Qwen3-TTS vocoder decode per served compiled shape, eager and compiled, on the tree on PYTHONPATH, saved for a
comparison across trees (codec_paths_compare.py).

usage: PYTHONPATH=<tree> python3 codec_paths_dump.py OUT.pt [--widths 1 2 8] [--buckets 1 4]

The tokenizer loads through the tree's load_qwen3_tts_tokenizer, so a tree that rewrites modules at load (V5's
codebook tables) decodes as it serves. Per shape: seeded codes, a fresh arena-gathered state for each path, the eager
decode, then precompile and the compiled decode, each waveform saved as float32.
"""

import argparse

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--buckets", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()
    device = torch.device("cuda")
    decoder = load_qwen3_tts_tokenizer(MODEL, device="cuda", dtype="bfloat16", attn_implementation=None).model.decoder
    num_quantizers = int(decoder.config.num_quantizers)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=max(args.buckets), device=device, dtype=torch.bfloat16)
    generator = torch.Generator(device=device).manual_seed(0)
    waveforms: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for width in args.widths:
            for bucket in args.buckets:
                codes = torch.randint(0, 2048, (bucket, num_quantizers, width), device=device, generator=generator)
                rows = torch.arange(bucket, device=device)
                eager = incremental.decode(codes, arena.gather_by_index(rows))
                incremental.precompile(codes, arena.gather_by_index(rows))
                compiled = incremental.decode(codes, arena.gather_by_index(rows), compiled=True)
                waveforms[f"eager {width} {bucket}"] = eager.float().cpu()
                waveforms[f"compiled {width} {bucket}"] = compiled.float().cpu()
    torch.save(waveforms, args.output)
    print(f"saved {len(waveforms)} waveforms to {args.output}", flush=True)


if __name__ == "__main__":
    main()
