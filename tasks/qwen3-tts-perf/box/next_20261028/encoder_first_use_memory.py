"""Where the device memory the speech tokenizer's first encode leaves behind lives: the caching allocator's allocated
and reserved bytes against the card's used memory, before and after the x-vector silence probe (the engine's first
encode), then after a second probe. Memory the allocator does not hold is the CUDA context's (loaded kernel modules,
library workspaces), which SGLang's KV budget reads as used.

usage: PYTHONPATH=TREE python3 encoder_first_use_memory.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base [--module-loading LAZY]
"""

import argparse
import gc
import os

import torch

import sglang_omni
from sglang_omni.models.qwen3_tts.engine_builder import derive_silence_codec_ids
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer


def report(label: str) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1 << 20)
    allocated = torch.cuda.memory_allocated() / (1 << 20)
    reserved = torch.cuda.memory_reserved() / (1 << 20)
    print(f"{label}: used {used:.0f} MiB, allocated {allocated:.0f}, reserved {reserved:.0f}, outside the allocator {used - reserved:.0f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(f"sglang_omni from {sglang_omni.__file__}, CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING', 'unset')}", flush=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.zeros(1, device=device)
    report("context")
    tokenizer = load_qwen3_tts_tokenizer(args.model, device=str(device), dtype="bfloat16", attn_implementation="sdpa")
    encoder_dtypes = sorted({str(parameter.dtype) for parameter in tokenizer.model.encoder.parameters()})
    report(f"tokenizer (encoder {', '.join(encoder_dtypes)})")
    ids = derive_silence_codec_ids(tokenizer, device)
    report(f"first silence probe ({ids.numel()} ids)")
    derive_silence_codec_ids(tokenizer, device)
    report("second silence probe")


if __name__ == "__main__":
    main()
