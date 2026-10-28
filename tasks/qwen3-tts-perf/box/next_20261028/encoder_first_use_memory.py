"""Where the device memory the speech tokenizer's first encode leaves behind lives: the caching allocator's allocated
and reserved bytes against the card's used memory, before and after the x-vector silence probe (the engine's first
encode), then after a second probe. Memory the allocator does not hold is the CUDA context's (loaded kernel modules,
library workspaces), which SGLang's KV budget reads as used.

usage: PYTHONPATH=TREE python3 encoder_first_use_memory.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base [--snapshot]

--snapshot records allocation history over the first probe and prints, for every segment still reserved after it, its
size, the bytes still allocated in it and the innermost frames under sglang_omni, qwen_tts or transformers of each block
still allocated, which names the tensors that keep a segment from being released.
"""

import argparse
import gc
import os

import torch

import sglang_omni
from sglang_omni.models.qwen3_tts.engine_builder import derive_silence_codec_ids
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer


def report(label: str) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / (1 << 20)
    allocated = torch.cuda.memory_allocated() / (1 << 20)
    reserved = torch.cuda.memory_reserved() / (1 << 20)
    print(f"{label}: used {used:.0f} MiB, allocated {allocated:.0f}, reserved {reserved:.0f}, outside the allocator {used - reserved:.0f}", flush=True)


def frames_of(block: dict[str, object]) -> str:
    frames = [
        f"{frame['filename'].split('site-packages/')[-1].split('/workspace/')[-1]}:{frame['line']} {frame['name']}"
        for frame in block.get("frames", [])
        if any(part in frame["filename"] for part in ("sglang_omni", "qwen_tts", "transformers"))
    ]
    return " <- ".join(frames[:3]) if frames else "no recorded frames"


def print_segments() -> None:
    for segment in torch.cuda.memory._snapshot()["segments"]:
        active = [block for block in segment["blocks"] if block["state"] == "active_allocated"]
        held = sum(block["size"] for block in active)
        print(f"segment {segment['total_size'] / (1 << 20):.1f} MiB, still allocated {held / (1 << 20):.1f} MiB in {len(active)} blocks", flush=True)
        if segment["total_size"] - held >= 32 << 20:
            for block in active:
                print(f"    {block['size'] / (1 << 20):.2f} MiB: {frames_of(block)}", flush=True)
        else:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--warm-codebooks", action="store_true")
    parser.add_argument("--warm-gemm", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    args = parser.parse_args()
    print(f"sglang_omni from {sglang_omni.__file__}, CUDA_MODULE_LOADING={os.environ.get('CUDA_MODULE_LOADING', 'unset')}", flush=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.zeros(1, device=device)
    report("context")
    tokenizer = load_qwen3_tts_tokenizer(args.model, device=str(device), dtype="bfloat16", attn_implementation="sdpa")
    encoder_dtypes = sorted({str(parameter.dtype) for parameter in tokenizer.model.encoder.parameters()})
    report(f"tokenizer (encoder {', '.join(encoder_dtypes)})")
    if args.warm_codebooks:
        tables = [module.embed for module in tokenizer.model.encoder.modules() if type(module).__name__ == "MimiEuclideanCodebook"]
        report(f"codebook tables built at load ({len(tables)})")
    else:
        pass
    if args.warm_gemm:
        weight = next(tokenizer.model.encoder.parameters())
        torch.ones(8, 8, device=device, dtype=weight.dtype) @ torch.ones(8, 8, device=device, dtype=weight.dtype)
        report("one small gemm")
    else:
        pass
    if args.snapshot:
        torch.cuda.memory._record_memory_history(max_entries=200000)
    else:
        pass
    ids = derive_silence_codec_ids(tokenizer, device)
    report(f"first silence probe ({ids.numel()} ids)")
    if args.snapshot:
        print_segments()
        torch.cuda.memory._record_memory_history(enabled=None)
    else:
        pass
    derive_silence_codec_ids(tokenizer, device)
    report("second silence probe")
    if args.graphs:
        batcher = Qwen3TTSRefCodeBatcher(tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
        report("reference graphs")
        torch._C._cuda_clearCublasWorkspaces()
        report("reference graphs, cuBLAS workspaces cleared")
        batcher.close()
    else:
        pass


if __name__ == "__main__":
    main()
