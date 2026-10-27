"""Device memory the Qwen3-TTS engine holds before SGLang sizes its KV pool, step by step as the server starts: the
speech tokenizer, the x-vector silence probe, the reference encoder graphs. Each figure is resident memory after
gc.collect and empty_cache, as SGLang's KV budget reads free memory. Run once per tree.

usage: PYTHONPATH=TREE python3 startup_memory.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
"""

import argparse
import gc

import torch

import sglang_omni
from sglang_omni.models.qwen3_tts.engine_builder import derive_silence_codec_ids
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer


def resident_mib() -> float:
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1 << 20)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    device = "cuda:0"
    torch.cuda.set_device(device)
    print(f"sglang_omni from {sglang_omni.__file__}")
    steps = [("context", resident_mib())]
    tokenizer = load_qwen3_tts_tokenizer(args.model, device=device, dtype="bfloat16", attn_implementation=None)
    steps.append(("speech tokenizer", resident_mib()))
    with torch.inference_mode():
        silence_ids = derive_silence_codec_ids(tokenizer, device)
    steps.append((f"silence probe ({silence_ids.numel()} ids)", resident_mib()))
    batcher = Qwen3TTSRefCodeBatcher(tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
    steps.append(("reference graphs", resident_mib()))
    previous = steps[0][1]
    for name, resident in steps:
        print(f"{name}: resident {resident:.0f} MiB, step {resident - previous:+.0f} MiB")
        previous = resident
    batcher.close()


if __name__ == "__main__":
    main()
