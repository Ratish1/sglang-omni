"""Device memory of the reference encoder as the tree serves it: the speech tokenizer after load, the batcher's graphs
after capture, and the peak over eager and graph encodes of a 2 s and a 20 s clip. Run once per tree.

usage: PYTHONPATH=TREE python3 encoder_memory.py --model Qwen/Qwen3-TTS-12Hz-1.7B-Base
"""

import argparse

import numpy as np
import torch

import sglang_omni
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer


def mib(value: int) -> str:
    return f"{value / (1 << 20):.1f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    device = "cuda:0"
    torch.cuda.set_device(device)
    print(f"sglang_omni from {sglang_omni.__file__}")
    before = torch.cuda.memory_allocated()
    tokenizer = load_qwen3_tts_tokenizer(args.model, device=device, dtype="bfloat16", attn_implementation=None)
    torch.cuda.synchronize()
    loaded = torch.cuda.memory_allocated()
    encoder = tokenizer.model.encoder
    encoder_bytes = sum(p.numel() * p.element_size() for p in encoder.parameters())
    encoder_bytes += sum(b.numel() * b.element_size() for b in encoder.buffers())
    print(
        f"encoder dtype {next(encoder.parameters()).dtype}, encoder parameters and buffers {mib(encoder_bytes)}, "
        f"tokenizer allocated {mib(loaded - before)}"
    )
    graphs = Qwen3TTSRefCodeBatcher(tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
    torch.cuda.synchronize()
    captured = torch.cuda.memory_allocated()
    print(f"graphs captured: allocated {mib(captured - loaded)} more, reserved {mib(torch.cuda.memory_reserved())}")
    eager = Qwen3TTSRefCodeBatcher(tokenizer)
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(0)
    for seconds in (2, 20):
        waveform = (0.1 * rng.standard_normal(24000 * seconds)).astype(np.float32)
        with torch.inference_mode():
            graphs.encode_waveform(waveform, 24000)
            eager.encode_waveform(waveform, 24000)
        torch.cuda.synchronize()
    print(f"peak over 2 s and 20 s encodes: {mib(torch.cuda.max_memory_allocated() - captured)} above the captured state")
    graphs.close()
    eager.close()


if __name__ == "__main__":
    main()
