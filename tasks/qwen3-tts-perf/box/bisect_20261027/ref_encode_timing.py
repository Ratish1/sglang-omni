"""Time per reference encode through our reference batcher as served (its CUDA graphs), on the bf16 tokenizer and on
the float32 one, over the same clips: each clip encoded once to warm, then timed with the device synchronized around
it. Prints the median, p90 and mean milliseconds per clip for each, and the clip lengths covered.

usage: python3 ref_encode_timing.py WAV_DIR --model Qwen/Qwen3-TTS-12Hz-1.7B-Base [--card 0] [--clips 200]
"""

import argparse
import time
from pathlib import Path

import librosa
import numpy as np
import torch

from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--clips", type=int, default=200)
    args = parser.parse_args()
    device = f"cuda:{args.card}"
    torch.cuda.set_device(device)
    clips = [
        librosa.load(str(path), sr=None, mono=True) for path in sorted(args.wavs.glob("*.wav"))[: args.clips]
    ]
    print(f"clips {len(clips)}, seconds q50 {np.median([len(w) / sr for w, sr in clips]):.2f}, "
          f"max {max(len(w) / sr for w, sr in clips):.2f}")
    for dtype in ("bfloat16", "float32"):
        tokenizer = load_qwen3_tts_tokenizer(args.model, device=device, dtype=dtype, attn_implementation=None)
        batcher = Qwen3TTSRefCodeBatcher(tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
        times_ms = []
        with torch.inference_mode():
            for waveform, sample_rate in clips:
                batcher.encode_waveform(waveform, sample_rate)
                torch.cuda.synchronize()
                began = time.perf_counter()
                batcher.encode_waveform(waveform, sample_rate)
                torch.cuda.synchronize()
                times_ms.append(1e3 * (time.perf_counter() - began))
        batcher.close()
        print(f"{dtype}: ms per clip q50 {np.median(times_ms):.2f}, q90 {np.quantile(times_ms, 0.9):.2f}, "
              f"mean {np.mean(times_ms):.2f}")


if __name__ == "__main__":
    main()
