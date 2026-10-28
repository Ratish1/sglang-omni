"""Device time of one reference encode as served, on the tree first on the path: the speech tokenizer loaded as the
server loads it (on the encoder branch a Base checkpoint's encoder is float32), the reference batcher with its CUDA
graphs, every clip encoded once to warm and then encoded again under the profiler. Prints the clips' lengths, the
device time and kernel count per encode, the wall time per encode with the device synchronized around it, and the
kernels that take the most device time in total.

usage: PYTHONPATH=TREE python3 reference_encode_census.py WAV_DIR --model Qwen/Qwen3-TTS-12Hz-1.7B-Base [--clips 200]
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

import sglang_omni
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

TOP_KERNELS = 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--clips", type=int, default=200)
    args = parser.parse_args()
    device = "cuda:0"
    torch.cuda.set_device(device)
    clips = [librosa.load(str(path), sr=None, mono=True) for path in sorted(args.wavs.glob("*.wav"))[: args.clips]]
    seconds = [len(waveform) / sample_rate for waveform, sample_rate in clips]
    print(f"sglang_omni from {sglang_omni.__file__}", flush=True)
    print(f"clips {len(clips)}, seconds q50 {np.median(seconds):.2f}, max {max(seconds):.2f}", flush=True)
    tokenizer = load_qwen3_tts_tokenizer(args.model, device=device, dtype="bfloat16", attn_implementation=None)
    encoder_dtypes = sorted({str(parameter.dtype) for parameter in tokenizer.model.encoder.parameters()})
    batcher = Qwen3TTSRefCodeBatcher(tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
    walls_ms = []
    with torch.inference_mode():
        for waveform, sample_rate in clips:
            batcher.encode_waveform(waveform, sample_rate)
        torch.cuda.synchronize()
        for waveform, sample_rate in clips:
            began = time.perf_counter()
            batcher.encode_waveform(waveform, sample_rate)
            torch.cuda.synchronize()
            walls_ms.append(1e3 * (time.perf_counter() - began))
        with profile(activities=[ProfilerActivity.CUDA]) as profiler:
            for waveform, sample_rate in clips:
                batcher.encode_waveform(waveform, sample_rate)
            torch.cuda.synchronize()
    batcher.close()
    totals_us: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            totals_us[event.name] += event.device_time
            counts[event.name] += 1
        else:
            pass
    encodes = len(clips)
    device_ms = sum(totals_us.values()) / 1e3 / encodes
    kernels = sum(counts.values()) / encodes
    print(f"encoder {', '.join(encoder_dtypes)}: device {device_ms:.3f} ms and {kernels:.1f} kernels per encode, "
          f"wall q50 {np.median(walls_ms):.2f} ms, mean {np.mean(walls_ms):.2f}", flush=True)
    for name, total_us in sorted(totals_us.items(), key=lambda item: -item[1])[:TOP_KERNELS]:
        print(f"  {total_us / encodes:8.1f} us per encode, {counts[name] / encodes:5.1f} calls: {name[:110]}", flush=True)


if __name__ == "__main__":
    main()
