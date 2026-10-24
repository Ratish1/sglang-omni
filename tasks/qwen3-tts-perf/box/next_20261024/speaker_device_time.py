"""Device time of the Qwen3-TTS speaker embedding per SeedTTS EN reference, on the tree on PYTHONPATH (one arm per
process): the prompt frontend's speaker graphs captured at the serving ladder, each reference resampled as the request
builder does, embed() called once to warm and then REPEATS times under the torch profiler; the sum of every device
kernel's time over the profiled calls, per call, and the host wall per call.

usage: PYTHONPATH=<tree> python3 speaker_device_time.py [--card 3] [--references 64] [--repeats 5]
"""

import argparse
import statistics
import time

import librosa
import soundfile
import torch
from torch.profiler import ProfilerActivity, profile

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.prompt_frontend import load_qwen3_tts_prompt_frontend
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer, register_qwen3_tts_hf_config
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
    parser.add_argument("--references", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    register_qwen3_tts_hf_config()
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    frontend = load_qwen3_tts_prompt_frontend(resolve_checkpoint(MODEL), device=device, dtype=torch.bfloat16)
    runner = frontend.speaker_encoder_graph_runner
    runner.capture(DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES, tokenizer.model.encode_downsample_rate)
    rate = frontend.speaker_encoder_sample_rate
    waveforms = []
    for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, args.references, split="en"):
        audio, source_rate = soundfile.read(sample.ref_audio, dtype="float32")
        waveforms.append(librosa.resample(audio, orig_sr=source_rate, target_sr=rate) if source_rate != rate else audio)
    with torch.inference_mode():
        for waveform in waveforms:
            runner.embed(waveform)
        torch.cuda.synchronize(device)
        walls_ms = []
        with profile(activities=[ProfilerActivity.CUDA]) as profiler:
            for _ in range(args.repeats):
                for waveform in waveforms:
                    start = time.perf_counter()
                    runner.embed(waveform)
                    torch.cuda.synchronize(device)
                    walls_ms.append(1000 * (time.perf_counter() - start))
    calls = args.repeats * len(waveforms)
    kernels = [event for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    device_us = sum(event.device_time for event in kernels)
    print(f"speaker embed over {calls} calls: device {device_us / calls:.1f} us a call, {len(kernels) / calls:.1f} kernels "
          f"a call; host wall median {statistics.median(walls_ms):.3f} ms; replays {runner.replays} misses {runner.misses}")


if __name__ == "__main__":
    main()
