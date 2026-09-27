"""Mapping and formal trace pair of the speaker encoder for the omni-gpu-deep-dive skill.

One real seed-tts clip (the clip whose mel length is closest to --seconds), the prompt
frontend's encoder. mapping: the runtime's eager call (stacks on, so every kernel names
its python line). formal: --formal eager replays the same eager call, the path main
serves; --formal graph replays the length bucketed forward from a CUDA graph captured
for the clip's bucket (the T3 design). Traces land in <out>/mapping and <out>/formal
and are gated for steady state by capture_pair.

usage: python speaker_encoder_pair.py --model ID --out DIR [--formal eager|graph]
       [--seconds 5] [--bucket 64] [--iters 20] [--warmup 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from speaker_encoder_bench import FMAX, FMIN, HOP, N_FFT, NUM_MELS, WIN, BucketGraph

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.prompt_frontend import load_qwen3_tts_prompt_frontend
from sglang_omni.models.qwen3_tts.stages import register_qwen3_tts_hf_config
from sglang_omni.utils.checkpoint import resolve_checkpoint

sys.path.insert(0, ".claude/skills/omni-gpu-deep-dive/scripts")
from omni_trace_pair import capture_pair  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--formal", choices=("eager", "graph"), default="eager")
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--bucket", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    device = torch.device("cuda")
    register_qwen3_tts_hf_config()
    frontend = load_qwen3_tts_prompt_frontend(
        resolve_checkpoint(args.model), device=device, dtype=torch.bfloat16
    )
    encoder = frontend.speaker_encoder.eval()
    rate = frontend.speaker_encoder_sample_rate

    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    clips = []
    for sample in load_seedtts_samples(args.meta, 200, split="en"):
        waveform, sr = librosa.load(sample.ref_audio, sr=None, mono=True)
        waveform = waveform.astype(np.float32)
        if sr != rate:
            waveform = librosa.resample(y=waveform, orig_sr=int(sr), target_sr=rate)
        clips.append(waveform)
    waveform = min(clips, key=lambda w: abs(len(w) / rate - args.seconds))
    mels = mel_spectrogram(
        torch.from_numpy(waveform).unsqueeze(0),
        n_fft=N_FFT,
        num_mels=NUM_MELS,
        sampling_rate=rate,
        hop_size=HOP,
        win_size=WIN,
        fmin=FMIN,
        fmax=FMAX,
    )
    mels = mels.to(device).to(torch.bfloat16)
    frames = mels.shape[2]
    eager_input = mels.transpose(1, 2)
    print(
        f"clip {len(waveform) / rate:.2f} s, {frames} mel frames, formal={args.formal}"
    )

    with torch.inference_mode():
        if args.formal == "graph":
            width = -(-frames // args.bucket) * args.bucket
            graph = BucketGraph(encoder, width, device, torch.bfloat16)
            print(f"bucket width {width}")

            def formal_body():
                return graph.run(mels, frames)

        else:

            def formal_body():
                return encoder(eager_input)[0]

        capture_pair(
            output_dir=Path(args.out),
            mapping_body=lambda: encoder(eager_input)[0],
            formal_body=formal_body,
            iters=args.iters,
            warmup=args.warmup,
        )


if __name__ == "__main__":
    main()
