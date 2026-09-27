"""Speaker embeddings of the T3 runtime runner against main's exact path, every clip.

Loads the prompt frontend from the tree on PYTHONPATH (the T3 branch), captures the
runner's graphs under a CUDA default device as the engine does, and for every distinct
seed-tts reference clip computes main's embedding (qwen-tts mel_spectrogram, the eager
encoder in bf16, the path main serves) and the runner's (cached mel, graph replay). Two
runner passes, forward then reverse order, so every bucket's stale tail differs; the
valid frames must not see the tail, so the passes must be bitwise equal. Cosine and max
abs of runner against main, with main bf16 against an fp32 eager (tf32 off) as the
floor, and the same floor for the runner.

usage: python speaker_runner_numerics.py --model ID [--meta ...] [--out FILE]
"""

from __future__ import annotations

import argparse
import copy
import json

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.prompt_frontend import load_qwen3_tts_prompt_frontend
from sglang_omni.models.qwen3_tts.speaker_encoder_cuda_graph import (
    DEFAULT_QWEN3_TTS_SPEAKER_ENCODER_BUCKET_FRAMES,
)
from sglang_omni.models.qwen3_tts.stages import register_qwen3_tts_hf_config
from sglang_omni.utils.checkpoint import resolve_checkpoint


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--out")
    args = parser.parse_args()
    device = torch.device("cuda")
    register_qwen3_tts_hf_config()
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    # the engine constructs the model and captures under a CUDA default device
    with torch.device(device):
        frontend = load_qwen3_tts_prompt_frontend(
            resolve_checkpoint(args.model), device=device, dtype=torch.bfloat16
        )
        frontend.speaker_encoder_graph_runner.capture(
            DEFAULT_QWEN3_TTS_SPEAKER_ENCODER_BUCKET_FRAMES
        )
    encoder = frontend.speaker_encoder.eval()
    runner = frontend.speaker_encoder_graph_runner
    rate = frontend.speaker_encoder_sample_rate
    encoder32 = copy.deepcopy(encoder).float()
    print(f"graphs captured for {sorted(runner.graphs)}")

    clips: dict[str, np.ndarray] = {}
    for sample in load_seedtts_samples(args.meta, 100000, split="en"):
        if sample.ref_audio in clips:
            continue
        waveform, sr = librosa.load(sample.ref_audio, sr=None, mono=True)
        waveform = waveform.astype(np.float32)
        if sr != rate:
            waveform = librosa.resample(y=waveform, orig_sr=int(sr), target_sr=rate)
        clips[sample.ref_audio] = waveform
    names = list(clips)
    print(f"{len(names)} distinct reference clips")

    def main_path(waveform: np.ndarray, module, dtype) -> torch.Tensor:
        mels = mel_spectrogram(
            torch.from_numpy(waveform).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return module(mels.to(device).to(dtype))[0]

    rows = []
    with torch.inference_mode():
        forward = {name: runner.embed(clips[name]).clone() for name in names}
        torch.cuda.synchronize()
        reverse = {name: runner.embed(clips[name]).clone() for name in reversed(names)}
        torch.cuda.synchronize()
        for name in names:
            waveform = clips[name]
            eager = main_path(waveform, encoder, torch.bfloat16)
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            eager32 = main_path(waveform, encoder32, torch.float32)
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            branch = forward[name]
            rows.append(
                {
                    "clip": name,
                    "frames": int(-(-len(waveform) // 256)) + 1,
                    "passes_bitwise": bool(torch.equal(branch, reverse[name])),
                    "cos_branch_main": cosine(branch, eager),
                    "cos_main_fp32": cosine(eager, eager32),
                    "cos_branch_fp32": cosine(branch, eager32),
                    "max_abs_branch_main": float(
                        (branch.float() - eager.float()).abs().max()
                    ),
                    "max_abs_main_fp32": float((eager.float() - eager32).abs().max()),
                    "norm_main": float(eager.float().norm()),
                    "norm_branch": float(branch.float().norm()),
                    "finite": bool(torch.isfinite(branch).all()),
                }
            )
    print(f"replays {runner.replays}, misses {runner.misses}")
    keys = (
        "cos_branch_main",
        "cos_main_fp32",
        "cos_branch_fp32",
        "max_abs_branch_main",
        "max_abs_main_fp32",
    )
    print(f"{'metric':<22}{'mean':>12}{'min':>12}{'max':>12}")
    for key in keys:
        values = [row[key] for row in rows]
        print(
            f"{key:<22}{sum(values) / len(values):>12.6f}{min(values):>12.6f}{max(values):>12.6f}"
        )
    print(
        f"passes bitwise equal: {sum(row['passes_bitwise'] for row in rows)} of {len(rows)}; "
        f"all finite: {all(row['finite'] for row in rows)}"
    )
    worst = sorted(rows, key=lambda row: row["cos_branch_main"])[:5]
    for row in worst:
        print(
            f"  worst {row['clip'].rsplit('/', 1)[-1]} frames {row['frames']} "
            f"cos {row['cos_branch_main']:.6f} floor {row['cos_main_fp32']:.6f}"
        )
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(rows, handle, indent=1)
    else:
        pass


if __name__ == "__main__":
    main()
