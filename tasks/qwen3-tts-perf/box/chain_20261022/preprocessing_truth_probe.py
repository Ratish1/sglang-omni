"""Qwen3-TTS reference preprocessing against the model's fp32 output: per reference, the speaker embedding and the
reference codes as serving computes them on the tree on PYTHONPATH (the prompt frontend's speaker graphs and the
reference code batcher, both captured at the serving ladder, bf16), or, with --truth, as the checkpoint's own code
computes them in fp32 (the official mel_spectrogram and the speaker encoder in fp32; the official tokenizer's encode in
fp32). --compare reads a truth file and two arms: per group (SeedTTS EN references, long references) each arm's
embedding cosine to truth and its codes' agreement with truth, and the second arm minus the first, paired by reference,
with a 95 % interval.

usage: PYTHONPATH=<tree> python3 preprocessing_truth_probe.py OUT.pt [--truth] [--card 3] [--meta L ...]
       python3 preprocessing_truth_probe.py --compare TRUTH.pt A.pt B.pt
"""

import argparse
import math
import os
import statistics
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts import request_builders
from sglang_omni.models.qwen3_tts.prompt_frontend import load_qwen3_tts_prompt_frontend
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.stages import (
    apply_qwen_tts_transformers_compatibility_patches,
    load_qwen3_tts_tokenizer,
    register_qwen3_tts_hf_config,
)
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
RATE = 24000


def references(metas: list[Path]) -> list[tuple[str, str]]:
    """(group, wav path) for every SeedTTS EN reference, then every distinct reference of the long lists."""
    rows = [("seedtts", sample.ref_audio) for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")]
    long_paths = sorted(
        {str(meta.parent / line.split("|")[2]) for meta in metas for line in meta.read_text().splitlines() if line}
    )
    return rows + [("long", path) for path in long_paths]


def waveform_at_rate(path: str) -> np.ndarray:
    audio, source_rate = soundfile.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    else:
        pass
    if source_rate != RATE:
        return librosa.resample(audio, orig_sr=source_rate, target_sr=RATE).astype(np.float32)
    else:
        return audio


def served(rows: list[tuple[str, str]], device: torch.device) -> dict[str, list[torch.Tensor]]:
    register_qwen3_tts_hf_config()
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    frontend = load_qwen3_tts_prompt_frontend(resolve_checkpoint(MODEL), device=device, dtype=torch.bfloat16)
    runner = frontend.speaker_encoder_graph_runner
    runner.capture(DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES, tokenizer.model.encode_downsample_rate)
    batcher = request_builders.Qwen3TTSRefCodeBatcher(
        tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
    )
    print(f"speaker graphs {runner.stats()}; reference graphs {batcher.graph_runner.stats()}", flush=True)
    embeddings, codes = [], []
    try:
        with torch.inference_mode():
            for _, path in rows:
                waveform = waveform_at_rate(path)
                embeddings.append(runner.embed(waveform).float().cpu())
                codes.append(batcher.encode(waveform, RATE).cpu())
    finally:
        batcher.close()
    print(f"speaker replays {runner.replays} misses {runner.misses}; reference {batcher.graph_runner.stats()}")
    return {"embeddings": embeddings, "codes": codes}


def truth(rows: list[tuple[str, str]], device: torch.device) -> dict[str, list[torch.Tensor]]:
    register_qwen3_tts_hf_config()
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts import Qwen3TTSTokenizer  # noqa: import-outside-toplevel  # needs the compatibility patches first
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram  # noqa: import-outside-toplevel  # same

    checkpoint = resolve_checkpoint(MODEL)
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        os.path.join(checkpoint, "speech_tokenizer"), device_map=str(device), dtype=torch.float32
    )
    frontend = load_qwen3_tts_prompt_frontend(checkpoint, device=device, dtype=torch.float32)
    embeddings, codes = [], []
    with torch.inference_mode():
        for _, path in rows:
            waveform = waveform_at_rate(path)
            mels = mel_spectrogram(
                torch.from_numpy(waveform).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=RATE,
                hop_size=256, win_size=1024, fmin=0, fmax=12000,
            ).transpose(1, 2)
            embeddings.append(frontend.speaker_encoder(mels.to(device))[0].float().cpu())
            codes.append(tokenizer.encode([waveform], sr=RATE).audio_codes[0].cpu())
    return {"embeddings": embeddings, "codes": codes}


def interval(values: list[float]) -> str:
    mean = statistics.mean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else float("nan")
    return f"{mean:+.6f} (95 % {mean - half:+.6f} to {mean + half:+.6f})"


def compare(truth_path: str, first_path: str, second_path: str) -> None:
    reference, first, second = (torch.load(path) for path in (truth_path, first_path, second_path))
    assert reference["rows"] == first["rows"] == second["rows"]
    for group in ("seedtts", "long"):
        picked = [i for i, (row_group, _) in enumerate(reference["rows"]) if row_group == group]
        cosines = {}
        frames = {}
        for name, arm in (("a", first), ("b", second)):
            cosines[name] = [
                torch.nn.functional.cosine_similarity(arm["embeddings"][i], reference["embeddings"][i], dim=0).item()
                for i in picked
            ]
            frames[name] = []
            length_mismatches = 0
            for i in picked:
                served_codes, truth_codes = arm["codes"][i], reference["codes"][i]
                common = min(served_codes.shape[0], truth_codes.shape[0])
                length_mismatches += served_codes.shape != truth_codes.shape
                frames[name].append((served_codes[:common] == truth_codes[:common]).all(dim=1).float().mean().item())
            print(f"{group} {name}: code lengths differing from fp32 on {length_mismatches} references")
        print(f"{group}: {len(picked)} references")
        for name in ("a", "b"):
            print(
                f"  {name}: embedding cosine to fp32 mean {statistics.mean(cosines[name]):.6f} median "
                f"{statistics.median(cosines[name]):.6f} min {min(cosines[name]):.6f}; frames with every code equal to "
                f"fp32 mean {statistics.mean(frames[name]):.4f} min {min(frames[name]):.4f}"
            )
        cosine_delta = [b - a for a, b in zip(cosines["a"], cosines["b"])]
        frame_delta = [b - a for a, b in zip(frames["a"], frames["b"])]
        print(f"  b - a: cosine {interval(cosine_delta)}, b closer on {sum(d > 0 for d in cosine_delta)}, "
              f"equal on {sum(d == 0 for d in cosine_delta)}")
        print(f"  b - a: frames equal to fp32 {interval(frame_delta)}, identical codes on "
              f"{sum(torch.equal(first['codes'][i], second['codes'][i]) for i in picked)} references")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?")
    parser.add_argument("--truth", action="store_true")
    parser.add_argument("--card", type=int, default=3)
    parser.add_argument("--meta", type=Path, nargs="*", default=[])
    parser.add_argument("--compare", nargs=3)
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    else:
        pass
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    rows = references(args.meta)
    result = truth(rows, device) if args.truth else served(rows, device)
    torch.save({"rows": rows, **result}, args.out)
    print(f"wrote {len(rows)} references to {args.out}")


if __name__ == "__main__":
    main()
