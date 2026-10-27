"""Reference codes of the Qwen3-TTS codec encoder four ways on the same clips: official's tokenizer encode of one clip
at its exact length in bf16 and in fp32, and our reference batcher in bf16 eagerly and through its CUDA graphs as
served. Prints, for every pair, the share of codes that agree overall and on codebook 0, the share of clips with every
code equal, and the median first frame where a clip's codes part.

usage: python3 ref_codes_check.py WAV_DIR --model Qwen/Qwen3-TTS-12Hz-0.6B-Base [--card 0] [--clips 200]
"""

import argparse
import itertools
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
    bf16 = load_qwen3_tts_tokenizer(args.model, device=device, dtype="bfloat16", attn_implementation=None)
    fp32 = load_qwen3_tts_tokenizer(args.model, device=device, dtype="float32", attn_implementation=None)
    eager = Qwen3TTSRefCodeBatcher(bf16)
    graphs = Qwen3TTSRefCodeBatcher(bf16, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
    arms: dict[str, list[torch.Tensor]] = {"official_bf16": [], "official_fp32": [], "ours_eager": [], "ours_graphs": []}
    for path in sorted(args.wavs.glob("*.wav"))[: args.clips]:
        waveform, sample_rate = librosa.load(str(path), sr=None, mono=True)
        with torch.inference_mode():
            arms["official_bf16"].append(bf16.encode([waveform], sr=sample_rate).audio_codes[0].cpu())
            arms["official_fp32"].append(fp32.encode([waveform], sr=sample_rate).audio_codes[0].cpu())
            arms["ours_eager"].append(eager.encode_waveform(waveform, sample_rate).cpu())
            arms["ours_graphs"].append(graphs.encode_waveform(waveform, sample_rate).cpu())
        torch.cuda.synchronize()
    eager.close()
    graphs.close()
    print(f"clips {len(arms['official_bf16'])}")
    print("pair | codes equal | codebook 0 equal | clips fully equal | median first frame parted (clips that part)")
    for first, second in itertools.combinations(arms, 2):
        equal_codes, codebook0, total_codes, total_frames, fully_equal, first_parted = 0, 0, 0, 0, 0, []
        for left, right in zip(arms[first], arms[second]):
            frames = min(left.shape[0], right.shape[0])
            equal = left[:frames].reshape(frames, -1) == right[:frames].reshape(frames, -1)
            equal_codes += int(equal.sum())
            total_codes += equal.numel()
            codebook0 += int(equal[:, 0].sum())
            total_frames += frames
            parted = torch.nonzero(~equal.all(dim=-1)).flatten()
            if parted.numel() == 0 and left.shape == right.shape:
                fully_equal += 1
            else:
                first_parted.append(int(parted[0]) if parted.numel() else frames)
        print(
            f"{first} / {second} | {equal_codes / total_codes:.4f} | {codebook0 / total_frames:.4f} | "
            f"{fully_equal}/{len(arms[first])} | {np.median(first_parted) if first_parted else '-'}"
        )


if __name__ == "__main__":
    main()
