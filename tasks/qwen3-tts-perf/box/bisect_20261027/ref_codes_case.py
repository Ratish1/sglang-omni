"""One reference clip's codes three ways in the serving image, on the speech tokenizer as the tree's loader leaves it
for a bf16 server: our batcher through its reference-encoder graphs (as served) and eagerly, and the official
tokenizer's own encode of the exact clip. Prints the
clip's samples and remainder against the hop, then for each pair the frames whose codes differ; saves the three code
tensors (frames, quantizers). The clip loads as both Qwen3TTSModel and our server load it (librosa, native rate, mono)
and the tokenizer resamples it.

usage: python3 ref_codes_case.py REF_AUDIO --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --out FILE.pt [--card 0]
"""

import argparse
import itertools
from pathlib import Path

import librosa
import torch

from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_audio")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    args = parser.parse_args()
    device = f"cuda:{args.card}"
    torch.cuda.set_device(device)
    tokenizer = load_qwen3_tts_tokenizer(args.model, device=device, dtype="bfloat16", attn_implementation=None)
    print(f"encoder dtype {next(tokenizer.model.encoder.parameters()).dtype}")
    eager = Qwen3TTSRefCodeBatcher(tokenizer)
    graphs = Qwen3TTSRefCodeBatcher(tokenizer, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
    waveform, sample_rate = librosa.load(args.ref_audio, sr=None, mono=True)
    samples = tokenizer._normalize_audio_inputs([waveform], sr=sample_rate)[0]
    hop = int(tokenizer.model.encode_downsample_rate)
    print(f"samples {samples.shape[-1]}, hop {hop}, remainder {samples.shape[-1] % hop}, frames {-(-samples.shape[-1] // hop)}")
    with torch.inference_mode():
        codes = {
            "ours_graphs": graphs.encode_waveform(waveform, sample_rate).cpu(),
            "ours_eager": eager.encode_waveform(waveform, sample_rate).cpu(),
            "official_env": tokenizer.encode([waveform], sr=sample_rate).audio_codes[0].cpu(),
        }
    torch.cuda.synchronize()
    eager.close()
    graphs.close()
    for first, second in itertools.combinations(codes, 2):
        left, right = codes[first], codes[second]
        frames = min(left.shape[0], right.shape[0])
        differing = torch.nonzero(~(left[:frames] == right[:frames]).all(dim=-1)).flatten().tolist()
        print(
            f"{first} {tuple(left.shape)} / {second} {tuple(right.shape)}: frames differing {differing}, "
            f"codes differing {int((left[:frames] != right[:frames]).sum())}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(codes, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
