"""Our served reference codes against official's exact-length encode on the real checkpoint, and the hooked encoder's
shared state: the tree's loader (encoder float32, tail hooks) feeds our batcher through its graphs and eagerly; an
independently loaded official tokenizer (float32, no hooks) encodes each clip at its exact length. Prints, per pair,
the share of codes equal, codebook 0 equal, clips fully equal and the frames that differ; then the stale-state case:
the hooked tokenizer's own encode of a long clip right after our batcher encoded a short one.

usage: python3 exact_length_check.py WAV_DIR --model Qwen/Qwen3-TTS-12Hz-1.7B-Base [--card 0] [--clips 200]
"""

import argparse
import itertools
import os
from pathlib import Path

import librosa
import torch

from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.checkpoint import resolve_checkpoint


def compare(first: list[torch.Tensor], second: list[torch.Tensor]) -> str:
    equal = total = codebook0 = frames_total = fully = 0
    parted = []
    for left, right in zip(first, second):
        frames = min(left.shape[0], right.shape[0])
        same = left[:frames] == right[:frames]
        equal += int(same.sum())
        total += same.numel()
        codebook0 += int(same[:, 0].sum())
        frames_total += frames
        rows = torch.nonzero(~same.all(dim=-1)).flatten().tolist()
        if not rows and left.shape == right.shape:
            fully += 1
        else:
            parted.append((rows[:3], frames))
    return (
        f"codes {equal / total:.4f}, codebook 0 {codebook0 / frames_total:.4f}, clips fully equal {fully}/{len(first)}, "
        f"first parts (frames differing, clip frames) {parted[:6]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--clips", type=int, default=200)
    parser.add_argument("--tf32", action="store_true")
    args = parser.parse_args()
    device = f"cuda:{args.card}"
    torch.cuda.set_device(device)
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts import Qwen3TTSTokenizer

    ours = load_qwen3_tts_tokenizer(args.model, device=device, dtype="bfloat16", attn_implementation=None)
    official = Qwen3TTSTokenizer.from_pretrained(
        os.path.join(resolve_checkpoint(args.model), "speech_tokenizer"), device_map=device, dtype=torch.float32
    )
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    eager = Qwen3TTSRefCodeBatcher(ours)
    graphs = Qwen3TTSRefCodeBatcher(ours, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
    arms: dict[str, list[torch.Tensor]] = {"official_exact": [], "ours_graphs": [], "ours_eager": []}
    clips = []
    for path in sorted(args.wavs.glob("*.wav"))[: args.clips]:
        waveform, sample_rate = librosa.load(str(path), sr=None, mono=True)
        clips.append((waveform, sample_rate))
        with torch.inference_mode():
            torch.backends.cuda.matmul.allow_tf32 = False
            arms["official_exact"].append(official.encode([waveform], sr=sample_rate).audio_codes[0].cpu())
            torch.backends.cuda.matmul.allow_tf32 = args.tf32
            arms["ours_graphs"].append(graphs.encode_waveform(waveform, sample_rate).cpu())
            arms["ours_eager"].append(eager.encode_waveform(waveform, sample_rate).cpu())
        torch.cuda.synchronize()
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"clips {len(clips)}, encoder dtype {next(ours.model.encoder.parameters()).dtype}, ours with TF32 matmuls {args.tf32}")
    for first, second in itertools.combinations(arms, 2):
        print(f"{first} / {second}: {compare(arms[first], arms[second])}")
    lengths = [waveform.size / sample_rate for waveform, sample_rate in clips]
    short = clips[lengths.index(min(lengths))]
    long = clips[lengths.index(max(lengths))]
    with torch.inference_mode():
        graphs.encode_waveform(*short)
        hooked = ours.encode([long[0]], sr=long[1]).audio_codes[0].cpu()
        reference = official.encode([long[0]], sr=long[1]).audio_codes[0].cpu()
    torch.cuda.synchronize()
    print(
        f"stale state: the hooked tokenizer's own encode of the longest clip ({max(lengths):.2f} s) after our batcher "
        f"encoded the shortest ({min(lengths):.2f} s): {compare([hooked], [reference])}"
    )
    eager.close()
    graphs.close()


if __name__ == "__main__":
    main()
