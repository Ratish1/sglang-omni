"""Block 8 component check: which precision the reference encoder's transformer needs for official's codes, and what
each costs in device time per encode.

On PR D's tree the encoder is float32 (official's precision); its transformer's GEMMs run on FFMA (484 us of the 0.77
ms an encode costs over main's bf16 encoder), while the convs already take TF32 through cuDNN as official's do. A global
TF32 switch also moved the quantizer's `torch.cdist` onto TF32, which picks codes directly. Here only the
`nn.Linear` layers of `encoder.encoder_transformer` change:
  float32  as served (FFMA)
  tf32     TF32 tensor cores, one pass
  tf32x3   three TF32 passes on split operands (high and low parts), float32-level products on tensor cores
  bf16     bf16 GEMMs, float32 in and out
Per variant, a fresh batcher (its graphs captured after the change) and an eager batcher encode every clip; codes are
compared with official's float32 exact-length encode (TF32 matmuls off), and the first 200 clips are encoded again under
the profiler for device time and kernels per encode.

usage: PYTHONPATH=D_TREE python3 encoder_precision_variants.py WAV_DIR --model Qwen/Qwen3-TTS-12Hz-1.7B-Base [--card 2]
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.checkpoint import resolve_checkpoint

VARIANTS = ("float32", "tf32", "tf32x3", "bf16")
TF32_LOW_BITS = -8192
TIMED_CLIPS = 200
TOP_KERNELS = 6


def tf32_split(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The operand's TF32-exact high part (sign, exponent, 10 mantissa bits) and the float32 rest."""
    high = (tensor.contiguous().view(torch.int32) & TF32_LOW_BITS).view(torch.float32)
    return high, tensor - high


def variant_forward(module: torch.nn.Linear, variant: str):
    weight = module.weight
    bias = module.bias
    weight_high, weight_low = tf32_split(weight)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = None if bias is None else bias.to(torch.bfloat16)

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        if variant == "bf16":
            return F.linear(hidden_states.to(torch.bfloat16), weight_bf16, bias_bf16).to(torch.float32)
        else:
            pass
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            if variant == "tf32":
                return F.linear(hidden_states, weight, bias)
            else:
                input_high, input_low = tf32_split(hidden_states)
                output = F.linear(input_high, weight_high, bias)
                output = output + F.linear(input_high, weight_low)
                return output + F.linear(input_low, weight_high)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    return forward


def install(linears: list[torch.nn.Linear], variant: str) -> None:
    for module in linears:
        if "forward" in module.__dict__:
            del module.forward
        else:
            pass
        if variant != "float32":
            module.forward = variant_forward(module, variant)
        else:
            pass


def compare(first: list[torch.Tensor], second: list[torch.Tensor]) -> str:
    equal = total = codebook0 = frames_total = fully = 0
    for left, right in zip(first, second):
        frames = min(left.shape[0], right.shape[0])
        same = left[:frames] == right[:frames]
        equal += int(same.sum())
        total += same.numel()
        codebook0 += int(same[:, 0].sum())
        frames_total += frames
        fully += int(bool(same.all()) and left.shape == right.shape)
    return f"codes {equal / total:.4f}, codebook 0 {codebook0 / frames_total:.4f}, clips fully equal {fully}/{len(first)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wavs", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--clips", type=int, default=1088)
    args = parser.parse_args()
    device = f"cuda:{args.card}"
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    apply_qwen_tts_transformers_compatibility_patches()
    from qwen_tts import Qwen3TTSTokenizer

    clips = [librosa.load(str(path), sr=None, mono=True) for path in sorted(args.wavs.glob("*.wav"))[: args.clips]]
    ours = load_qwen3_tts_tokenizer(args.model, device=device, dtype="bfloat16", attn_implementation=None)
    official = Qwen3TTSTokenizer.from_pretrained(
        os.path.join(resolve_checkpoint(args.model), "speech_tokenizer"), device_map=device, dtype=torch.float32
    )
    linears = [
        module for module in ours.model.encoder.encoder_transformer.modules() if isinstance(module, torch.nn.Linear)
    ]
    print(
        f"clips {len(clips)}, encoder {next(ours.model.encoder.parameters()).dtype}, transformer linears {len(linears)}, "
        f"cudnn tf32 {torch.backends.cudnn.allow_tf32}",
        flush=True,
    )
    with torch.inference_mode():
        reference = [official.encode([waveform], sr=rate).audio_codes[0].cpu() for waveform, rate in clips]
    for variant in VARIANTS:
        install(linears, variant)
        graphs = Qwen3TTSRefCodeBatcher(ours, graph_bucket_frames=DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES)
        eager = Qwen3TTSRefCodeBatcher(ours)
        with torch.inference_mode():
            graph_codes = [graphs.encode_waveform(waveform, rate).cpu() for waveform, rate in clips]
            eager_codes = [eager.encode_waveform(waveform, rate).cpu() for waveform, rate in clips]
            torch.cuda.synchronize()
            walls_ms = []
            for waveform, rate in clips[:TIMED_CLIPS]:
                began = time.perf_counter()
                graphs.encode_waveform(waveform, rate)
                torch.cuda.synchronize()
                walls_ms.append(1e3 * (time.perf_counter() - began))
            with profile(activities=[ProfilerActivity.CUDA]) as profiler:
                for waveform, rate in clips[:TIMED_CLIPS]:
                    graphs.encode_waveform(waveform, rate)
                torch.cuda.synchronize()
        graphs.close()
        eager.close()
        totals_us: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for event in profiler.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                totals_us[event.name] += event.device_time
                counts[event.name] += 1
            else:
                pass
        encodes = min(len(clips), TIMED_CLIPS)
        print(
            f"== {variant}: device {sum(totals_us.values()) / 1e3 / encodes:.3f} ms and "
            f"{sum(counts.values()) / encodes:.1f} kernels per encode, wall q50 {np.median(walls_ms):.2f} ms",
            flush=True,
        )
        print(f"   official / graphs: {compare(reference, graph_codes)}", flush=True)
        print(f"   official / eager:  {compare(reference, eager_codes)}", flush=True)
        for name, total_us in sorted(totals_us.items(), key=lambda item: -item[1])[:TOP_KERNELS]:
            print(f"   {total_us / encodes:8.1f} us, {counts[name] / encodes:5.1f} calls: {name[:100]}", flush=True)
    install(linears, "float32")


if __name__ == "__main__":
    main()
