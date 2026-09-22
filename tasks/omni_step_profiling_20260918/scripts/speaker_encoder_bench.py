"""Speaker embedding (T3) bench: the runtime's eager path against a length bucketed forward.

No server. Loads the prompt frontend (the speaker encoder as the preprocessing process
has it), takes real seed-tts reference clips, and measures per clip:

  mel      the runtime's mel_spectrogram (filterbank rebuilt per call) against the same
           with the filterbank and window cached, on the CPU and on the GPU: ms and the
           max abs difference.
  eager    the runtime's encoder call on the clip's own length, cold (a new length, so
           the cuDNN plan cache misses) and warm (the same call again).
  bucket   the clip padded to a length bucket through a length aware forward over the
           module's own weights (reflect pads gathered from the true length, the SE mean
           and the pooling masked), eager after a warm call per bucket, and replayed
           from a CUDA graph captured per bucket.
  quality  cosine of every arm against the eager call, with the bf16 eager against an
           fp32 eager (tf32 off) as the noise floor, and the bucketed forward against the
           eager forward both in fp32 as the exactness proof.

usage: python speaker_encoder_bench.py --model DIR [--meta zhaochenyang20/seed-tts-eval-arrow]
       [--samples 60] [--bucket 64] [--reps 5]
"""

from __future__ import annotations

import argparse
import copy
import statistics
import time

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.prompt_frontend import load_qwen3_tts_prompt_frontend
from sglang_omni.models.qwen3_tts.stages import register_qwen3_tts_hf_config

N_FFT, NUM_MELS, HOP, WIN, FMIN, FMAX = 1024, 128, 256, 1024, 0, 12000
CLIP_VAL = 1e-5


def mel_cached(
    y: torch.Tensor, basis: torch.Tensor, window: torch.Tensor
) -> torch.Tensor:
    """mel_spectrogram of qwen-tts with the filterbank and window built once."""
    padding = (N_FFT - HOP) // 2
    y = F.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    return torch.log(torch.clamp(torch.matmul(basis, spec), min=CLIP_VAL))


def reflect_index(length: torch.Tensor, width: int, pad: int) -> torch.Tensor:
    """Gather index of a reflect pad by pad on each side of a signal of `length` valid
    positions inside a buffer of `width`; positions past the valid end are clamped."""
    pos = torch.arange(-pad, width + pad, device=length.device)
    index = pos.abs()
    over = index - (length - 1)
    index = torch.where(over > 0, (length - 1) - over, index)
    return index.clamp(0, width - 1)


def conv_same(
    conv: torch.nn.Conv1d, x: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    pad = conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
    if pad:
        x = x.index_select(2, reflect_index(length, x.shape[2], pad))
    return F.conv1d(x, conv.weight, conv.bias, dilation=conv.dilation)


def tdnn(block, x, length):
    return F.relu(conv_same(block.conv, x, length))


def encode_bucketed(encoder, mels: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """The ECAPA forward over the encoder's weights, exact for the first `length`
    frames of mels [1, C, T] whatever fills the rest of T."""
    width = mels.shape[2]
    mask = (torch.arange(width, device=mels.device) < length).to(mels.dtype)[None, None]
    x = mels
    outputs = []
    for i, layer in enumerate(encoder.blocks):
        if i == 0:
            x = tdnn(layer, x, length)
        else:
            residual = x
            x = tdnn(layer.tdnn1, x, length)
            parts = []
            for j, part in enumerate(torch.chunk(x, layer.res2net_block.scale, dim=1)):
                if j == 0:
                    out = part
                elif j == 1:
                    out = tdnn(layer.res2net_block.blocks[j - 1], part, length)
                else:
                    out = tdnn(layer.res2net_block.blocks[j - 1], part + out, length)
                parts.append(out)
            x = torch.cat(parts, dim=1)
            x = tdnn(layer.tdnn2, x, length)
            se = layer.se_block
            mean = (x * mask).sum(2, keepdim=True) / length.to(x.dtype)
            gate = torch.sigmoid(se.conv2(F.relu(se.conv1(mean))))
            x = x * gate + residual
        outputs.append(x)
    x = torch.cat(outputs[1:], dim=1)
    x = tdnn(encoder.mfa, x, length)
    asp = encoder.asp
    x = x * mask
    weights = mask / length.to(x.dtype)
    mean = (weights * x).sum(2)
    std = torch.sqrt((weights * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(asp.eps))
    attention = torch.cat(
        [
            x,
            mean.unsqueeze(2).expand(-1, -1, width),
            std.unsqueeze(2).expand(-1, -1, width),
        ],
        dim=1,
    )
    attention = asp.conv(torch.tanh(tdnn(asp.tdnn, attention, length)))
    attention = attention.masked_fill(mask == 0, float("-inf"))
    attention = F.softmax(attention, dim=2)
    mean = (attention * x).sum(2)
    std = torch.sqrt((attention * (x - mean.unsqueeze(2)).pow(2)).sum(2).clamp(asp.eps))
    pooled = torch.cat((mean, std), dim=1).unsqueeze(2)
    return encoder.fc(pooled).squeeze(-1)[0]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def device_ms(fn, reps: int) -> tuple[float, float]:
    """Median device ms (CUDA events) and host wall ms of fn over reps."""
    device, wall = [], []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        torch.cuda.synchronize()
        t = time.perf_counter()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t) * 1e3)
        device.append(start.elapsed_time(end))
    return statistics.median(device), statistics.median(wall)


def cpu_ms(fn, reps: int) -> float:
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times)


class BucketGraph:
    def __init__(self, encoder, width: int, device, dtype):
        self.mels = torch.zeros((1, NUM_MELS, width), device=device, dtype=dtype)
        self.length = torch.full((1,), width, device=device, dtype=torch.long)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                encode_bucketed(encoder, self.mels, self.length)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.out = encode_bucketed(encoder, self.mels, self.length)

    def run(self, mels: torch.Tensor, length: int) -> torch.Tensor:
        self.mels[:, :, : mels.shape[2]].copy_(mels)
        self.length.fill_(length)
        self.graph.replay()
        return self.out


def row(label: str, values: list[float]) -> None:
    values = sorted(values)
    print(
        f"  {label:<44}{len(values):>5}{statistics.fmean(values):>10.3f}"
        f"{values[len(values) // 2]:>10.3f}{values[-1]:>10.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--bucket", type=int, default=64)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()
    device = torch.device("cuda")
    register_qwen3_tts_hf_config()
    frontend = load_qwen3_tts_prompt_frontend(
        args.model, device=device, dtype=torch.bfloat16
    )
    encoder = frontend.speaker_encoder.eval()
    rate = frontend.speaker_encoder_sample_rate
    encoder32 = copy.deepcopy(encoder).float()
    print(
        f"speaker encoder on {torch.cuda.get_device_name(device)}, {rate} Hz, tf32 cudnn {torch.backends.cudnn.allow_tf32}"
    )

    clips = []
    for sample in load_seedtts_samples(args.meta, args.samples, split="en"):
        # note(ratish): the wrapper loads references this way, then resamples
        waveform, sr = librosa.load(sample.ref_audio, sr=None, mono=True)
        waveform = waveform.astype(np.float32)
        if sr != rate:
            waveform = librosa.resample(y=waveform, orig_sr=int(sr), target_sr=rate)
        clips.append(waveform)
    print(
        f"{len(clips)} clips, {min(len(c) for c in clips) / rate:.1f} to {max(len(c) for c in clips) / rate:.1f} s"
    )

    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    basis = torch.from_numpy(
        librosa_mel_fn(sr=rate, n_fft=N_FFT, n_mels=NUM_MELS, fmin=FMIN, fmax=FMAX)
    ).float()
    window = torch.hann_window(WIN)
    basis_gpu, window_gpu = basis.to(device), window.to(device)

    def runtime_mel(waveform):
        return mel_spectrogram(
            torch.from_numpy(waveform).unsqueeze(0),
            n_fft=N_FFT,
            num_mels=NUM_MELS,
            sampling_rate=rate,
            hop_size=HOP,
            win_size=WIN,
            fmin=FMIN,
            fmax=FMAX,
        )

    stats: dict[str, list[float]] = {}
    lengths = []
    graphs: dict[int, BucketGraph] = {}
    memory_before = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        for waveform in clips:
            y = torch.from_numpy(waveform).unsqueeze(0)
            mel_runtime = runtime_mel(waveform)
            mel_cache = mel_cached(y, basis, window)
            mel_gpu = mel_cached(y.to(device), basis_gpu, window_gpu)
            stats.setdefault("mel runtime cpu ms", []).append(
                cpu_ms(lambda: runtime_mel(waveform), args.reps)
            )
            stats.setdefault("mel cached cpu ms", []).append(
                cpu_ms(lambda: mel_cached(y, basis, window), args.reps)
            )
            stats.setdefault("mel cached gpu ms (device)", []).append(
                device_ms(
                    lambda: mel_cached(y.to(device), basis_gpu, window_gpu), args.reps
                )[0]
            )
            stats.setdefault("mel cached vs runtime max abs", []).append(
                float((mel_cache - mel_runtime).abs().max())
            )
            stats.setdefault("mel gpu vs runtime max abs", []).append(
                float((mel_gpu.cpu() - mel_runtime).abs().max())
            )

            mels = mel_runtime.to(device).to(torch.bfloat16)
            frames = mels.shape[2]
            lengths.append(frames)
            eager_input = mels.transpose(1, 2)
            torch.cuda.synchronize()
            t = time.perf_counter()
            eager = encoder(eager_input)[0]
            torch.cuda.synchronize()
            stats.setdefault("eager cold wall ms (new length)", []).append(
                (time.perf_counter() - t) * 1e3
            )
            dev, wall = device_ms(lambda: encoder(eager_input), args.reps)
            stats.setdefault("eager warm wall ms", []).append(wall)
            stats.setdefault("eager device ms", []).append(dev)

            width = -(-frames // args.bucket) * args.bucket
            padded = F.pad(mels, (0, width - frames))
            length = torch.tensor([frames], device=device)
            bucketed = encode_bucketed(encoder, padded, length)
            for _ in range(2):
                encode_bucketed(encoder, padded, length)
            dev, wall = device_ms(
                lambda: encode_bucketed(encoder, padded, length), args.reps
            )
            stats.setdefault("bucket eager warm wall ms", []).append(wall)
            stats.setdefault("bucket eager device ms", []).append(dev)
            if width not in graphs:
                graphs[width] = BucketGraph(encoder, width, device, torch.bfloat16)
            graph = graphs[width]
            replay = graph.run(mels, frames).clone()
            dev, wall = device_ms(lambda: graph.run(mels, frames), args.reps)
            stats.setdefault("bucket graph wall ms", []).append(wall)
            stats.setdefault("bucket graph device ms", []).append(dev)

            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            eager32 = encoder32(mel_runtime.to(device).transpose(1, 2))[0]
            bucket32 = encode_bucketed(
                encoder32, F.pad(mel_runtime.to(device), (0, width - frames)), length
            )
            torch.backends.cudnn.allow_tf32 = True
            stats.setdefault("cos eager bf16 vs eager fp32 (floor)", []).append(
                cosine(eager, eager32)
            )
            stats.setdefault("cos bucket bf16 vs eager bf16", []).append(
                cosine(bucketed, eager)
            )
            stats.setdefault("cos graph bf16 vs eager bf16", []).append(
                cosine(replay, eager)
            )
            stats.setdefault("cos bucket bf16 vs eager fp32", []).append(
                cosine(bucketed, eager32)
            )
            stats.setdefault("cos bucket fp32 vs eager fp32 (exactness)", []).append(
                cosine(bucket32, eager32)
            )
            stats.setdefault("max abs bucket fp32 vs eager fp32", []).append(
                float((bucket32 - eager32).abs().max())
            )
            stats.setdefault("max abs graph vs bucket bf16", []).append(
                float((replay - bucketed).abs().max())
            )

    widths = sorted(set(-(-f // args.bucket) * args.bucket for f in lengths))
    print(
        f"\nmel frames {min(lengths)} to {max(lengths)} ({len(set(lengths))} distinct of {len(lengths)}), "
        f"bucket {args.bucket}: {len(widths)} widths {widths}"
    )
    print(f"  {'metric':<44}{'n':>5}{'mean':>10}{'p50':>10}{'max':>10}")
    for label, values in stats.items():
        row(label, values)
    print(
        f"\n{len(graphs)} bucket graphs with their buffers: "
        f"{(torch.cuda.memory_allocated(device) - memory_before) / 2**20:.0f} MiB"
    )


if __name__ == "__main__":
    main()
