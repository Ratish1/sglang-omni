"""HiFT's decode graph (real checkpoint) with one kernel's tile changed at a time: replay time against the as built
tiles and the output difference, to split hift_kernel_tiles_c20261018.py's combined arm by kernel.
usage: PYTHONPATH=<tree> python3 hift_decode_arms_c20261018.py <out_dir>"""

import json
import math
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from huggingface_hub import snapshot_download
from sglang.kernels.ops.diffusion.common.numerics import div_rn_f32
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from sglang_omni.models.fun_cosyvoice3 import hift_kernels, stages

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
CELLS = [(rows, frames) for frames in (64, 128, 256) for rows in (1, 4, 16)]
DEVICE = torch.device("cuda")
# (snake tile, stft tile, istft tile, IEEE division)
ARMS = {
    "as built": ((1024, 4), (128, 4), (256, 4), False),
    "snake 512/4w": ((512, 4), (128, 4), (256, 4), False),
    "stft 64/8w": ((1024, 4), (64, 8), (256, 4), False),
    "stft 64/4w": ((1024, 4), (64, 4), (256, 4), False),
    "istft 64/2w": ((1024, 4), (128, 4), (64, 2), False),
    "istft 64/2w div_rn": ((1024, 4), (128, 4), (64, 2), True),
    "snake 512/4w istft 64/2w div_rn": ((512, 4), (128, 4), (64, 2), True),
}
LINES: list[str] = []


def say(line: str) -> None:
    print(line, flush=True)
    LINES.append(line)


@triton.jit(do_not_specialize=["frames", "samples"])
def istft_div_rn_kernel(
    real_ptr, imag_ptr, basis_ptr, squared_window_ptr, out_ptr, frames, samples,
    N_FFT: tl.constexpr, HOP: tl.constexpr, BINS: tl.constexpr, BINS_BLOCK: tl.constexpr,
    BLOCK_SAMPLES: tl.constexpr,
):
    row = tl.program_id(1).to(tl.int64)
    output = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
    is_output = output < samples
    position = output + N_FFT // 2
    bins = tl.arange(0, BINS_BLOCK)
    is_bin = bins < BINS
    signal = tl.zeros((BLOCK_SAMPLES,), dtype=tl.float32)
    envelope = tl.zeros((BLOCK_SAMPLES,), dtype=tl.float32)
    for overlap in tl.static_range(N_FFT // HOP):
        frame = position // HOP - overlap
        tap = position - frame * HOP
        is_frame = is_output & (frame >= 0) & (frame < frames)
        mask = is_frame[:, None] & is_bin[None, :]
        offsets = (row * BINS + bins[None, :]) * frames + frame[:, None]
        real = tl.load(real_ptr + offsets, mask=mask, other=0.0)
        imag = tl.load(imag_ptr + offsets, mask=mask, other=0.0)
        cosine = tl.load(basis_ptr + bins[None, :] * N_FFT + tap[:, None], mask=mask, other=0.0)
        sine = tl.load(basis_ptr + (BINS + bins[None, :]) * N_FFT + tap[:, None], mask=mask, other=0.0)
        value = tl.sum(real * cosine + imag * sine, axis=1)
        signal += tl.where(is_frame, value, 0.0)
        envelope += tl.load(squared_window_ptr + tap, mask=is_frame, other=0.0)
    tl.store(out_ptr + row * samples + output, div_rn_f32(signal, envelope), mask=is_output)


def direct_snake(x, alpha, inverse_alpha, tile):
    batch, channels, samples = x.shape
    out = torch.empty_like(x)
    grid = (triton.cdiv(samples, tile[0]), batch * channels)
    hift_kernels.snake_kernel[grid](x, alpha, inverse_alpha, out, samples, channels, BLOCK_SAMPLES=tile[0],
                                    num_warps=tile[1])
    return out


def direct_stft(padded_signal, basis, hop_length, tile):
    rows, padded_samples = padded_signal.shape
    components, window = basis.shape
    frames = (padded_samples - window) // hop_length + 1
    spectrum = padded_signal.new_empty(rows, components, frames)
    grid = (triton.cdiv(frames, tile[0]), rows)
    hift_kernels.stft_kernel[grid](padded_signal, basis, spectrum, frames, padded_samples, N_FFT=window,
                                   HOP=hop_length, BINS=components // 2, BLOCK_FRAMES=tile[0], num_warps=tile[1])
    return spectrum


def direct_istft(real, imag, basis, squared_window, hop_length, tile, div_rn):
    rows, bin_count, frames = real.shape
    components, window = basis.shape
    samples = hop_length * (frames - 1)
    out = real.new_empty(rows, samples)
    grid = (triton.cdiv(samples, tile[0]), rows)
    kernel = istft_div_rn_kernel if div_rn else hift_kernels.istft_kernel
    kernel[grid](real, imag, basis, squared_window, out, frames, samples, N_FFT=window, HOP=hop_length,
                 BINS=bin_count, BINS_BLOCK=triton.next_power_of_2(bin_count), BLOCK_SAMPLES=tile[0],
                 num_warps=tile[1])
    return out


def main() -> None:
    torch.manual_seed(0)
    say(f"torch {torch.__version__} triton {triton.__version__} {torch.cuda.get_device_name()}")
    checkpoint_dir = snapshot_download(MODEL, local_files_only=True)
    _, hift = stages.load_cosyvoice3_flow_hift_lightweight(checkpoint_dir, device="cuda")
    stages.keep_hift_constants_on_device(hift, "cuda")
    stages.patch_causal_conv_cache()
    for submodule in list(hift.modules()):
        while is_parametrized(submodule):
            name = next(iter(submodule.parametrizations.keys()))
            remove_parametrizations(submodule, name, leave_parametrized=True)
    stages.patch_hift_snake(hift)
    stages.patch_hift_spectra(hift, DEVICE)
    samples_per_frame = int(math.prod(hift.upsample_rates) * hift.istft_params["hop_len"])
    real_ops = (stages.snake, stages.short_time_spectrum, stages.overlap_add_inverse)
    report = {}
    with torch.inference_mode():
        for rows, frames in CELLS:
            mel = torch.randn(rows, 80, frames, device=DEVICE) * 2 - 6
            source = torch.randn(rows, 1, frames * samples_per_frame, device=DEVICE) * 0.1
            graphs = {}
            for arm, (snake_tile, stft_tile, istft_tile, div_rn) in ARMS.items():
                stages.snake = lambda a, b, c, t=snake_tile: direct_snake(a, b, c, t)
                stages.short_time_spectrum = lambda a, b, c, t=stft_tile: direct_stft(a, b, c, t)
                stages.overlap_add_inverse = (
                    lambda a, b, c, d, e, t=istft_tile, rn=div_rn: direct_istft(a, b, c, d, e, t, rn))
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        hift.decode(x=mel, s=source, finalize=True)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = hift.decode(x=mel, s=source, finalize=True)
                graphs[arm] = (graph, output)
            stages.snake, stages.short_time_spectrum, stages.overlap_add_inverse = real_ops
            times = {arm: [] for arm in ARMS}
            for _ in range(9):
                for arm, (graph, _) in graphs.items():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times[arm].append(start.elapsed_time(end) / 20)
            for graph, _ in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            reference = graphs["as built"][1].float()
            cell = {}
            for arm, (_, output) in graphs.items():
                difference = (output.float() - reference).abs()
                cell[arm] = {
                    "replay_ms": statistics.median(times[arm]),
                    "replay_spread": (max(times[arm]) - min(times[arm])) / statistics.median(times[arm]),
                    "max_abs_difference": difference.max().item(),
                    "int16_differs": ((output.float() * 32767).round() != (reference * 32767).round())
                    .float().mean().item(),
                }
            report[f"rows={rows} F={frames}"] = cell
            base = cell["as built"]["replay_ms"]
            say(f"rows={rows} F={frames}: as built {base:.3f} ms (spread {cell['as built']['replay_spread']:.2%})")
            for arm, values in cell.items():
                if arm != "as built":
                    say(f"  {arm}: {values['replay_ms'] / base - 1:+.2%} max|d| {values['max_abs_difference']:.2e} "
                        f"int16 differs {values['int16_differs']:.2e}")
                else:
                    pass
            del graphs
            torch.cuda.empty_cache()
    for arm in ARMS:
        ratios = [report[f"rows={r} F={f}"][arm]["replay_ms"] / report[f"rows={r} F={f}"]["as built"]["replay_ms"]
                  for r, f in CELLS]
        say(f"geomean {arm}: {math.exp(sum(math.log(x) for x in ratios) / len(ratios)) - 1:+.2%}")
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    (OUT / "summary.txt").write_text("\n".join(LINES) + "\n")
    say("done")


if __name__ == "__main__":
    main()
