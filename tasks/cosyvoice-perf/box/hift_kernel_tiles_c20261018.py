"""HiFT's Snake, STFT and inverse STFT kernels at the decode's own shapes (real checkpoint): device time per decode
for each tile and warp count, registers and spills, the instructions behind Snake's products and the inverse's
division, the dispatcher's host cost per eager call, the eager HiFT call with and without it, and the decode graph's
replay time and output at the current tiles against the fastest.
usage: PYTHONPATH=<tree> python3 hift_kernel_tiles_c20261018.py <out_dir>"""

import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import triton
import triton.language as tl
from huggingface_hub import snapshot_download
from sglang.kernels.ops.diffusion.common.numerics import div_rn_f32, mul_rn_f32
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
from triton.language.extra import libdevice

from sglang_omni.models.fun_cosyvoice3 import hift_kernels, stages

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
ROWS = (1, 4, 16)
FRAMES = (64, 128, 256)
EAGER_FRAMES = 400
SNAKE_TILES = [(256, 4), (512, 4), (1024, 4), (1024, 8), (2048, 4), (2048, 8)]
STFT_TILES = [(block, warps) for block in (32, 64, 128, 256) for warps in (2, 4, 8)]
ISTFT_TILES = [(block, warps) for block in (64, 128, 256, 512) for warps in (2, 4, 8)]
CURRENT = {"snake": (1024, 4), "stft": (128, 4), "istft": (256, 4)}
DEVICE = torch.device("cuda")
REPORT: dict = {}
LINES: list[str] = []


def say(line: str) -> None:
    print(line, flush=True)
    LINES.append(line)


@triton.jit(do_not_specialize=["samples", "channels"])
def snake_rows_first_kernel(
    x_ptr, alpha_ptr, inverse_alpha_ptr, out_ptr, samples, channels, BLOCK_SAMPLES: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.program_id(1) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
    is_sample = offsets < samples
    alpha = tl.load(alpha_ptr + row % channels)
    inverse_alpha = tl.load(inverse_alpha_ptr + row % channels)
    x = tl.load(x_ptr + row * samples + offsets, mask=is_sample)
    sine = libdevice.sin(mul_rn_f32(x, alpha))
    out = x + mul_rn_f32(inverse_alpha, mul_rn_f32(sine, sine))
    tl.store(out_ptr + row * samples + offsets, out, mask=is_sample)


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


def device_ms(launch, launches: int = 20, replays: int = 10, trials: int = 5) -> float:
    """Median device time of one launch, from a graph of back to back launches."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(launches):
            launch()
    per_launch = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        per_launch.append(start.elapsed_time(end) / (replays * launches))
    del graph
    return statistics.median(per_launch)


def host_us(call, calls: int = 20000) -> float:
    for _ in range(200):
        call()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / calls * 1e6


def build_hift() -> torch.nn.Module:
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
    return hift


def instruction_counts(ptx: str) -> dict[str, int]:
    names = ["mul.rn.f32", "mul.f32", "fma.rn.f32", "add.f32", "add.rn.f32", "div.full.f32", "div.rn.f32",
             "div.approx.f32", "rcp.approx", ".ftz"]
    return {name: len(re.findall(re.escape(name), ptx)) for name in names}


def main() -> None:
    torch.manual_seed(0)
    say(f"torch {torch.__version__} triton {triton.__version__} {torch.cuda.get_device_name()}")
    say(f"hift_kernels {hift_kernels.__file__}")
    hift = build_hift()
    n_fft = hift.istft_params["n_fft"]
    hop = hift.istft_params["hop_len"]
    bins = n_fft // 2 + 1
    samples_per_frame = int(math.prod(hift.upsample_rates) * hop)
    squared_window = hift.stft_window.to(DEVICE).pow(2)

    # the kernels' call shapes in one decode per frame tier, batch 1
    calls: list[tuple[str, tuple[int, ...]]] = []
    real_ops = (stages.snake, stages.short_time_spectrum, stages.overlap_add_inverse)

    def recording_snake(x, alpha, inverse_alpha):
        calls.append(("snake", tuple(x.shape)))
        return real_ops[0](x, alpha, inverse_alpha)

    def recording_stft(padded_signal, basis, hop_length):
        calls.append(("stft", tuple(padded_signal.shape)))
        return real_ops[1](padded_signal, basis, hop_length)

    def recording_istft(real, imag, basis, squared_window, hop_length):
        calls.append(("istft", tuple(real.shape)))
        return real_ops[2](real, imag, basis, squared_window, hop_length)

    stages.snake, stages.short_time_spectrum, stages.overlap_add_inverse = (
        recording_snake, recording_stft, recording_istft)
    shapes: dict[int, dict[str, Counter]] = {}
    with torch.inference_mode():
        for frames in (*FRAMES, EAGER_FRAMES):
            calls.clear()
            mel = torch.randn(1, 80, frames, device=DEVICE) * 2 - 6
            source = torch.randn(1, 1, frames * samples_per_frame, device=DEVICE) * 0.1
            hift.decode(x=mel, s=source, finalize=True)
            shapes[frames] = {name: Counter(shape for kind, shape in calls if kind == name)
                              for name in ("snake", "stft", "istft")}
            say(f"shapes F={frames}: " + "; ".join(
                f"{name} " + ", ".join(f"{shape}x{count}" for shape, count in counter.items())
                for name, counter in shapes[frames].items()))
    stages.snake, stages.short_time_spectrum, stages.overlap_add_inverse = real_ops
    REPORT["shapes"] = {str(f): {k: {str(s): c for s, c in v.items()} for k, v in d.items()}
                        for f, d in shapes.items()}

    # launchers over preallocated tensors
    def snake_inputs(shape):
        batch, channels, samples = shape
        x = torch.randn(shape, device=DEVICE) * 3
        alpha = torch.rand(channels, device=DEVICE) + 0.5
        return x, alpha, 1.0 / (alpha + 1e-9), torch.empty_like(x)

    def snake_launcher(shape, tensors, tile, rows_first):
        batch, channels, samples = shape
        block, warps = tile
        x, alpha, inverse_alpha, out = tensors
        if rows_first:
            kernel, grid = snake_rows_first_kernel, (batch * channels, triton.cdiv(samples, block))
        else:
            kernel, grid = hift_kernels.snake_kernel, (triton.cdiv(samples, block), batch * channels)
        return lambda: kernel[grid](x, alpha, inverse_alpha, out, samples, channels,
                                    BLOCK_SAMPLES=block, num_warps=warps)

    def stft_inputs(shape):
        rows, padded_samples = shape
        frames = (padded_samples - n_fft) // hop + 1
        return (torch.randn(shape, device=DEVICE), torch.randn(2 * bins, n_fft, device=DEVICE),
                torch.empty(rows, 2 * bins, frames, device=DEVICE))

    def stft_launcher(shape, tensors, tile, _):
        rows, padded_samples = shape
        frames = (padded_samples - n_fft) // hop + 1
        block, warps = tile
        padded, basis, spectrum = tensors
        grid = (triton.cdiv(frames, block), rows)
        return lambda: hift_kernels.stft_kernel[grid](
            padded, basis, spectrum, frames, padded_samples, N_FFT=n_fft, HOP=hop, BINS=bins,
            BLOCK_FRAMES=block, num_warps=warps)

    def istft_inputs(shape):
        rows, _, frames = shape
        return (torch.randn(shape, device=DEVICE), torch.randn(shape, device=DEVICE),
                torch.randn(2 * bins, n_fft, device=DEVICE), torch.empty(rows, hop * (frames - 1), device=DEVICE))

    def istft_launcher(shape, tensors, tile, div_rn):
        rows, _, frames = shape
        samples = hop * (frames - 1)
        block, warps = tile
        real, imag, basis, out = tensors
        kernel = istft_div_rn_kernel if div_rn else hift_kernels.istft_kernel
        grid = (triton.cdiv(samples, block), rows)
        return lambda: kernel[grid](
            real, imag, basis, squared_window, out, frames, samples, N_FFT=n_fft, HOP=hop, BINS=bins,
            BINS_BLOCK=triton.next_power_of_2(bins), BLOCK_SAMPLES=block, num_warps=warps)

    # registers, spills and instructions
    compiled_info = {}
    probe = {"snake": (1, 64, 1024), "stft": (1, 1040), "istft": (1, bins, 257)}
    for name, inputs, launcher, tiles, variants in (
        ("snake", snake_inputs, snake_launcher, SNAKE_TILES, (False, True)),
        ("stft", stft_inputs, stft_launcher, STFT_TILES, (False,)),
        ("istft", istft_inputs, istft_launcher, ISTFT_TILES, (False, True)),
    ):
        tensors = inputs(probe[name])
        for variant in variants:
            for tile in tiles:
                compiled = launcher(probe[name], tensors, tile, variant)()
                key = f"{name} {'variant' if variant else 'as built'} {tile[0]}/{tile[1]}w"
                compiled_info[key] = {"regs": compiled.n_regs, "spills": compiled.n_spills}
                if tile == CURRENT[name]:
                    compiled_info[key]["ptx"] = instruction_counts(compiled.asm["ptx"])
                    say(f"{key}: regs {compiled.n_regs} spills {compiled.n_spills} {compiled_info[key]['ptx']}")
                else:
                    pass
    REPORT["compiled"] = compiled_info
    say("regs/spills: " + "; ".join(f"{k} {v['regs']}/{v['spills']}" for k, v in compiled_info.items()))

    # device time per decode per tile
    cells = [(rows, frames) for frames in FRAMES for rows in ROWS] + [(1, EAGER_FRAMES)]
    sweep: dict[str, dict[str, float]] = {}
    for name, inputs, launcher, tiles, variants in (
        ("snake", snake_inputs, snake_launcher, SNAKE_TILES, (False, True)),
        ("stft", stft_inputs, stft_launcher, STFT_TILES, (False,)),
        ("istft", istft_inputs, istft_launcher, ISTFT_TILES, (False, True)),
    ):
        for rows, frames in cells:
            totals: dict[str, float] = {}
            for shape1, count in shapes[frames][name].items():
                shape = (rows, *shape1[1:])
                tensors = inputs(shape)
                for variant in variants:
                    for tile in tiles:
                        key = f"{'variant' if variant else 'as built'} {tile[0]}/{tile[1]}w"
                        totals[key] = totals.get(key, 0.0) + count * device_ms(
                            launcher(shape, tensors, tile, variant))
                del tensors
                torch.cuda.empty_cache()
            sweep[f"{name} rows={rows} F={frames}"] = totals
            current_key = f"as built {CURRENT[name][0]}/{CURRENT[name][1]}w"
            ranked = sorted(totals.items(), key=lambda item: item[1])
            say(f"{name} rows={rows} F={frames}: as built {totals[current_key] * 1000:.1f} us per decode; " +
                ", ".join(f"{k} {v / totals[current_key] - 1:+.1%}" for k, v in ranked[:6]))
    REPORT["sweep"] = sweep

    # the tile with the lowest geometric mean of its time against the as built tile, over every cell
    best = {}
    for name in ("snake", "stft", "istft"):
        current_key = f"as built {CURRENT[name][0]}/{CURRENT[name][1]}w"
        keys = [k for k in sweep[f"{name} rows=1 F=64"] if k.startswith("as built")]
        geomeans = {}
        for key in keys:
            ratios = [sweep[f"{name} rows={r} F={f}"][key] / sweep[f"{name} rows={r} F={f}"][current_key]
                      for r, f in cells]
            geomeans[key] = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        ranked = sorted(geomeans.items(), key=lambda item: item[1])
        say(f"{name} geomean against as built: " + ", ".join(f"{k} {v - 1:+.1%}" for k, v in ranked))
        block, warps = ranked[0][0].removeprefix("as built ").removesuffix("w").split("/")
        best[name] = (int(block), int(warps))
    REPORT["best"] = best
    say(f"best tiles {best}")

    # dispatcher cost per eager call: the registered op against the same launch called directly
    def direct_snake(x, alpha, inverse_alpha, tile=CURRENT["snake"]):
        batch, channels, samples = x.shape
        out = torch.empty_like(x)
        grid = (triton.cdiv(samples, tile[0]), batch * channels)
        hift_kernels.snake_kernel[grid](x, alpha, inverse_alpha, out, samples, channels,
                                        BLOCK_SAMPLES=tile[0], num_warps=tile[1])
        return out

    def direct_stft(padded_signal, basis, hop_length, tile=CURRENT["stft"]):
        rows, padded_samples = padded_signal.shape
        components, window = basis.shape
        frames = (padded_samples - window) // hop_length + 1
        spectrum = padded_signal.new_empty(rows, components, frames)
        grid = (triton.cdiv(frames, tile[0]), rows)
        hift_kernels.stft_kernel[grid](padded_signal, basis, spectrum, frames, padded_samples, N_FFT=window,
                                       HOP=hop_length, BINS=components // 2, BLOCK_FRAMES=tile[0], num_warps=tile[1])
        return spectrum

    def direct_istft(real, imag, basis, squared_window, hop_length, tile=CURRENT["istft"], div_rn=False):
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

    x, alpha, inverse_alpha, _ = snake_inputs((1, 64, 512))
    padded, basis, _ = stft_inputs((1, 1040))
    real, imag, inverse_basis, _ = istft_inputs((1, bins, 257))
    dispatch = {
        "snake registered": host_us(lambda: hift_kernels.snake(x, alpha, inverse_alpha)),
        "snake direct": host_us(lambda: direct_snake(x, alpha, inverse_alpha)),
        "stft registered": host_us(lambda: hift_kernels.short_time_spectrum(padded, basis, hop)),
        "stft direct": host_us(lambda: direct_stft(padded, basis, hop)),
        "istft registered": host_us(
            lambda: hift_kernels.overlap_add_inverse(real, imag, inverse_basis, squared_window, hop)),
        "istft direct": host_us(lambda: direct_istft(real, imag, inverse_basis, squared_window, hop)),
    }
    REPORT["dispatch_us"] = dispatch
    say("host us per call: " + ", ".join(f"{k} {v:.2f}" for k, v in dispatch.items()))

    # the eager HiFT call (the buffered path) with the registered ops against direct launches
    with torch.inference_mode():
        mel = torch.randn(1, 80, EAGER_FRAMES, device=DEVICE) * 2 - 6
        walls = {"registered": [], "direct": []}
        for _ in range(5):
            for arm in ("registered", "direct"):
                if arm == "direct":
                    stages.snake, stages.short_time_spectrum, stages.overlap_add_inverse = (
                        direct_snake, direct_stft, direct_istft)
                else:
                    stages.snake, stages.short_time_spectrum, stages.overlap_add_inverse = real_ops
                for _ in range(3):
                    hift.inference(speech_feat=mel, finalize=True)
                torch.cuda.synchronize()
                for _ in range(10):
                    start = time.perf_counter()
                    hift.inference(speech_feat=mel, finalize=True)
                    torch.cuda.synchronize()
                    walls[arm].append((time.perf_counter() - start) * 1000)
        stages.snake, stages.short_time_spectrum, stages.overlap_add_inverse = real_ops
    eager = {arm: statistics.median(values) for arm, values in walls.items()}
    REPORT["eager_hift_ms"] = eager
    say(f"eager HiFT F={EAGER_FRAMES} wall median ms: registered {eager['registered']:.3f}, direct "
        f"{eager['direct']:.3f} ({eager['direct'] / eager['registered'] - 1:+.2%})")

    # the decode graph per cell: as built tiles, the best tiles, and the as built tiles with the IEEE division
    arms = {
        "as built": (CURRENT["snake"], CURRENT["stft"], CURRENT["istft"], False),
        "best": (best["snake"], best["stft"], best["istft"], False),
        "div_rn": (CURRENT["snake"], CURRENT["stft"], CURRENT["istft"], True),
    }
    decode = {}
    with torch.inference_mode():
        for rows, frames in [(rows, frames) for frames in FRAMES for rows in ROWS]:
            mel = torch.randn(rows, 80, frames, device=DEVICE) * 2 - 6
            source = torch.randn(rows, 1, frames * samples_per_frame, device=DEVICE) * 0.1
            graphs = {}
            for arm, (snake_tile, stft_tile, istft_tile, div_rn) in arms.items():
                stages.snake = lambda a, b, c, t=snake_tile: direct_snake(a, b, c, tile=t)
                stages.short_time_spectrum = lambda a, b, c, t=stft_tile: direct_stft(a, b, c, tile=t)
                stages.overlap_add_inverse = (
                    lambda a, b, c, d, e, t=istft_tile, rn=div_rn: direct_istft(a, b, c, d, e, tile=t, div_rn=rn))
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
            times = {arm: [] for arm in arms}
            for _ in range(7):
                for arm, (graph, _) in graphs.items():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times[arm].append(start.elapsed_time(end) / 20)
            for arm, (graph, _) in graphs.items():
                graph.replay()
            torch.cuda.synchronize()
            reference = graphs["as built"][1].float()
            cell = {}
            for arm, (graph, output) in graphs.items():
                difference = (output.float() - reference).abs()
                int16_differs = ((output.float() * 32767).round() != (reference * 32767).round()).float().mean()
                cell[arm] = {"replay_ms": statistics.median(times[arm]),
                             "max_abs_difference": difference.max().item(),
                             "int16_differs": int16_differs.item()}
            decode[f"rows={rows} F={frames}"] = cell
            base = cell["as built"]["replay_ms"]
            say(f"decode rows={rows} F={frames}: as built {base:.3f} ms; " + ", ".join(
                f"{arm} {v['replay_ms'] / base - 1:+.2%} max|d| {v['max_abs_difference']:.2e} "
                f"int16 differs {v['int16_differs']:.2e}" for arm, v in cell.items() if arm != "as built"))
            del graphs
            torch.cuda.empty_cache()
    REPORT["decode"] = decode
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=1))
    (OUT / "summary.txt").write_text("\n".join(LINES) + "\n")
    say("done")


if __name__ == "__main__":
    main()
