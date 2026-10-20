"""HiFT probes for the decode graphs design (DESIGN_HIFT_GRAPHS_20261017.md), on the checkpoint's HiFT as the
vocoder loads it (constants on the device, the causal conv cache patch), run from a tree whose stages.py carries the
spectra and Snake patches (perf/cosyvoice3-hift-spectra-kernels).

1. Launches and time per part of a HiFT group (F0 in float64, the sine source, the decode), stock and with the
   kernels, at the shapes a c16 step makes: CUDA kernels per call, device time and host wall per part.
2. Batch invariance of the decode: one row decoded alone at its own width against the same row padded to wider
   frame tiers and beside other rows at larger row tiers, as a hop (left aligned) and as a final (right aligned),
   compared bit for bit over the samples a step emits from it.
3. The decode captured as a CUDA graph (kernels only: stock torch.istft reads the host): replay against the eager
   call at the same shape, bit for bit, and the host wall of each.
4. The decode's device time per row at 1 to 16 rows, eager and replayed.

Variants: stock or kernels (the spectra and Snake patches), each with cuDNN's TF32 as serving leaves it (on) or off
(suffix _tf32_off).

usage (in the container, one card): PYTHONPATH=<tree> python3 hift_probe_c20261017.py [variant,variant,...]"""

import sys
import time

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
CONTEXT, HOLD = stages.HIFT_DECODE_CONTEXT_FRAMES, 8
ROW_TIERS = (1, 2, 4, 8, 16)
FRAME_TIERS = (64, 96, 128, 160, 256)


def load_hift(with_kernels):
    _, hift = stages.load_cosyvoice3_flow_hift_lightweight(resolve_checkpoint(MODEL), device="cuda")
    stages.keep_hift_constants_on_device(hift, "cuda")
    stages.patch_causal_conv_cache()
    if with_kernels:
        stages.patch_hift_spectra(hift, torch.device("cuda"))
        stages.patch_hift_snake(hift)
    else:
        pass
    return hift


def mel(rows, frames, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(rows, 80, frames, generator=generator) * 2 - 4).cuda()


def f0_part(hift, mels):
    return hift.f0_predictor(mels.double(), finalize=True).to(mels)


def source_part(hift, f0):
    excitation = hift.f0_upsamp(f0[:, None])
    excitation, _, _ = hift.m_source(excitation.transpose(1, 2))
    return excitation.transpose(1, 2)


def decode_part(hift, mels, source):
    return hift.decode(x=mels, s=source, finalize=True)


def kernels_and_time(fn, calls=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    torch.cuda.synchronize()
    wall_ms = 1e3 * (time.perf_counter() - start) / calls
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    kernels = [event for event in prof.events() if event.device_type == DeviceType.CUDA]
    return len(kernels) / calls, sum(event.device_time for event in kernels) / calls / 1e3, wall_ms


def part_table(hift, name):
    print(f"\n## 1. parts of a HiFT group, {name}")
    print("| rows | frames | part | CUDA kernels a call | device ms | wall ms |")
    print("| --- | --- | --- | --- | --- | --- |")
    for rows, frames in ((1, 87), (8, 87), (16, 87), (16, 241)):
        mels = mel(rows, frames, 0)
        f0 = f0_part(hift, mels)
        source = source_part(hift, f0)
        for part, fn in (("F0", lambda: f0_part(hift, mels)), ("source", lambda: source_part(hift, f0)),
                         ("decode", lambda: decode_part(hift, mels, source))):
            count, device_ms, wall_ms = kernels_and_time(fn)
            print(f"| {rows} | {frames} | {part} | {count:.0f} | {device_ms:.2f} | {wall_ms:.2f} |")


def placed(row_mel, row_source, rows, frames, final, seed):
    """The row at index 0 of a batch of `rows` rows `frames` wide, left aligned as a hop or right aligned as a
    final, the other rows random."""
    width = row_mel.shape[2]
    mels = mel(rows, frames, seed)
    sources = torch.randn(rows, 1, frames * 480, device="cuda") * 0.1
    offset = frames - width if final else 0
    mels[0].zero_()
    sources[0].zero_()
    mels[0, :, offset : offset + width] = row_mel[0]
    sources[0, :, offset * 480 : (offset + width) * 480] = row_source[0]
    return mels, sources, offset


def invariance_table(hift, name):
    print(f"\n## 2. batch invariance of the decode, {name}")
    print("| as | row frames | rows tier | frames tier | emitted samples bit equal | max abs diff |")
    print("| --- | --- | --- | --- | --- | --- |")
    for final in (False, True):
        for width in (50, 87):
            row_mel = mel(1, width, 7)
            row_source = source_part(hift, f0_part(hift, row_mel))
            alone = decode_part(hift, row_mel, row_source)
            first, last = (CONTEXT if final else 0), (width if final else width - HOLD)
            reference = alone[0, first * 480 : last * 480]
            for rows in ROW_TIERS:
                for frames in FRAME_TIERS:
                    if frames < width:
                        continue
                    mels, sources, offset = placed(row_mel, row_source, rows, frames, final, rows * 1000 + frames)
                    out = decode_part(hift, mels, sources)[0, (offset + first) * 480 : (offset + last) * 480]
                    diff = (out - reference).abs().max().item()
                    print(f"| {'final' if final else 'hop'} | {width} | {rows} | {frames} | "
                          f"{bool(torch.equal(out, reference))} | {diff:.3g} |")


def graph_table(hift):
    print("\n## 3 and 4. the decode as a CUDA graph (kernels)")
    print("| rows | frames | replay bit equal to eager | eager wall ms | replay wall ms | eager device ms per row "
          "| replay device ms per row |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream()
    # every graph that shares the pool stays alive, as a runner keeps its tiers
    graphs = []
    for frames in (96, 128):
        for rows in ROW_TIERS:
            mels = mel(rows, frames, 3)
            sources = source_part(hift, f0_part(hift, mels))
            static_mels, static_sources = mels.clone(), sources.clone()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    decode_part(hift, static_mels, static_sources)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                static_out = decode_part(hift, static_mels, static_sources)
            graphs.append(graph)
            graph.replay()
            eager = decode_part(hift, mels, sources)
            torch.cuda.synchronize()
            equal = bool(torch.equal(static_out, eager))
            _, eager_device, eager_wall = kernels_and_time(lambda: decode_part(hift, mels, sources))
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            begin.record()
            for _ in range(20):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            replay_wall = 1e3 * (time.perf_counter() - start) / 20
            replay_device = begin.elapsed_time(end) / 20
            print(f"| {rows} | {frames} | {equal} | {eager_wall:.2f} | {replay_wall:.2f} | "
                  f"{eager_device / rows:.2f} | {replay_device / rows:.2f} |")


with torch.inference_mode():
    print(f"TF32 as in serving's defaults: cudnn {torch.backends.cudnn.allow_tf32}, "
          f"matmul {torch.backends.cuda.matmul.allow_tf32}, cudnn benchmark {torch.backends.cudnn.benchmark}")
    variants = sys.argv[1].split(",") if len(sys.argv) > 1 else ["stock", "kernels", "kernels_tf32_off"]
    for variant in variants:
        hift = load_hift(with_kernels=variant.startswith("kernels"))
        torch.backends.cudnn.allow_tf32 = not variant.endswith("tf32_off")
        name = f"{variant} (cudnn TF32 {torch.backends.cudnn.allow_tf32})"
        part_table(hift, name)
        invariance_table(hift, name)
        if variant.startswith("kernels"):
            graph_table(hift)
        else:
            pass
        del hift
