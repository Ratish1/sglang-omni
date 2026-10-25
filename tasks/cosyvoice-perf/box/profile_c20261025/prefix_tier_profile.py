"""#3132 review F1 (10-25): what a prefix graph replay costs per frame tier, on the trained 22 layer DiT.

One runner captures the union of the tier designs below over one pool and graph pool, at the serving page table
width. A measurement loads a step through the runner's own run() at a chosen tier, then times that tier's graph with
CUDA events on the vocoder stream: the median of 9 samples of 3 replays.

Parts (--parts):
  kernels  2 rows x 200 new frames (no prefix) at tiers 400, 416, 448 and 450, and 2 rows x 208 at 416: device time
           by kernel over 5 replays (torch profiler), the top kernels and the largest differences against tier 400.
  padding  fixed steps (1 x 100, 2 x 200 and 8 x 200 new frames, no prefix) at every captured tier from the step's
           frames to 30 % above them; whether each tier's output equals the step's smallest tier's.
  full     every captured tier filled exactly by rows of at most 200 new frames, no prefix.
  designs  with --census: per census cell, the step frames modulo the shortest hop, the rows, and for up to --sample
           steps each design's tier and replay time summed (rows share one block list, so a prefix costs its real
           reads), plus each design's padded share over every step of the cell.

Designs: l4 (row_tiers(16) whole longest hops), l6 (the prefill ladder over frames, #3132), union (l4 and l6, the
review's candidate), q50 (every multiple of the shortest hop), qspec (the shortest hop times SGLang's speculative
decode batch ladder, cuda_graph_hook.py:481-487), qdecode (times its normal decode ladder, :473-478).

usage: python3 prefix_tier_profile.py --out DIR [--parts kernels,padding,full] [--census FILE --sample N]
"""

import argparse
import bisect
import collections
import json
import random
import statistics
from pathlib import Path

import torch
from sglang.srt.arg_groups.cuda_graph_hook import generate_prefill_cuda_graph_batch_sizes
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.fun_cosyvoice3.prefix_cache import (
    BLOCK_FRAMES,
    PrefixCacheRow,
    PrefixKVPool,
    PrefixStepLayout,
    compile_forward_prefix,
    grow_rows,
    release_rows,
    solve_flow_euler_prefix,
)
from sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph import PrefixCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.solve_graph_capture import SolveGraphCapture, row_tiers
from sglang_omni.models.fun_cosyvoice3.stages import create_vocoder_executor
from sglang_omni.platforms import current_platform

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MAX_ROWS = 16
SHORTEST_HOP = 50
LONGEST_HOP = 200
TOP = MAX_ROWS * LONGEST_HOP
POOL_FRAMES = 8192


def spec_ladder(top):
    return sorted(
        {q for q in [*range(1, 9), *range(10, 33, 2), *range(40, 65, 4), *range(72, 257, 8)] if q <= top} | {top}
    )


def decode_ladder(top):
    return sorted({q for q in [1, 2, 4, 8, 12, *range(16, 257, 8)] if q <= top} | {top})


def designs():
    l4 = [rows * LONGEST_HOP for rows in row_tiers(MAX_ROWS)]
    l6 = [f for f in generate_prefill_cuda_graph_batch_sizes(TOP - 1) if f >= SHORTEST_HOP] + [TOP]
    quanta = TOP // SHORTEST_HOP
    return {
        "l4": l4,
        "l6": l6,
        "union": sorted(set(l4) | set(l6)),
        "q50": [SHORTEST_HOP * q for q in range(1, quanta + 1)],
        "qspec": [SHORTEST_HOP * q for q in spec_ladder(quanta)],
        "qdecode": [SHORTEST_HOP * q for q in decode_ladder(quanta)],
    }


class TierRunner(PrefixCudaGraphRunner):
    """The runner with an explicit tier list, each tier's layout as the runner derives it."""

    def __init__(self, *args, tier_frames, **kwargs):
        super().__init__(*args, **kwargs)
        chunk_size = self.estimator.chunk_size
        self.tier_frames = list(tier_frames)
        self.layouts = []
        for frames in self.tier_frames:
            row_slots = min(kwargs["max_rows"], frames // kwargs["min_hop_frames"])
            self.layouts.append(
                PrefixStepLayout(
                    half_frames=frames,
                    row_slots=row_slots,
                    segment_count=2 * (-(-frames // chunk_size) + row_slots + 1),
                    page_table_width=kwargs["max_frames"],
                )
            )


class Bench:
    def __init__(self, runner, streams, time_span, mel_channels, speaker_channels, device):
        self.runner = runner
        self.streams = streams
        self.time_span = time_span
        self.mel_channels = mel_channels
        self.speaker_channels = speaker_channels
        self.device = device
        self.tiers = list(runner.tier_frames)
        self.captured = list(runner.captured)

    def load(self, tier, new_frames, prefix_frames):
        """The step through run() at this tier; returns the tier's graph and run()'s output."""
        frames = sum(new_frames)
        assert tier >= frames, (tier, frames)
        widest = max(prefix + new for prefix, new in zip(prefix_frames, new_frames))
        blocks = list(range(-(-widest // BLOCK_FRAMES)))
        caches = [
            (PrefixCacheRow(blocks=blocks, committed_frames=prefix), PrefixCacheRow(blocks=blocks, committed_frames=prefix))
            for prefix in prefix_frames
        ]
        inputs = {
            field: torch.cat(
                [self.streams[index % len(self.streams)][field][:new] for index, new in enumerate(new_frames)]
            ).unsqueeze(0)
            for field in ("noise", "mu", "mel_conditioning")
        }
        speakers = torch.stack(
            [self.streams[index % len(self.streams)]["speaker_embeddings"] for index in range(len(new_frames))]
        )
        index = self.tiers.index(tier)
        self.runner.tier_frames, self.runner.captured = [tier], [self.captured[index]]
        try:
            output = self.runner.run(
                **inputs, speaker_embeddings=speakers, time_span=self.time_span, new_frames=list(new_frames),
                caches=caches,
            )
        finally:
            self.runner.tier_frames, self.runner.captured = self.tiers, self.captured
        assert output is not None
        return self.captured[index].graph, output

    def time(self, graph, samples=9, replays=3):
        values = []
        for _ in range(samples):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end) / replays)
        return statistics.median(values)

    def step_ms(self, tier, new_frames, prefix_frames, samples=9, replays=3):
        graph, _ = self.load(tier, new_frames, prefix_frames)
        return self.time(graph, samples, replays)


def kernel_times(graph, replays=5):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    times = collections.Counter()
    counts = collections.Counter()
    for event in prof.key_averages():
        total = event.device_time_total
        if total > 0:
            times[event.key] += total / 1000 / replays
            counts[event.key] += event.count // replays
    return times, counts


def part_kernels(bench, log):
    cases = [("2x200@400", 400, [200, 200]), ("2x200@416", 416, [200, 200]), ("2x200@448", 448, [200, 200]),
             ("2x200@450", 450, [200, 200]), ("2x208@416", 416, [208, 208])]
    results = {}
    for name, tier, new in cases:
        graph, _ = bench.load(tier, new, [0] * len(new))
        times, counts = kernel_times(graph)
        results[name] = (times, counts, bench.time(graph))
        log(f"kernels {name}: replay {results[name][2]:.3f} ms, kernels {sum(counts.values())} per replay, "
            f"device {sum(times.values()):.3f} ms per replay")
    base_times = results["2x200@400"][0]
    for name, (times, counts, _) in results.items():
        log(f"  {name} top kernels (ms per replay, count):")
        for key, value in times.most_common(8):
            log(f"    {value:8.3f} {counts[key]:5d}  {key[:110]}")
        if name != "2x200@400":
            deltas = sorted(((times[key] - base_times[key], key) for key in set(times) | set(base_times)), reverse=True)
            log(f"  {name} minus 2x200@400, largest increases (ms per replay):")
            for delta, key in deltas[:6]:
                log(f"    {delta:+8.3f}  {times[key]:8.3f} vs {base_times[key]:8.3f}  {key[:100]}")
        else:
            pass


def part_padding(bench, log):
    for new in ([100], [200, 200], [200] * 8):
        frames = sum(new)
        tiers = [tier for tier in bench.tiers if frames <= tier <= frames * 1.3]
        reference = None
        line = []
        for tier in tiers:
            graph, output = bench.load(tier, new, [0] * len(new))
            equal = "" if reference is None else ("=" if torch.equal(reference, output) else "!=")
            reference = output if reference is None else reference
            line.append(f"{tier}:{bench.time(graph):.3f}{equal}")
        log(f"padding {len(new)}x{new[0]} ({frames} frames): " + "  ".join(line))


def part_full(bench, log):
    rows_out = []
    for tier in bench.tiers:
        row_count = -(-tier // LONGEST_HOP)
        new = [tier // row_count + (1 if index < tier % row_count else 0) for index in range(row_count)]
        rows_out.append(f"{tier}:{bench.step_ms(tier, new, [0] * row_count):.3f}")
    log("full (tier:ms, rows of at most 200 frames filling the tier): " + "  ".join(rows_out))


def part_designs(bench, log, census_path, sample):
    cells = collections.defaultdict(list)
    cell = None
    for line in Path(census_path).read_text().splitlines():
        record = json.loads(line)
        if "cell" in record:
            cell = record["cell"]
        elif cell is not None:
            cells[cell].append(record)
        else:
            pass
    tier_designs = designs()
    for cell, all_steps in cells.items():
        steps = [step for step in all_steps if step["frames"] <= TOP]
        frames = [step["frames"] for step in steps]
        off_quantum = sum(1 for value in frames if value % SHORTEST_HOP)
        rows = collections.Counter(len(step["new_frames"]) for step in steps)
        frame_counts = collections.Counter(frames)
        log(f"designs {cell}: {len(all_steps)} steps, {len(all_steps) - len(steps)} above {TOP} frames (eager), "
            f"{off_quantum} not a multiple of {SHORTEST_HOP} frames, rows {dict(sorted(rows.items()))}, mean frames "
            f"{statistics.mean(frames):.0f}, commonest frames {frame_counts.most_common(12)}")
        for name, tiers in tier_designs.items():
            picks = [tiers[bisect.bisect_left(tiers, value)] for value in frames]
            log(f"  {name}: {len(tiers)} graphs, padded share {1 - sum(frames) / sum(picks):.1%}")
        fitting = [step for step in steps if max(p + n for p, n in zip(step["prefix_frames"], step["new_frames"])) <= POOL_FRAMES]
        chosen = random.Random(0).sample(fitting, min(sample, len(fitting)))
        totals = collections.Counter()
        for step in chosen:
            needed = {name: tiers[bisect.bisect_left(tiers, step["frames"])] for name, tiers in tier_designs.items()}
            measured = {tier: bench.step_ms(tier, step["new_frames"], step["prefix_frames"], samples=3, replays=2)
                        for tier in set(needed.values())}
            for name, tier in needed.items():
                totals[name] += measured[tier]
        log(f"  priced on {len(chosen)} sampled steps ({len(steps) - len(fitting)} skipped above {POOL_FRAMES} "
            f"frames a row), replay ms summed:")
        for name in tier_designs:
            log(f"    {name:8s} {totals[name]:9.1f} ms  ({totals[name] / totals['l6'] - 1:+.2%} against l6)")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parts", default="kernels,padding,full")
    parser.add_argument("--census")
    parser.add_argument("--sample", type=int, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = open(args.out / "report.txt", "a")

    def log(text):
        print(text, flush=True)
        report.write(text + "\n")
        report.flush()

    torch.manual_seed(3132)
    scheduler = create_vocoder_executor(
        MODEL, device="cuda", gpu_id=0, dtype="bfloat16", max_batch_size=MAX_ROWS, flow_prefix_cache_gb=0,
        enable_dit_torch_compile=False, enable_flow_cuda_graph=False, enable_flow_prefix_cuda_graph=False,
        enable_flow_whole_history_cuda_graph=False, enable_hift_cuda_graph=False,
    )
    vocoder = scheduler.vocoder
    estimator = vocoder.flow.packed_estimator
    assert estimator.is_ragged
    device = torch.device("cuda:0")
    attention = estimator.dit.transformer_blocks[0].attn
    dtype = torch.bfloat16
    pool = PrefixKVPool(
        layer_num=len(estimator.dit.transformer_blocks), euler_steps=10, head_num=attention.heads,
        head_dim=attention.inner_dim // attention.heads, capacity_frames=POOL_FRAMES, device=device, dtype=dtype,
    )
    pool.forward = compile_forward_prefix()
    mel_channels = vocoder.flow.output_size
    speaker_channels = vocoder.flow.spk_embed_affine_layer.out_features
    max_frames = vocoder.flow.decoder.rand_noise.shape[2]
    unit = torch.linspace(0, 1, 11, device=device, dtype=dtype)
    time_span = (1 - torch.cos(unit * 0.5 * torch.pi)).float()
    capture_tiers = sorted(set().union(*designs().values()))
    log(f"device {torch.cuda.get_device_name()}, chunk_size {estimator.chunk_size}, page table width {max_frames}, "
        f"capture tiers {len(capture_tiers)}")
    with vocoder.stream_context, torch.autocast("cuda", dtype=dtype):
        estimator.build_modulations(time_span, dtype)
        for lengths in ((50,), (50, 70)):
            pairs = [(PrefixCacheRow(), PrefixCacheRow()) for _ in lengths]
            for pair, length in zip(pairs, lengths):
                assert grow_rows(pool, list(pair), [length, length])
            solve_flow_euler_prefix(
                estimator, pool, torch.randn(1, sum(lengths), mel_channels, device=device), time_span,
                torch.randn(1, sum(lengths), mel_channels, device=device),
                torch.randn(len(lengths), speaker_channels, device=device, dtype=dtype),
                torch.randn(1, sum(lengths), mel_channels, device=device), list(lengths), pairs, cfg_rate=0.7,
            )
            for pair in pairs:
                release_rows(pool, list(pair))
        runner = TierRunner(
            estimator, pool,
            graphs=SolveGraphCapture(current_platform.get_device_graph_backend(device), device=device,
                                     autocast_dtype=dtype),
            frame_dtype=torch.float32, speaker_dtype=dtype, cfg_rate=0.7, mel_channels=mel_channels,
            speaker_channels=speaker_channels, max_rows=MAX_ROWS, hop_frames=LONGEST_HOP,
            min_hop_frames=SHORTEST_HOP, max_frames=max_frames, tier_frames=capture_tiers,
        )
        runner.capture()
        torch.cuda.synchronize()
        streams = [{
            "noise": torch.randn(TOP, mel_channels, device=device),
            "mu": torch.randn(TOP, mel_channels, device=device),
            "mel_conditioning": torch.randn(TOP, mel_channels, device=device) * 0.1,
            "speaker_embeddings": torch.randn(speaker_channels, device=device, dtype=dtype),
        } for _ in range(MAX_ROWS)]
        bench = Bench(runner, streams, time_span, mel_channels, speaker_channels, device)
        for part in args.parts.split(","):
            if part == "kernels":
                part_kernels(bench, log)
            elif part == "padding":
                part_padding(bench, log)
            elif part == "full":
                part_full(bench, log)
            elif part == "designs":
                part_designs(bench, log, args.census, args.sample)
            else:
                raise ValueError(part)
    log("profile done")
    scheduler.stop()


if __name__ == "__main__":
    main()
