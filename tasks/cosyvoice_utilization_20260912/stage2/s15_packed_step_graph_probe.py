"""One Euler step of the plain packed Flow replayed from a breakable CUDA graph.

The step graph of the hop cache branch reaches cached hops only. This probe
captures the same step over main's PackedDiT, with no K/V cache, so streaming
hops and finals both replay. The step runs at the smallest captured size that
holds the call's packed frames; the inputs are zero padded to it. Two ops stay
eager between the graph's segments because their shapes follow the rows: the
ragged FA3 call and the conv position embedding. cos and sin are graph inputs.

Per shape and call kind it reports eager against replayed (wall, kernels,
kernel time), the mel's distance from eager (max abs diff, SNR), whether
poisoned padding changes a bit of the replayed mel, and the memory the
captures took.

  python s15_packed_step_graph_probe.py --model .../snapshots/master --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_left
from pathlib import Path

import torch
import torch.nn.functional as F
from s8_dit_exact_cuts import SHAPES, items, measure
from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)

from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT, PackedRows
from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    load_cosyvoice3_flow_hift,
    patch_chunk_mask,
)

SIZES_PER_DOUBLING = 8


def step_sizes(unit: int, limit: int) -> tuple[int, ...]:
    """Multiples of unit whose spacing doubles every SIZES_PER_DOUBLING sizes,
    then the largest multiple the limit holds."""
    top = limit // unit * unit
    sizes: list[int] = []
    size = spacing = unit
    while size < top:
        sizes.append(size)
        if size >= SIZES_PER_DOUBLING * spacing:
            spacing *= 2
        size += spacing
    return (*sizes, top)


def pad_frames(packed: torch.Tensor, frames: int, value: float = 0.0) -> torch.Tensor:
    """(1, total, channels) -> (1, frames, channels)."""
    if packed.shape[1] == frames:
        return packed
    return F.pad(packed, (0, 0, 0, frames - packed.shape[1]), value=value)


class StepGraphDiT(PackedDiT):
    """PackedDiT whose forward replays one captured step per step size."""

    def __init__(self, dit: torch.nn.Module, *, device: str, sizes: tuple[int, ...]):
        super().__init__(dit, device=device)
        self.sizes = sizes
        self.graphs = BaseBreakableCudaGraphRunner(self.step, torch.device(device))
        self.graphs.max_entries = 0
        self.mode = "eager"
        self.padding = 0.0
        self.rows: PackedRows | None = None
        self.attention = None
        self.step_rope: tuple[torch.Tensor, torch.Tensor] | None = None
        self.used_sizes: set[int] = set()

    def forward(self, x, mu, spks, cond, t, rows, attention):
        total = x.shape[1]
        index = bisect_left(self.sizes, total)
        if self.mode == "eager" or index == len(self.sizes):
            return super().forward(x, mu, spks, cond, t, rows, attention)
        frames = self.sizes[index]
        is_new_size = frames not in self.used_sizes
        self.used_sizes.add(frames)
        self.rows, self.attention = rows, attention
        cos, sin = PackedDiT.rope(self, rows)
        inputs = {
            "x": pad_frames(x, frames, self.padding),
            "mu": pad_frames(mu, frames, self.padding),
            "spks": pad_frames(spks, frames, self.padding),
            "cond": pad_frames(cond, frames, self.padding),
            "t": t,
            "cos": pad_frames(cos, frames),
            "sin": pad_frames(sin, frames),
        }
        if self.mode == "capture" and is_new_size:
            # The captured weight casts must be the graph's own: a cast that
            # autocast caches dies with its context.
            device_type = x.device.type
            with torch.autocast(
                device_type,
                dtype=torch.get_autocast_dtype(device_type),
                cache_enabled=False,
            ):
                self.graphs.capture(**inputs)
        return self.graphs(**inputs)[:, :total]

    def step(self, *, x, mu, spks, cond, t, cos, sin):
        self.step_rope = (cos, sin)
        return super().forward(x, mu, spks, cond, t, self.rows, self.attend_rows)

    def rope(self, rows):
        return self.step_rope

    @eager_on_graph(True)
    def attend_rows(self, query, key, value):
        total = self.rows.total
        out = self.attention(query[:, :total], key[:, :total], value[:, :total])
        return pad_frames(out, query.shape[1])

    @eager_on_graph(True)
    def conv_pos_embed(self, h, rows):
        rows = self.rows
        out = PackedDiT.conv_pos_embed(self, h[:, : rows.total], rows)
        return pad_frames(out, h.shape[1])


def snr_db(truth: list[torch.Tensor], mel: list[torch.Tensor]) -> float:
    signal = sum(float((a.double() ** 2).sum()) for a in truth)
    noise = sum(
        float(((a.double() - b.double()) ** 2).sum()) for a, b in zip(truth, mel)
    )
    return math.inf if noise == 0 else round(10 * math.log10(signal / noise), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=18000)
    args = parser.parse_args()

    flow, hift = load_cosyvoice3_flow_hift(args.model, device=args.device)
    patch_chunk_mask()
    for module in flow.decoder.estimator.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
    vocoder = CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    chunk = int(flow.decoder.estimator.static_chunk_size)
    estimator = StepGraphDiT(
        flow.decoder.estimator,
        device=args.device,
        sizes=step_sizes(2 * chunk, args.limit),
    )
    vocoder.flow.packed_estimator = estimator
    inputs = {shape: items(*shape, flow) for shape in SHAPES}
    calls = {
        f"{kind} {shape}": call
        for shape, batch in inputs.items()
        for kind, call in (
            ("hop", lambda batch=batch: vocoder.hop_batch(batch)),
            ("final", lambda batch=batch: vocoder.leftover_batch(batch)),
        )
    }

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "sizes": len(estimator.sizes),
        "largest_size": estimator.sizes[-1],
        "rows": [],
    }
    eager = {}
    for name, call in calls.items():
        eager[name] = {**measure(call), "mel": [row.clone() for row in call()]}

    torch.cuda.synchronize()
    free_before = torch.cuda.mem_get_info()[0]
    estimator.mode = "capture"
    for name, call in calls.items():
        call()
    torch.cuda.synchronize()
    report["capture_mib"] = round((free_before - torch.cuda.mem_get_info()[0]) / 2**20)
    report["captured_sizes"] = sorted(estimator.used_sizes)

    estimator.mode = "replay"
    for name, call in calls.items():
        replayed = measure(call)
        mel = [row.clone() for row in call()]
        estimator.padding = 1000.0
        poisoned = [row.clone() for row in call()]
        estimator.padding = 0.0
        truth = eager[name]["mel"]
        report["rows"].append(
            {
                "call": name,
                "frames": sum(int(row.shape[-1]) for row in mel),
                **{
                    f"eager_{key}": value
                    for key, value in eager[name].items()
                    if key != "mel"
                },
                **{f"replay_{key}": value for key, value in replayed.items()},
                "bit_identical_to_eager": all(
                    torch.equal(a, b) for a, b in zip(truth, mel)
                ),
                "max_abs_diff": max(
                    float((a - b).abs().max()) for a, b in zip(truth, mel)
                ),
                "snr_db": snr_db(truth, mel),
                "poisoned_padding_inert": all(
                    torch.equal(a, b) for a, b in zip(mel, poisoned)
                ),
            }
        )
        print(json.dumps(report["rows"][-1]), flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
