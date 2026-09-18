"""Wall time of one Flow call with float32 DiT weights or with them cast once.

dit_precast_hop.py settled identity and launch counts in one process, but it
profiled between its timings, and once the profiler has run in a process every
later launch pays for its callbacks. This one never profiles: one weight state
per process, the same inputs and entry points, wall time only, each shape timed
in three interleaved rounds so drift in the host shows up as spread.

  python dit_precast_time.py --model .../snapshots/master
  python dit_precast_time.py --model .../snapshots/master --cast
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    FlowBatchInput,
    load_cosyvoice3_flow_hift,
    patch_chunk_mask,
)

WARMUP = 3
ITERATIONS = 10
ROUNDS = 3

# rows, prompt tokens, generated tokens so far (the hop reads all of them)
SHAPES: tuple[tuple[int, int, int], ...] = (
    (1, 100, 28),
    (1, 150, 228),
    (2, 150, 128),
    (4, 150, 128),
    (8, 150, 128),
    (16, 150, 128),
)


def items(rows: int, prompt: int, tokens: int, flow) -> list[FlowBatchInput]:
    generator = torch.Generator().manual_seed(rows * 1000 + tokens)
    return [
        FlowBatchInput(
            token=torch.randint(
                0, 6000, (1, tokens + 8 * row), generator=generator, dtype=torch.int32
            ),
            prompt_token=torch.randint(
                0, 6000, (1, prompt), generator=generator, dtype=torch.int32
            ),
            prompt_feat=torch.randn(
                1, prompt * 2, flow.output_size, generator=generator
            ),
            embedding=torch.randn(
                1, flow.spk_embed_affine_layer.in_features, generator=generator
            ),
        )
        for row in range(rows)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cast", action="store_true")
    args = parser.parse_args()

    flow, hift = load_cosyvoice3_flow_hift(args.model, device="cuda:0")
    patch_chunk_mask()
    if args.cast:
        for module in flow.decoder.estimator.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                module.to(torch.bfloat16)
    vocoder = CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)

    calls = {}
    for shape in SHAPES:
        batch = items(*shape, flow)
        calls[f"hop {shape}"] = lambda batch=batch: vocoder.hop_batch(batch)
        calls[f"final {shape}"] = lambda batch=batch: vocoder.leftover_batch(batch)

    for call in calls.values():
        for _ in range(WARMUP):
            call()
    torch.cuda.synchronize()

    walls: dict[str, list[float]] = {name: [] for name in calls}
    for _ in range(ROUNDS):
        for name, call in calls.items():
            for _ in range(ITERATIONS):
                started = time.perf_counter()
                call()
                torch.cuda.synchronize()
                walls[name].append((time.perf_counter() - started) * 1000)

    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "cast": args.cast,
                "rows": [
                    {
                        "call": name,
                        "median_ms": round(statistics.median(values), 2),
                        "min_ms": round(min(values), 2),
                        "round_medians_ms": [
                            round(
                                statistics.median(
                                    values[
                                        index * ITERATIONS : (index + 1) * ITERATIONS
                                    ]
                                ),
                                2,
                            )
                            for index in range(ROUNDS)
                        ],
                    }
                    for name, values in walls.items()
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
