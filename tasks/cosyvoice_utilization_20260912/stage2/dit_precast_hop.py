"""One Flow call before and after casting the DiT weights once, same process.

Loads the real Flow as the vocoder factory does, runs hops and finals through
the serving entry points (hop_batch, leftover_batch, their autocast and
inference mode included) on float32 weights, then casts the DiT's Linear and
Conv1d parameters to bfloat16 and runs the same inputs again. Reports, per
shape: wall time, the number of dtype copies one call launches, the number of
kernels it launches, and whether the mel is bit identical.

  python dit_precast_hop.py --model .../snapshots/master
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    FlowBatchInput,
    load_cosyvoice3_flow_hift,
    patch_chunk_mask,
)

WARMUP = 3
ITERATIONS = 10

# rows, prompt tokens, generated tokens so far (the hop reads all of them)
SHAPES: tuple[tuple[int, int, int], ...] = (
    (1, 100, 28),
    (1, 150, 228),
    (4, 150, 128),
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


def measure(call) -> dict[str, float | int]:
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    walls = []
    for _ in range(ITERATIONS):
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - started) * 1000)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        call()
        torch.cuda.synchronize()
    events = trace.events()
    return {
        "wall_ms_median": round(statistics.median(walls), 2),
        "wall_ms_min": round(min(walls), 2),
        "dtype_copies": sum(1 for event in events if event.name == "aten::_to_copy"),
        "kernels": sum(
            1 for event in events if event.device_type == torch.autograd.DeviceType.CUDA
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    flow, hift = load_cosyvoice3_flow_hift(args.model, device="cuda:0")
    patch_chunk_mask()
    vocoder = CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    inputs = {shape: items(*shape, flow) for shape in SHAPES}

    def run_all() -> dict:
        rows = {}
        for shape, batch in inputs.items():
            for kind, call in (
                ("hop", lambda batch=batch: vocoder.hop_batch(batch)),
                ("final", lambda batch=batch: vocoder.leftover_batch(batch)),
            ):
                result = measure(call)
                result["mel"] = [mel.clone() for mel in call()]
                # Whether the same weights reproduce themselves, so a difference
                # after the cast can be told from one the kernels make anyway.
                result["repeatable"] = all(
                    torch.equal(a, b) for a, b in zip(result["mel"], call())
                )
                rows[f"{kind} {shape}"] = result
        return rows

    before = run_all()
    weights_before = torch.cuda.memory_allocated()
    for module in flow.decoder.estimator.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            module.to(torch.bfloat16)
    torch.cuda.empty_cache()
    weights_after = torch.cuda.memory_allocated()
    after = run_all()

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "allocated_mib_before_cast": round(weights_before / (1 << 20), 1),
        "allocated_mib_after_cast": round(weights_after / (1 << 20), 1),
        "rows": [],
    }
    for name, old in before.items():
        new = after[name]
        report["rows"].append(
            {
                "call": name,
                "repeatable_float32": old["repeatable"],
                "repeatable_cast": new["repeatable"],
                "bit_identical": all(
                    torch.equal(a, b) for a, b in zip(old["mel"], new["mel"])
                ),
                "max_abs_diff": max(
                    float((a - b).abs().max()) for a, b in zip(old["mel"], new["mel"])
                ),
                **{
                    f"{key}_float32": old[key]
                    for key in (
                        "wall_ms_median",
                        "wall_ms_min",
                        "dtype_copies",
                        "kernels",
                    )
                },
                **{
                    f"{key}_cast": new[key]
                    for key in (
                        "wall_ms_median",
                        "wall_ms_min",
                        "dtype_copies",
                        "kernels",
                    )
                },
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
