"""Kernel time of the fused predictor layer against the plain path over one predictor
sequence (the opening pair and fourteen single tokens through five layers), the launch
count of each, the distance of each to fp32, and with `sweep` the launch configuration
grid of the fused path.

usage, from the tools tree with PYTHONPATH set to the fused tree:
  python3 predictor_fused_probe.py [sweep]
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import replace

import torch
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.runtime_context import get_context
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_omni.components import predictor_kernels
from tests.unit_test.qwen3_omni.test_predictor_kernels import (
    Reference,
    build_talker,
    fuse,
    predictor_inputs,
    relative_error,
    run_sequence,
)

BATCHES = (1, 12, 32)


def kernel_time(fn, iters: int = 20) -> tuple[float, int, dict[str, float]]:
    """Device time per call in us, kernel launches per call, and the time by kernel."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as trace:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    events = [e for e in trace.key_averages() if e.device_type.name == "CUDA"]
    by_name = {e.key[:60]: e.self_device_time_total / iters for e in events}
    total = sum(by_name.values())
    launches = sum(e.count for e in events) // iters
    return total, launches, by_name


def main() -> None:
    device = torch.device("cuda")
    talker = build_talker(device, seed=3)
    print(
        torch.cuda.get_device_name(device),
        "triton",
        predictor_kernels.triton.__version__,
    )
    for batch in BATCHES:
        steps = predictor_inputs(device, batch, seed=4)
        talker.predictor_fused = None
        plain_us, plain_launches, plain_by_name = kernel_time(
            lambda: run_sequence(talker, steps)
        )
        fuse(talker)
        fused_us, fused_launches, fused_by_name = kernel_time(
            lambda: run_sequence(talker, steps)
        )
        print(
            f"batch {batch:2d}: plain {plain_us:7.0f} us in {plain_launches:4d} launches;"
            f" fused {fused_us:7.0f} us in {fused_launches:4d} launches"
            f" ({fused_us / plain_us:.2f}x time, shape {talker.predictor_fused.shape})"
        )
        for label, by_name in (("plain", plain_by_name), ("fused", fused_by_name)):
            for name, us in sorted(by_name.items(), key=lambda item: -item[1])[:8]:
                print(f"    {label} {us:7.1f} us  {name}")
    print(
        "distance to fp32 over the sequence (relative L2), plain then fused, per step"
    )
    for batch in (1, 12):
        steps = predictor_inputs(device, batch, seed=4)
        talker.predictor_fused = None
        plain = run_sequence(talker, steps)
        fuse(talker)
        fused = run_sequence(talker, steps)
        reference = Reference(talker, batch)
        cache_len = 0
        rows = []
        with torch.no_grad():
            for step, plain_out, fused_out in zip(steps, plain, fused):
                expected = reference.forward(step, cache_len)
                rows.append(
                    f"{relative_error(plain_out, expected):.2e}/{relative_error(fused_out, expected):.2e}"
                )
                cache_len += step.shape[1]
        print(f"  batch {batch}: " + " ".join(rows))
    if "sweep" in sys.argv[1:]:
        shape = talker.predictor_fused.shape
        results = []
        for warps, stages, split in itertools.product((4, 8), (2, 3, 4), (1, 2, 4, 8)):
            predictor_kernels.NUM_WARPS = warps
            predictor_kernels.NUM_STAGES = stages
            talker.predictor_fused = predictor_kernels.FusedPredictorLayer(
                replace(shape, split_hidden=split, split_qkv=split),
                64,
                device,
                torch.bfloat16,
            )
            times = []
            for batch in (1, 12, 32):
                steps = predictor_inputs(device, batch, seed=4)
                times.append(
                    kernel_time(lambda: run_sequence(talker, steps), iters=10)[0]
                )
            results.append((warps, stages, split, times))
            print(
                f"warps {warps} stages {stages} split {split}:"
                + "".join(f" {t:7.0f}" for t in times),
                flush=True,
            )
        print("best by the sum over batches:")
        for warps, stages, split, times in sorted(results, key=lambda r: sum(r[3]))[:8]:
            print(
                f"  warps {warps} stages {stages} split {split}:"
                + "".join(f" {t:7.0f}" for t in times)
            )
    else:
        pass


if __name__ == "__main__":
    with get_context().override_server_args(
        cuda_graph_config=CudaGraphConfig(
            prefill=PhaseConfig(backend=Backend.DISABLED)
        ),
    ):
        main()
