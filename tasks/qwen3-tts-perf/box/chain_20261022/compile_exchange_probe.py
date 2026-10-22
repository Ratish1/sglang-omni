"""Whether torch.compile (fullgraph, static shapes, as the vocoder's width-8 graphs compile) traces the codec state
kernels' Triton launches and keeps their in-place writes: a conv step written the slot way (producer into a padded
input, history exchanged with a pool, conv without bias, bias and SnakeBeta fused, transposed conv overlap exchanged)
run eager and compiled on the same inputs, outputs and pools compared bit for bit, and the compiled step's kernels.

usage: PYTHONPATH=<tree> python3 compile_exchange_probe.py [--card 3]
"""

import argparse

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from sglang_omni.kernels.bias_snake_beta import SnakeBetaCoefficients, fused_bias_snake_beta
from sglang_omni.kernels.codec_state import history_exchange, overlap_exchange

ROWS = 4
SLOTS = 9
LENGTH = 8
CHANNELS = 384
HISTORY = 18
OVERLAP = 4
STRIDE = 4


def step(
    frames: torch.Tensor,
    history_pool: torch.Tensor,
    overlap_pool: torch.Tensor,
    slot_index: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    transconv_weight: torch.Tensor,
    coefficients: SnakeBetaCoefficients,
) -> torch.Tensor:
    padded = frames.new_empty(ROWS, HISTORY + LENGTH, CHANNELS)
    padded[:, HISTORY:] = frames + 1.0
    history_exchange(padded, history_pool, slot_index, new_samples=LENGTH, is_cold=False)
    conv_output = F.conv1d(padded.transpose(1, 2), weight, None, dilation=3).transpose(1, 2)
    activated = conv_output.new_empty(ROWS, LENGTH, CHANNELS)
    fused_bias_snake_beta(conv_output.contiguous(), bias, coefficients, activated, residual=frames)
    output = F.conv_transpose1d(activated.transpose(1, 2), transconv_weight, None, stride=STRIDE).transpose(1, 2)
    output = output.contiguous()
    overlap_exchange(output, overlap_pool, slot_index, emitted_samples=LENGTH * STRIDE, is_cold=False)
    return output[:, : LENGTH * STRIDE] * 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    torch.manual_seed(0)

    def bf16(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*shape, device=device) * scale).to(torch.bfloat16)

    frames = bf16(ROWS, LENGTH, CHANNELS)
    history_pool = bf16(SLOTS, HISTORY, CHANNELS)
    overlap_pool = bf16(SLOTS, OVERLAP, CHANNELS // 2)
    slot_index = torch.tensor([3, 0, 7, 5], device=device)
    weight = bf16(CHANNELS, CHANNELS, 7, scale=0.02)
    bias = bf16(CHANNELS, scale=0.1)
    transconv_weight = bf16(CHANNELS, CHANNELS // 2, OVERLAP + STRIDE, scale=0.02)
    activation = torch.nn.Module()
    activation.alpha = bf16(CHANNELS, scale=0.1)
    activation.beta = bf16(CHANNELS, scale=0.1)
    activation.no_div_by_zero = 1e-9
    coefficients = SnakeBetaCoefficients.from_activation(activation)

    results = {}
    for name, function in (("eager", step), ("compiled", torch.compile(step, dynamic=False, fullgraph=True))):
        pools = (history_pool.clone(), overlap_pool.clone())
        with torch.inference_mode():
            first = function(frames, *pools, slot_index, weight, bias, transconv_weight, coefficients)
            second = function(frames, *pools, slot_index, weight, bias, transconv_weight, coefficients)
        torch.cuda.synchronize(device)
        results[name] = (first.clone(), second.clone(), *pools)
        print(f"{name}: ran")

    names = ("first output", "second output", "history pool", "overlap pool")
    for label, eager, compiled in zip(names, results["eager"], results["compiled"]):
        print(f"{label}: equal {torch.equal(eager, compiled)} max diff {(eager.float() - compiled.float()).abs().max().item():.3e}")
    untouched = [slot for slot in range(SLOTS) if slot not in slot_index.tolist()]
    print(
        "untouched slots kept:",
        torch.equal(results["compiled"][2][untouched], history_pool[untouched]),
        torch.equal(results["compiled"][3][untouched], overlap_pool[untouched]),
    )

    compiled_step = torch.compile(step, dynamic=False, fullgraph=True)
    pools = (history_pool.clone(), overlap_pool.clone())
    with torch.inference_mode():
        compiled_step(frames, *pools, slot_index, weight, bias, transconv_weight, coefficients)
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CUDA]) as profiler:
            compiled_step(frames, *pools, slot_index, weight, bias, transconv_weight, coefficients)
            torch.cuda.synchronize(device)
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            print(f"  kernel {event.time_range.elapsed_us():7.2f} us  {event.name[:100]}")
        else:
            pass


if __name__ == "__main__":
    main()
