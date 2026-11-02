"""Do the talker's and predictor's GEMM weights survive FP8? Rounds each to the e4m3 grid under one static per-tensor scale
(its largest magnitude maps to 448, the format's largest finite value, as an FP8 checkpoint's static scale does), stores the
result back as bf16, and writes a checkpoint directory that serves unchanged through today's kernels. Norms, embeddings, the
speaker encoder and the vocoder are untouched. This tests weight rounding only: activations stay bf16, so a pass says FP8
weights are worth building, not that FP8 serving is correct.

usage: python3 fp8_weights_sim.py --source SNAPSHOT_DIR --out DIR
"""

import argparse
import re
import statistics
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP8_MAX = 448.0
FP8_SMALLEST_STEP = 2.0**-9
GEMM_WEIGHT = re.compile(
    r"^talker\.("
    r"model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
    r"|codec_head"
    r"|code_predictor\.model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
    r"|code_predictor\.lm_head\.\d+"
    r"|code_predictor\.small_to_mtp_projection"
    r")\.weight$"
)


def round_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """The weight on the e4m3 grid under its static scale, and the scale."""
    values = weight.float()
    scale = float(values.abs().max()) / FP8_MAX
    if scale == 0.0:
        return weight, scale
    else:
        pass
    scaled = values / scale
    _, exponents = torch.frexp(scaled)
    # note: |x| = m * 2^e with m in [0.5, 1), so the leading bit is 2^(e-1); a hidden bit and three stored bits step at 2^(e-4).
    steps = torch.maximum(torch.ldexp(torch.ones_like(scaled), exponents - 4), torch.tensor(FP8_SMALLEST_STEP))
    rounded = torch.round(scaled / steps) * steps
    return (rounded * scale).to(weight.dtype), scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for entry in args.source.iterdir():
        target = args.out / entry.name
        if entry.name != "model.safetensors" and not target.exists():
            target.symlink_to(entry.resolve())
        else:
            pass
    tensors: dict[str, torch.Tensor] = {}
    errors: dict[str, list[float]] = {"talker": [], "predictor": []}
    rounded_params = 0
    total_params = 0
    with safe_open(str(args.source / "model.safetensors"), "pt") as source:
        metadata = source.metadata()
        for key in source.keys():
            tensor = source.get_tensor(key)
            total_params += tensor.numel()
            if GEMM_WEIGHT.match(key):
                rounded, _ = round_to_fp8(tensor)
                distance = float((rounded.float() - tensor.float()).norm() / tensor.float().norm().clamp_min(1e-12))
                errors["predictor" if "code_predictor" in key else "talker"].append(distance)
                rounded_params += tensor.numel()
                tensors[key] = rounded
            else:
                tensors[key] = tensor
    save_file(tensors, str(args.out / "model.safetensors"), metadata=metadata or {"format": "pt"})
    print(f"rounded {rounded_params / 1e6:.1f} M of {total_params / 1e6:.1f} M parameters")
    for group, values in errors.items():
        print(f"{group}: {len(values)} tensors, relative L2 median {statistics.median(values):.4f}, max {max(values):.4f}")


if __name__ == "__main__":
    main()
