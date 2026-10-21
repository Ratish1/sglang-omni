"""The tree's bias SnakeBeta kernel on every decode shape: time, and every output against eager SnakeBeta.

Run on the box from a tree with the stored-coefficient kernel: cd <tree> && PYTHONPATH=<tree>
    python3 /workspace/box/tasks/pr2800_review/box/snake_kernel_time.py --out /data/pr2800/<run>
The 29 bias SnakeBeta launches of a 10-frame decode at 1 and 8 rows, each timed as code2wav_kernel_sweep.py times it
(20 launches in one graph, 50 replays), with SnakeBetaCoefficients.from_activation of an HF SnakeBeta whose alpha and
beta are drawn at N(0, 0.5), and its output and sum compared bit for bit with the eager adds and SnakeBeta forward.
Writes <out>/kernel_time.json; run200's "flat" sums were 83.3 / 394.5 us (rows 1 / 8).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeSnakeBeta

from sglang_omni.kernels.bias_snake_beta import SnakeBetaCoefficients, fused_bias_snake_beta

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code2wav_kernel_sweep import DTYPE, MODEL, graph_time_us, shapes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).code2wav_config
    _, _, snakes = shapes(config)
    records = []
    for rows in (1, 8):
        for shape in snakes:
            channels, length = shape["channels"], shape["length"]
            activation = Qwen3OmniMoeSnakeBeta(channels).to(device="cuda", dtype=DTYPE).eval()
            with torch.no_grad():
                activation.alpha.normal_(0.0, 0.5)
                activation.beta.normal_(0.0, 0.5)
            coefficients = SnakeBetaCoefficients.from_activation(activation)
            conv = torch.randn(rows, length + 7, channels, device="cuda", dtype=DTYPE)[:, 3 : 3 + length]
            residual = torch.randn(rows, length, channels, device="cuda", dtype=DTYPE)
            bias = torch.randn(channels, device="cuda", dtype=DTYPE)
            out = torch.zeros(rows, length + 60, channels, device="cuda", dtype=DTYPE)[:, 54 : 54 + length]
            total = residual.clone()
            with torch.inference_mode():
                expected_sum = conv + bias + residual if shape["residual"] else conv + bias
                expected = activation(expected_sum.transpose(1, 2)).transpose(1, 2)
                fused_bias_snake_beta(conv, bias, coefficients, out, residual=total if shape["residual"] else None,
                                      sum_out=total if shape["total"] else None)
                equal = bool(torch.equal(out, expected) and (not shape["total"] or torch.equal(total, expected_sum)))
                us = graph_time_us(lambda: fused_bias_snake_beta(
                    conv, bias, coefficients, out, residual=total if shape["residual"] else None,
                    sum_out=total if shape["total"] else None))
            records.append(dict(rows=rows, **shape, us=round(us, 3), equal=equal))
            print(f"rows {rows} {shape['name']:30s} C {channels:5d} L {length:6d} {us:6.2f} us equal {equal}",
                  flush=True)
    (args.out / "kernel_time.json").write_text(json.dumps(dict(records=records), indent=1))
    for rows in (1, 8):
        group = [r for r in records if r["rows"] == rows]
        print(f"rows {rows} (29 launches): {sum(r['us'] for r in group):.1f} us; all equal to eager "
              f"{all(r['equal'] for r in group)}")


if __name__ == "__main__":
    main()
