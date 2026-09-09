# SPDX-License-Identifier: Apache-2.0
"""E1 of plan 04: is the residual form of RMSNorm bit identical to add then norm.

Runs on the box against the pinned sglang. For each row count it draws random
bf16 x and residual, computes today's predictor path, the standalone add then
the norm, and the fused path, RMSNorm.forward_cuda(x, residual), on the same
class the predictor uses, and counts the trials where both the normed output
and the returned residual are equal bit for bit.

Usage: python e1_fused_add_rmsnorm.py [--hidden 1024] [--trials 1000]
"""
from __future__ import annotations

import argparse

import torch

from sglang_omni.vendor.sglang.layers import RMSNorm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--rows", default="1,2,4,8,16,32,64")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(1234)
    norm = RMSNorm(args.hidden, eps=args.eps).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(args.hidden, device=device) * 0.1 + 1.0)

    print(f"class={type(norm).__module__}.{type(norm).__name__} hidden={args.hidden}")
    print(
        "rows | normed equal | residual equal | max |normed diff| | max |residual diff|"
    )
    for rows in (int(r) for r in args.rows.split(",")):
        normed_equal = residual_equal = 0
        max_out = max_res = 0.0
        for _ in range(args.trials):
            x = torch.randn(rows, args.hidden, device=device, dtype=torch.bfloat16)
            residual = torch.randn(
                rows, args.hidden, device=device, dtype=torch.bfloat16
            )
            with torch.no_grad():
                today_residual = residual + x
                today_normed = norm(today_residual)
                fused_normed, fused_residual = norm.forward_cuda(
                    x.clone(), residual.clone()
                )
            normed_equal += int(torch.equal(today_normed, fused_normed))
            residual_equal += int(torch.equal(today_residual, fused_residual))
            max_out = max(
                max_out,
                (today_normed.float() - fused_normed.float()).abs().max().item(),
            )
            max_res = max(
                max_res,
                (today_residual.float() - fused_residual.float()).abs().max().item(),
            )
        print(
            f"{rows:4d} | {normed_equal:5d}/{args.trials} | "
            f"{residual_equal:5d}/{args.trials} | {max_out:.3e} | {max_res:.3e}"
        )


if __name__ == "__main__":
    main()
