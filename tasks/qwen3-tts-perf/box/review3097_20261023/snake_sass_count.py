"""Instruction counts of the bias SnakeBeta kernel's eager-rounding and fp32-once arithmetic (snake_variants_bench.py's
kernel), compiled for code2wav's (96 channels, 19,200 positions) act2 launch and its residual-and-sum launch: total
SASS instructions and the opcodes whose count differs between the two.

usage: python3 snake_sass_count.py [--card 2]
"""

import argparse
import collections
import json
import re

import torch
from snake_variants_bench import launch


def counts(kernel) -> collections.Counter:
    opcodes = collections.Counter()
    for line in kernel.asm["sass"].splitlines():
        match = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
        if match:
            opcodes[match.group(1)] += 1
        else:
            pass
    return opcodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    conv = torch.randn(1, 19200, 96, device=device, dtype=torch.bfloat16)
    bias = torch.randn(96, device=device, dtype=torch.bfloat16)
    alpha_scale = torch.ones(96, device=device, dtype=torch.bfloat16)
    inverse_beta = torch.ones(96, device=device, dtype=torch.bfloat16)
    for shape, residual, sum_out in (("act2", None, None), ("residual_sum", torch.zeros_like(conv), torch.zeros_like(conv))):
        found = {}
        for variant in ("eager_nofma", "fp32_nofma"):
            found[variant] = counts(launch(conv, bias, alpha_scale, inverse_beta, torch.empty_like(conv), residual,
                                           sum_out, variant))
        eager, fp32 = found["eager_nofma"], found["fp32_nofma"]
        differing = {op: [eager[op], fp32[op]] for op in sorted(set(eager) | set(fp32)) if eager[op] != fp32[op]}
        print(json.dumps({"shape": shape, "total": [sum(eager.values()), sum(fp32.values())],
                          "differing_eager_fp32": differing}), flush=True)


if __name__ == "__main__":
    main()
