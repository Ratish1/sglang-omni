"""P1-e1 paired read of the run files (predictor_pair_bench.py run outputs).

Per batch size and mode: sub-step 0 logits (the output of the first pass, same inputs
in every arm) against the fp32 truth for base and pair on every row, and for the rows
whose codes differ between base and pair, how many are closer to the truth's codes
(Hamming distance over the 15 predicted codes) in each arm.

usage: python predictor_pair_paired.py --base B.pt --pair P.pt --truth T.pt
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--truth", required=True)
    args = parser.parse_args()
    base, pair, truth = (torch.load(p) for p in (args.base, args.pair, args.truth))
    print(
        f"{'bs':>3} {'mode':>8} {'sub0 meanabs base':>18} {'pair':>10} {'sub0 maxabs base':>17} {'pair':>10}"
        f" {'rows differ':>12} {'pair closer':>12} {'base closer':>12} {'tie':>5}"
    )
    for key, reference in truth.items():
        batch, sampled = key
        b, p = base[key], pair[key]
        sub0 = [
            (arm["logits"][:, 0] - reference["logits"][:, 0]).abs() for arm in (b, p)
        ]
        differ = (b["codes"] != p["codes"]).any(dim=1)
        distance_b = (b["codes"][:, 1:] != reference["codes"][:, 1:]).sum(dim=1)
        distance_p = (p["codes"][:, 1:] != reference["codes"][:, 1:]).sum(dim=1)
        closer_p = int(((distance_p < distance_b) & differ).sum())
        closer_b = int(((distance_b < distance_p) & differ).sum())
        tie = int(differ.sum()) - closer_p - closer_b
        print(
            f"{batch:>3} {'sampled' if sampled else 'greedy':>8} {sub0[0].mean().item():>18.4e} {sub0[1].mean().item():>10.4e}"
            f" {sub0[0].max().item():>17.4e} {sub0[1].max().item():>10.4e}"
            f" {int(differ.sum()):>12} {closer_p:>12} {closer_b:>12} {tie:>5}"
        )


if __name__ == "__main__":
    main()
