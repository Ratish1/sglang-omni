# SPDX-License-Identifier: Apache-2.0
"""Separate probability and rank-order effects on the paired first codec draw."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sglang.srt.layers.sampler import (
    multinomial_with_seed,
    top_k_top_p_min_p_sampling_from_probs_torch,
)


def readout(
    sampling: dict[str, torch.Tensor]
) -> tuple[int, torch.Tensor, torch.Tensor]:
    logits = sampling["raw_logits"]
    penalty = sampling["repetition_penalty"]
    penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
    logits = torch.where(sampling["repetition_mask"], penalized, logits)
    logits = logits.masked_fill(sampling["suppress_mask"], float("-inf"))
    probabilities = (logits / sampling["temperature"]).softmax(-1).view(1, -1)
    selected, weights, permutation, _ = top_k_top_p_min_p_sampling_from_probs_torch(
        probabilities,
        sampling["top_k"].reshape(1),
        sampling["top_p"].reshape(1),
        probabilities.new_zeros(1),
        False,
        sampling["seed"].reshape(1),
        sampling["position"].reshape(1),
        return_filtered_probs=True,
    )
    assert int(selected[0]) == int(sampling["sampled_code"])
    weights_by_token = torch.zeros_like(weights).scatter(1, permutation.long(), weights)
    return int(selected[0]), weights_by_token, permutation.long()


def swap_ranks(
    weights_by_token: torch.Tensor,
    ranks: torch.Tensor,
    seed: torch.Tensor,
    position: torch.Tensor,
) -> int:
    weights_by_rank = weights_by_token.gather(1, ranks)
    selected_rank = multinomial_with_seed(
        weights_by_rank.double().log(), seed.reshape(1), position.reshape(1)
    )
    return int(ranks.gather(1, selected_rank)[0, 0])


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    reports = []
    for line in (arguments.population / "generation.jsonl").read_text().splitlines():
        row = json.loads(line)
        pair = row["pairs"]["first"]
        if not pair["opening_state_equal"]:
            continue
        else:
            pass
        directory = arguments.population / row["directory"]
        native = torch.load(
            directory / "none" / "decode-sampling-001.pt",
            map_location="cuda:0",
            weights_only=True,
        )
        swapped = torch.load(
            directory / "first" / "decode-sampling-001.pt",
            map_location="cuda:0",
            weights_only=True,
        )
        seed = native["seed"]
        position = native["position"]
        assert torch.equal(seed, swapped["seed"]) and torch.equal(
            position, swapped["position"]
        )
        native_code, native_weights, native_ranks = readout(native)
        swapped_code, swapped_weights, swapped_ranks = readout(swapped)
        assert swap_ranks(native_weights, native_ranks, seed, position) == native_code
        assert (
            swap_ranks(swapped_weights, swapped_ranks, seed, position) == swapped_code
        )
        rank_only = swap_ranks(native_weights, swapped_ranks, seed, position)
        weights_only = swap_ranks(swapped_weights, native_ranks, seed, position)
        native_distribution = native_weights / native_weights.sum()
        swapped_distribution = swapped_weights / swapped_weights.sum()
        reports.append(
            {
                "sample_id": row["sample_id"],
                "cohort": row["cohort"],
                "native_code": native_code,
                "swapped_code": swapped_code,
                "only_swap_ranks": rank_only,
                "only_swap_weights": weights_only,
                "native_code_probability_before": float(
                    native_distribution[0, native_code]
                ),
                "native_code_probability_after": float(
                    swapped_distribution[0, native_code]
                ),
                "total_variation": float(
                    0.5 * (native_distribution - swapped_distribution).abs().sum()
                ),
            }
        )
    with arguments.output.open("x") as output:
        json.dump(reports, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
else:
    pass
