"""Separate probability drift from rank-noise reassignment on actual complete histories."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from sglang.srt.layers.sampler import (
    multinomial_with_seed,
    top_k_top_p_min_p_sampling_from_probs_torch,
)

from scripts.debug_qwen3_omni_sampling_alignment import (
    SamplingParameters,
    sample_permuted_weights,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    comparison = parser.add_mutually_exclusive_group(required=True)
    comparison.add_argument("--reference-directory", type=Path)
    comparison.add_argument("--native-only", action="store_true")
    parser.add_argument("--expect-token-coordinates", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    reports = []
    for directory in sorted(
        path for path in arguments.capture_directory.iterdir() if path.is_dir()
    ):
        parameters = SamplingParameters.model_validate_json(
            (directory / "sampling.json").read_bytes()
        )
        assert parameters.min_p == 0
        finished = torch.load(directory / "finished.pt", weights_only=True)
        initial = torch.load(directory / "0000-before_prefill.pt", weights_only=True)
        files = sorted(directory.glob("*-output.pt"))
        assert [filename.name for filename in files] == [
            f"{frame:04d}-output.pt" for frame in range(len(finished["output_ids"]))
        ]
        top_k = torch.tensor([parameters.top_k], device=device)
        top_p = torch.tensor([parameters.top_p], device=device)
        minimum_probability = torch.tensor([parameters.min_p], device=device)
        seed = torch.tensor([parameters.sampling_seed], device=device)
        repetition_penalty = torch.tensor(
            [[parameters.repetition_penalty]], dtype=torch.float32, device=device
        )
        temperature = torch.tensor(
            [[parameters.temperature]], dtype=torch.float32, device=device
        )
        repeated = torch.zeros(
            (1, len(torch.load(files[0], weights_only=True)["logits"])),
            dtype=torch.bool,
            device=device,
        )
        counts = {
            owner: Counter()
            for owner in (() if arguments.native_only else ("checkpoint", "native"))
        }
        token_coordinate_checks = 0
        for frame, filename in enumerate(files):
            observed = torch.load(filename, weights_only=True)
            assert bool(observed["is_extend"]) == (frame == 0)
            phase = "before_prefill" if frame == 0 else "before_decode"
            before = torch.load(
                directory / f"{frame:04d}-{phase}.pt", weights_only=True
            )
            assert torch.equal(before["output_ids"], finished["output_ids"][:frame])
            if frame:
                assert int(before["sampling_seed"][0]) == parameters.sampling_seed
                assert torch.equal(
                    before["repetition_mask"].to(device).unsqueeze(0), repeated
                )
            else:
                pass
            sources = {"observed": observed}
            for owner in counts:
                sources[owner] = torch.load(
                    arguments.reference_directory
                    / directory.name
                    / owner
                    / filename.name,
                    weights_only=True,
                )
                expected_inputs = (
                    initial["prefill_embeds"].unsqueeze(0)
                    if frame == 0
                    else finished["decode_input_history"][frame - 1].view(1, 1, -1)
                )
                assert torch.equal(sources[owner]["hidden_states"][0], expected_inputs)
                assert sources[owner]["hidden_states"][0].dtype == expected_inputs.dtype
            position = torch.tensor(
                [len(initial["prefill_embeds"]) + frame - 1], device=device
            )
            assert int(observed["sequence_length"]) == int(position[0]) + 1
            assert int(before["positions"][-1]) == int(position[0])
            choices: dict[str, int] = {}
            token_weights: dict[str, torch.Tensor] = {}
            permutations: dict[str, torch.Tensor] = {}
            for source, output in sources.items():
                if source == "observed" and frame == 0:
                    probabilities = output["logits"].unsqueeze(0).to(device)
                else:
                    logits = output["logits"].to(device)
                    if source == "observed":
                        logits = logits.unsqueeze(0)
                    else:
                        pass
                    penalized = torch.where(
                        logits > 0,
                        logits / repetition_penalty,
                        logits * repetition_penalty,
                    )
                    logits = torch.where(repeated, penalized, logits)
                    logits[:, parameters.suppress_tokens] = float("-inf")
                    probabilities = (logits / temperature).softmax(-1)
                chosen, weights, permutation, selected_weight = (
                    top_k_top_p_min_p_sampling_from_probs_torch(
                        probabilities,
                        top_k,
                        top_p,
                        minimum_probability,
                        False,
                        seed,
                        position,
                        return_filtered_probs=True,
                    )
                )
                choices[source] = int(chosen[0])
                assert weights.isfinite().all() and weights.sum() > 0
                permutations[source] = permutation.long()
                token_weights[source] = torch.zeros_like(weights).scatter_(
                    1, permutations[source], weights
                )
                assert torch.equal(
                    selected_weight,
                    token_weights[source].gather(1, chosen.long().view(-1, 1)).view(-1),
                )
            assert (
                choices["observed"]
                == int(observed["codec_ids"][0])
                == int(finished["output_ids"][frame])
            )
            if arguments.expect_token_coordinates:
                expected = multinomial_with_seed(
                    token_weights["observed"].double().log(), seed, position
                )
                assert int(expected[0, 0]) == choices["observed"], frame
                token_coordinate_checks += 1
            else:
                pass
            for owner, owner_counts in counts.items():
                aligned = sample_permuted_weights(
                    token_weights[owner], permutations["observed"], seed, position
                )
                reordered_observed = sample_permuted_weights(
                    token_weights["observed"], permutations[owner], seed, position
                )
                token_choices = [
                    int(
                        multinomial_with_seed(
                            token_weights[source].double().log(), seed, position
                        )[0, 0]
                    )
                    for source in ("observed", owner)
                ]
                normalized = [
                    token_weights[source] / token_weights[source].sum(-1, keepdim=True)
                    for source in ("observed", owner)
                ]
                row = {
                    "request_id": directory.name,
                    "frame": frame,
                    "cache_owner": owner,
                    "native_draw_exact": True,
                    "native_rank_choice": choices["observed"],
                    "reference_rank_choice": choices[owner],
                    "reference_weights_native_rank_choice": aligned,
                    "native_weights_reference_rank_choice": reordered_observed,
                    "fixed_token_choices": token_choices,
                    "filtered_distribution_total_variation": float(
                        0.5 * (normalized[0] - normalized[1]).abs().sum()
                    ),
                    "observed_choice_native_probability": float(
                        normalized[0][0, choices["observed"]]
                    ),
                    "observed_choice_reference_probability": float(
                        normalized[1][0, choices["observed"]]
                    ),
                }
                assert row["observed_choice_native_probability"] > 0
                with (arguments.output_directory / "frames.jsonl").open("a") as output:
                    output.write(json.dumps(row) + "\n")
                owner_counts["frames"] += 1
                owner_counts["ordinary_agreement"] += (
                    choices["observed"] == choices[owner]
                )
                owner_counts["native_rank_aligned_agreement"] += (
                    choices["observed"] == aligned
                )
                owner_counts["fixed_token_agreement"] += (
                    token_choices[0] == token_choices[1]
                )
                owner_counts["order_only_native_changes"] += (
                    choices["observed"] != reordered_observed
                )
            repeated[:, choices["observed"]] = True
        report = {
            "request_id": directory.name,
            "native_only": arguments.native_only,
            "native_draws_reproduced": len(files),
            "token_coordinate_checks": token_coordinate_checks,
            "counts": {owner: dict(values) for owner, values in counts.items()},
        }
        reports.append(report)
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(reports, indent=2)
        )
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
