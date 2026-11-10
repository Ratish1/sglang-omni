"""Measure rank-keyed sampling sensitivity on complete fixed native histories."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path

import torch
from pydantic import BaseModel
from sglang.srt.layers.sampler import (
    multinomial_with_seed,
    top_k_top_p_min_p_sampling_from_probs_torch,
)


class SamplingParameters(BaseModel):
    temperature: float
    repetition_penalty: float
    top_k: int
    top_p: float
    min_p: float
    sampling_seed: int
    suppress_tokens: list[int]


class ReferenceFrame(BaseModel):
    frame_index: int
    native_main_code: int
    is_codec_eos: bool


class ReferenceRequest(BaseModel):
    request_id: str
    frames: list[ReferenceFrame]


class ReferenceReport(BaseModel):
    model_revision: str
    requests: list[ReferenceRequest]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--sampling-directory", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    reference = ReferenceReport.model_validate_json(
        (arguments.reference_directory / "results.json").read_bytes()
    )
    assert reference.model_revision == arguments.model_revision
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    reports = []
    with (arguments.output_directory / "frames.jsonl").open("w") as output:
        for request in reference.requests:
            directory = arguments.sampling_directory / request.request_id
            parameters = SamplingParameters.model_validate_json(
                (directory / "sampling.json").read_bytes()
            )
            assert parameters.min_p == 0
            logits = torch.load(
                arguments.reference_directory / f"{request.request_id}-logits.pt",
                map_location="cuda",
                weights_only=True,
            )
            repeated = torch.zeros_like(logits["checkpoint"][:1], dtype=torch.bool)
            suppressed = repeated.clone()
            suppressed[:, parameters.suppress_tokens] = True
            top_k = torch.tensor([parameters.top_k], device="cuda")
            top_p = torch.tensor([parameters.top_p], device="cuda")
            minimum_probability = torch.zeros(1, device="cuda")
            seed = torch.tensor([parameters.sampling_seed], device="cuda")
            counts: Counter[str] = Counter()
            assert len(request.frames) == len(logits["checkpoint"])
            for frame in request.frames:
                phase = "post_prefill" if frame.frame_index == 0 else "post_decode"
                captured = torch.load(
                    directory / f"{frame.frame_index:04d}-{phase}.pt",
                    map_location="cuda",
                    weights_only=True,
                )
                native_probabilities = captured["logits"]
                assert int(captured["codec_ids"][0, 0]) == frame.native_main_code
                checkpoint_logits = logits["checkpoint"][
                    frame.frame_index : frame.frame_index + 1
                ]
                penalized = torch.where(
                    checkpoint_logits > 0,
                    checkpoint_logits / parameters.repetition_penalty,
                    checkpoint_logits * parameters.repetition_penalty,
                )
                checkpoint_logits = torch.where(repeated, penalized, checkpoint_logits)
                checkpoint_probabilities = torch.softmax(
                    checkpoint_logits.masked_fill(suppressed, float("-inf"))
                    / parameters.temperature,
                    dim=-1,
                )
                positions = logits["positions"][
                    frame.frame_index : frame.frame_index + 1
                ]
                choices = []
                weights_by_token = []
                permutations = []
                for probabilities in (native_probabilities, checkpoint_probabilities):
                    choice, weights, permutation, _ = (
                        top_k_top_p_min_p_sampling_from_probs_torch(
                            probabilities,
                            top_k,
                            top_p,
                            minimum_probability,
                            False,
                            seed,
                            positions,
                            return_filtered_probs=True,
                        )
                    )
                    choices.append(int(choice[0]))
                    weights_by_token.append(
                        torch.zeros_like(weights).scatter_(
                            1, permutation.long(), weights
                        )
                    )
                    permutations.append(permutation.long())
                assert choices[0] == frame.native_main_code
                aligned_weights = weights_by_token[1].gather(1, permutations[0])
                assert torch.equal(
                    torch.zeros_like(aligned_weights).scatter_(
                        1, permutations[0], aligned_weights
                    ),
                    weights_by_token[1],
                )
                aligned_rank = multinomial_with_seed(
                    aligned_weights.double().log(), seed, positions
                )
                aligned_choice = int(permutations[0].gather(1, aligned_rank)[0, 0])
                token_order_choices = [
                    int(
                        multinomial_with_seed(weights.double().log(), seed, positions)[
                            0, 0
                        ]
                    )
                    for weights in weights_by_token
                ]
                normalized = [
                    weights / weights.sum(-1, keepdim=True)
                    for weights in weights_by_token
                ]
                support_difference = int(
                    ((weights_by_token[0] > 0) != (weights_by_token[1] > 0)).sum()
                )
                rank_changes = int(
                    (
                        (permutations[0] != permutations[1])
                        & (
                            (weights_by_token[0].gather(1, permutations[0]) > 0)
                            | (weights_by_token[1].gather(1, permutations[1]) > 0)
                        )
                    ).sum()
                )
                record = {
                    "request_id": request.request_id,
                    "frame_index": frame.frame_index,
                    "is_codec_eos": frame.is_codec_eos,
                    "native_choice_reproduced": True,
                    "native_rank_choice": choices[0],
                    "checkpoint_rank_choice": choices[1],
                    "checkpoint_weights_native_rank_choice": aligned_choice,
                    "fixed_token_order_choices": token_order_choices,
                    "filtered_distribution_total_variation": float(
                        0.5 * (normalized[0] - normalized[1]).abs().sum()
                    ),
                    "different_support_tokens": support_difference,
                    "different_positive_rank_slots": rank_changes,
                }
                output.write(json.dumps(record) + "\n")
                output.flush()
                counts["frames"] += 1
                counts["native_choices_reproduced"] += 1
                counts["rank_choices_agree"] += choices[0] == choices[1]
                counts["native_rank_aligned_choices_agree"] += (
                    choices[0] == aligned_choice
                )
                counts["fixed_token_order_choices_agree"] += (
                    token_order_choices[0] == token_order_choices[1]
                )
                repeated[:, frame.native_main_code] = True
            reports.append({"request_id": request.request_id, "counts": dict(counts)})
            print(json.dumps(reports[-1]), flush=True)
    sampler_source = Path(inspect.getfile(inspect.unwrap(multinomial_with_seed)))
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(
            {
                "model_revision": arguments.model_revision,
                "sampler_source_sha256": hashlib.sha256(
                    sampler_source.read_bytes()
                ).hexdigest(),
                "requests": reports,
                "scope": "Actual native probabilities reproduce every recorded choice before attribution. Checkpoint logits follow the same complete native input history and codec repetition mask. Rank alignment permutes already-filtered checkpoint weights without changing their token probabilities. Fixed token order is a separate noise-coordinate control, not native serving semantics or a proposed quality fix. No new generation or seed sweep.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
