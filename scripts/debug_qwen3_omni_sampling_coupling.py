"""Separate probability changes from ranked-noise changes on captured draws."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from transformers import Qwen3OmniMoeConfig


@dataclass(kw_only=True)
class SamplingDraw:
    token_log_weights: torch.Tensor
    token_noise: torch.Tensor
    sampled_token_id: int
    sampled_rank: int
    seed: int
    position: int
    repetition_mask: torch.Tensor


def replay_draw(
    directory: Path,
    step: int,
    temperature: float,
    repetition_penalty: float,
    top_k: int,
    top_p: float,
    codebook_size: int,
    eos_token_id: int,
) -> SamplingDraw:
    before = torch.load(
        directory / f"{step:04d}-before_decode.pt",
        map_location="cuda",
        weights_only=True,
    )
    after = torch.load(
        directory / f"{step:04d}-post_decode.pt",
        map_location="cuda",
        weights_only=True,
    )
    draw = sampling_draw(
        after["logits"],
        before["repetition_mask"],
        before["sampling_seed"].reshape(1),
        before["positions"].reshape(1),
        temperature,
        repetition_penalty,
        top_k,
        top_p,
        codebook_size,
        eos_token_id,
    )
    assert draw.sampled_token_id == int(after["codec_ids"][0, 0])
    return draw


def sampling_draw(
    logits: torch.Tensor,
    repetition_mask: torch.Tensor,
    seed: torch.Tensor,
    positions: torch.Tensor,
    temperature: float,
    repetition_penalty: float,
    top_k: int,
    top_p: float,
    codebook_size: int,
    eos_token_id: int,
) -> SamplingDraw:
    logits = logits.float()
    assert logits.shape[0] == 1 and torch.isfinite(logits).all()
    penalized = torch.where(
        logits > 0, logits / repetition_penalty, logits * repetition_penalty
    )
    logits = torch.where(repetition_mask, penalized, logits)
    token_ids = torch.arange(logits.shape[-1], device=logits.device)
    allowed = (token_ids < codebook_size) | (token_ids == eos_token_id)
    probabilities = torch.softmax(
        logits.masked_fill(~allowed, -torch.inf) / temperature, dim=-1
    )
    sampled, weights, sorted_token_ids, _ = top_k_top_p_min_p_sampling_from_probs_torch(
        probabilities,
        torch.tensor([top_k], device=logits.device),
        torch.tensor([top_p], device=logits.device),
        torch.zeros(1, device=logits.device),
        False,
        seed,
        positions,
        return_filtered_probs=True,
    )
    hashed = murmur_hash32(seed.to(torch.uint64), positions, token_ids)
    noise = hashed.to(torch.float64) / torch.iinfo(torch.uint32).max
    noise.log_().clamp_(min=torch.finfo(noise.dtype).min, max=-(2.0**-32)).neg_()
    noise.log_().neg_()
    log_weights = weights.double().log()
    selected_rank = (log_weights + noise).argmax(dim=-1)
    assert int(sorted_token_ids[0, selected_rank[0]]) == int(sampled[0])
    token_log_weights = torch.empty_like(log_weights).scatter_(
        1, sorted_token_ids.long(), log_weights
    )
    token_noise = torch.empty_like(noise).scatter_(1, sorted_token_ids.long(), noise)
    return SamplingDraw(
        token_log_weights=token_log_weights,
        token_noise=token_noise,
        sampled_token_id=int(sampled[0]),
        sampled_rank=int(selected_rank[0]),
        seed=int(seed[0]),
        position=int(positions[0]),
        repetition_mask=repetition_mask,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--before-directory", type=Path, required=True)
    parser.add_argument("--after-directory", type=Path, required=True)
    parser.add_argument(
        "--case", action="append", required=True, help="Request UUID:decode step"
    )
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    reports = []
    for case in arguments.case:
        request_id, step_text = case.rsplit(":", 1)
        draws = [
            replay_draw(
                directory / request_id,
                int(step_text),
                arguments.temperature,
                arguments.repetition_penalty,
                arguments.top_k,
                arguments.top_p,
                config.code_predictor_config.vocab_size,
                config.codec_eos_token_id,
            )
            for directory in (arguments.before_directory, arguments.after_directory)
        ]
        before, after = draws
        assert before.seed == after.seed and before.position == after.position
        assert torch.equal(before.repetition_mask, after.repetition_mask)
        probabilities = [draw.token_log_weights.softmax(-1) for draw in draws]
        support = (probabilities[0] > 0) | (probabilities[1] > 0)
        report = {
            "request_id": request_id,
            "decode_step": int(step_text),
            "seed": before.seed,
            "position": before.position,
            "before_sampled_token_id": before.sampled_token_id,
            "after_sampled_token_id": after.sampled_token_id,
            "before_sampled_rank": before.sampled_rank,
            "after_sampled_rank": after.sampled_rank,
            "new_probabilities_old_token_noise_choice": int(
                (after.token_log_weights + before.token_noise).argmax()
            ),
            "old_probabilities_new_token_noise_choice": int(
                (before.token_log_weights + after.token_noise).argmax()
            ),
            "probability_total_variation": float(
                (probabilities[0] - probabilities[1]).abs().sum() / 2
            ),
            "support_equal": bool(
                torch.equal(probabilities[0] > 0, probabilities[1] > 0)
            ),
            "supported_tokens_with_changed_rank_noise": int(
                ((before.token_noise != after.token_noise) & support).sum()
            ),
        }
        reports.append(report)
        arguments.output_file.write_text(json.dumps(reports, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
