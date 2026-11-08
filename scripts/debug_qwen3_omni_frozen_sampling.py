"""Decompose captured native/reference draws without changing their random state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig

from scripts.debug_qwen3_omni_sampling_coupling import sampling_draw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--reference-directory", type=Path, action="append", required=True
    )
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    assert not arguments.output_file.exists()
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    reports = []
    for directory in arguments.reference_directory:
        summary = json.loads((directory / "summary.json").read_text())
        for observation in summary:
            filename = (
                directory
                / f"{observation['request_id']}.{observation['history']}.{observation['frame']}.pt"
            )
            capture = torch.load(filename, map_location="cuda", weights_only=True)
            native, reference = [
                sampling_draw(
                    capture[name],
                    capture["repetition_mask"],
                    capture["seed"],
                    capture["positions"],
                    capture["temperature"],
                    capture["repetition_penalty"],
                    capture["top_k"],
                    capture["top_p"],
                    config.code_predictor_config.vocab_size,
                    config.codec_eos_token_id,
                )
                for name in ("native_logits", "reference_logits")
            ]
            assert native.sampled_token_id == observation["native_token_id"]
            assert reference.sampled_token_id == observation["reference_token_id"]
            native_probabilities = native.token_log_weights.softmax(-1)
            reference_probabilities = reference.token_log_weights.softmax(-1)
            support = (native_probabilities > 0) | (reference_probabilities > 0)
            report = {
                "reference_file": str(filename),
                "native_token_id": native.sampled_token_id,
                "reference_token_id": reference.sampled_token_id,
                "native_rank": native.sampled_rank,
                "reference_rank": reference.sampled_rank,
                "reference_probabilities_native_noise": int(
                    (reference.token_log_weights + native.token_noise).argmax()
                ),
                "native_probabilities_reference_noise": int(
                    (native.token_log_weights + reference.token_noise).argmax()
                ),
                "filtered_probability_total_variation": float(
                    (native_probabilities - reference_probabilities).abs().sum() / 2
                ),
                "supported_tokens_with_changed_noise": int(
                    ((native.token_noise != reference.token_noise) & support).sum()
                ),
            }
            reports.append(report)
    assert reports
    arguments.output_file.write_text(json.dumps(reports, indent=2))
    print(
        json.dumps(
            {
                "draw_pairs": len(reports),
                "changed_choices": sum(
                    row["native_token_id"] != row["reference_token_id"]
                    for row in reports
                ),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
