# SPDX-License-Identifier: Apache-2.0
"""Compare captured Talker forwards with the checkpoint implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    device = torch.device("cuda", arguments.gpu)
    torch.cuda.set_device(device)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    comparisons: list[dict[str, str | int | float | list[int]]] = []
    cache = None
    with torch.inference_mode():
        for step in range(arguments.steps):
            before_phase = "before_prefill" if step == 0 else "before_decode"
            after_phase = "prefill_forward" if step == 0 else "post_decode"
            before = torch.load(
                arguments.trace_directory / f"{step:04d}-{before_phase}.pt",
                weights_only=True,
            )
            after = torch.load(
                arguments.trace_directory / f"{step:04d}-{after_phase}.pt",
                weights_only=True,
            )
            embeddings = (
                before["prefill_embeds"] if step == 0 else before["feedback_input"]
            ).to(device)
            positions = before["positions"].to(device).reshape(1, 1, -1)
            outputs = reference.model(
                inputs_embeds=embeddings.unsqueeze(0),
                position_ids=positions.expand(3, 1, -1),
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            logits = reference.codec_head(outputs.last_hidden_state[:, -1]).float()
            actual_logits = after["logits"].to(device).float()
            difference = logits - actual_logits
            comparison: dict[str, str | int | float | list[int]] = {
                "step": step,
                "logits_max_absolute_error": difference.abs().max().item(),
                "logits_relative_error": (difference.norm() / logits.norm()).item(),
                "reference_top_tokens": logits[0].topk(8).indices.tolist(),
                "actual_top_tokens": actual_logits[0].topk(8).indices.tolist(),
            }
            sampled = torch.load(
                arguments.trace_directory
                / f"{step:04d}-{'post_prefill' if step == 0 else 'post_decode'}.pt",
                weights_only=True,
            )
            codes = sampled["codec_ids"].to(device)
            native_hidden = sampled["hidden_states"].to(device).unsqueeze(1)
            first_embedding = reference.model.codec_embedding(codes[:, :1])
            predicted = reference.code_predictor.generate(
                inputs_embeds=torch.cat((native_hidden, first_embedding), dim=1),
                max_new_tokens=config.num_code_groups - 1,
                do_sample=False,
            )
            comparison["reference_residual_codes"] = predicted[0].tolist()
            comparison["actual_residual_codes"] = codes[0, 1:].tolist()
            embedding_rows = [first_embedding]
            for index, embedding in enumerate(
                reference.code_predictor.get_input_embeddings()
            ):
                embedding_rows.append(embedding(codes[:, index + 1 : index + 2]))
            feedback = torch.cat(embedding_rows, dim=1).sum(dim=1)
            actual_feedback = sampled["codec_feedback"].to(device)
            comparison["feedback_max_absolute_error"] = (
                (feedback - actual_feedback).abs().max().item()
            )
            comparisons.append(comparison)
            print(json.dumps(comparison), flush=True)
    arguments.output.write_text(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
