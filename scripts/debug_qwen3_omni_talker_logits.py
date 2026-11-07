# SPDX-License-Identifier: Apache-2.0
"""Replay captured Talker inputs through the reference backbone without sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-directory", action="append", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--max-decode-frames", type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    reports = []
    with torch.inference_mode():
        for directory in arguments.trace_directory:
            prefill_file = next(directory.glob("*-before_prefill.pt"))
            before = torch.load(prefill_file, map_location="cpu", weights_only=True)
            prefill = before["prefill_embeds"]
            assert torch.equal(before["positions"], torch.arange(len(prefill)))
            before_files = [prefill_file, *sorted(directory.glob("*-before_decode.pt"))]
            if arguments.max_decode_frames is not None:
                before_files = before_files[: arguments.max_decode_frames + 1]
            else:
                pass
            cache = None
            frames = []
            for frame_index, before_file in enumerate(before_files):
                captured = torch.load(
                    before_file, map_location="cpu", weights_only=True
                )
                phase = "post_prefill" if frame_index == 0 else "post_decode"
                after_file = before_file.with_name(
                    f"{before_file.name.split('-')[0]}-{phase}.pt"
                )
                native = torch.load(after_file, map_location="cpu", weights_only=True)
                embeddings = (
                    prefill[None]
                    if frame_index == 0
                    else captured["feedback_input"][:, None]
                ).to(reference.device)
                positions = captured["positions"].to(reference.device)
                if frame_index > 0:
                    assert positions.tolist() == [cache.get_seq_length()]
                else:
                    pass
                output = reference.model(
                    inputs_embeds=embeddings,
                    position_ids=positions.view(1, 1, -1).expand(4, 1, -1),
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values
                expected = reference.codec_head(output.last_hidden_state[:, -1]).float()
                actual = native["logits"].to(reference.device).float()
                native_hidden_states = native["hidden_states"].to(reference.device)
                reprojected = reference.codec_head(native_hidden_states).float()
                if frame_index == 0:
                    # note (ratish): the prefill sampler overwrites its logits with probabilities.
                    actual = reprojected
                    repetition_mask = torch.zeros_like(actual, dtype=torch.bool)
                else:
                    repetition_mask = captured["repetition_mask"].to(reference.device)
                finite = torch.isfinite(actual)
                assert torch.isfinite(expected).all() and finite.any()
                difference = actual[finite] - expected[finite]
                token_ids = torch.arange(expected.shape[-1], device=expected.device)
                allowed = (token_ids < config.text_config.vocab_size - 1024) | (
                    token_ids == config.codec_eos_token_id
                )
                distributions = []
                for logits in (actual, expected):
                    penalized = torch.where(
                        logits > 0,
                        logits / arguments.repetition_penalty,
                        logits * arguments.repetition_penalty,
                    )
                    scaled = torch.where(repetition_mask, penalized, logits)
                    distributions.append(
                        torch.log_softmax(
                            scaled[:, allowed] / arguments.temperature, dim=-1
                        )
                    )
                native_distribution, reference_distribution = distributions
                chosen_token = int(native["codec_ids"][0, 0])
                allowed_ids = token_ids[allowed]
                chosen_column = int((allowed_ids == chosen_token).nonzero()[0])
                codes = native["codec_ids"].to(reference.device)
                predictor_embeddings = [
                    native_hidden_states[:, None],
                    reference.model.codec_embedding(codes[:, :1]),
                ]
                for group in range(config.num_code_groups - 2):
                    predictor_embeddings.append(
                        reference.code_predictor.model.codec_embedding[group](
                            codes[:, group + 1 : group + 2]
                        )
                    )
                predictor_output = reference.code_predictor.model(
                    inputs_embeds=torch.cat(predictor_embeddings, dim=1),
                    use_cache=False,
                ).last_hidden_state
                predictor_logits = torch.stack(
                    [
                        head(predictor_output[:, group + 1])
                        for group, head in enumerate(reference.code_predictor.lm_head)
                    ],
                    dim=1,
                ).float()
                assert torch.isfinite(predictor_logits).all()
                selected_logits = predictor_logits.gather(
                    -1, codes[:, 1:, None]
                ).squeeze(-1)
                predictor_argmax = predictor_logits.argmax(dim=-1)
                predictor_ranks = (predictor_logits > selected_logits[:, :, None]).sum(
                    dim=-1
                ) + 1
                report = {
                    "trace_file": after_file.name,
                    "native_logits_source": (
                        "reprojected_prefill_hidden_states"
                        if frame_index == 0
                        else "captured_decode_logits"
                    ),
                    "hidden_state_relative_l2_error": float(
                        (
                            native_hidden_states.float()
                            - output.last_hidden_state[:, -1].float()
                        ).norm()
                        / output.last_hidden_state[:, -1].float().norm()
                    ),
                    "head_reprojection_matches_captured": (
                        None if frame_index == 0 else torch.equal(actual, reprojected)
                    ),
                    "maximum_absolute_logit_error": float(difference.abs().max()),
                    "relative_logit_l2_error": float(
                        difference.norm() / expected[finite].norm()
                    ),
                    "pre_topk_kl_native_to_reference": float(
                        (
                            native_distribution.exp()
                            * (native_distribution - reference_distribution)
                        ).sum()
                    ),
                    "chosen_token": chosen_token,
                    "chosen_native_probability_before_topk": float(
                        native_distribution[0, chosen_column].exp()
                    ),
                    "chosen_reference_probability_before_topk": float(
                        reference_distribution[0, chosen_column].exp()
                    ),
                    "chosen_reference_rank": int(
                        (
                            reference_distribution[0]
                            > reference_distribution[0, chosen_column]
                        ).sum()
                    )
                    + 1,
                    "native_residual_codes": codes[0, 1:].tolist(),
                    "reference_predictor_argmax": predictor_argmax[0].tolist(),
                    "native_residual_reference_ranks": predictor_ranks[0].tolist(),
                    "native_residual_reference_logit_gaps": (
                        predictor_logits.max(dim=-1).values - selected_logits
                    )[0].tolist(),
                }
                frames.append(report)
            report = {"trace_directory": str(directory), "frames": frames}
            reports.append(report)
            (arguments.output_directory / "results.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(
                json.dumps({"trace_directory": str(directory), "frames": len(frames)}),
                flush=True,
            )


if __name__ == "__main__":
    main()
