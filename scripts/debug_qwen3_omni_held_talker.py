# SPDX-License-Identifier: Apache-2.0
"""Compare native and checkpoint Talker on the same recorded codec histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from transformers import Qwen3OmniMoeConfig
from transformers.cache_utils import DynamicCache
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


def filtered_distribution(
    logits: torch.Tensor, sampling: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, int]:
    penalties = sampling["repetition_penalty"]
    penalized = torch.where(logits > 0, logits / penalties, logits * penalties)
    processed = torch.where(sampling["repetition_mask"], penalized, logits)
    masked = processed.masked_fill(sampling["suppress_mask"], float("-inf"))
    probabilities = (masked / sampling["temperature"]).softmax(-1).view(1, -1)
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
    distribution = torch.zeros_like(weights).scatter(1, permutation.long(), weights)
    return distribution / distribution.sum(), int(selected[0])


def replay_history(
    reference: Qwen3OmniMoeTalkerForConditionalGeneration,
    trace_directory: Path,
) -> list[dict[str, int | float | bool | list[int] | list[float] | None]]:
    target_device = reference.device
    prefill = torch.load(
        trace_directory / "backbone-prefill-000.pt",
        map_location=target_device,
        weights_only=True,
    )
    finish = torch.load(
        trace_directory / "backbone-finish-000.pt",
        map_location=target_device,
        weights_only=True,
    )
    assert finish["thinker_done"]
    codec_files = sorted(trace_directory.glob("codec-*.pt"))
    assert len(codec_files) > 1
    assert (
        len(list(trace_directory.glob("backbone-decode-*.pt"))) == len(codec_files) - 1
    )
    assert (
        len(list(trace_directory.glob("decode-sampling-*.pt"))) == len(codec_files) - 1
    )
    cache = None
    frames = []
    previous_feedback = None
    for frame_index, codec_file in enumerate(codec_files):
        assert codec_file.name == f"codec-{frame_index:03d}.pt"
        observed = torch.load(codec_file, map_location=target_device, weights_only=True)
        codec_ids = observed["codec_ids"].reshape(1, -1).long()
        assert observed["active_rows"] == 1
        assert codec_ids.shape[1] == reference.config.num_code_groups
        main_codec_token = int(codec_ids[0, 0])
        is_eos = main_codec_token == reference.config.codec_eos_token_id
        assert is_eos == (frame_index == len(codec_files) - 1)
        if frame_index == 0:
            input_embeddings = prefill["embeddings"].unsqueeze(0)
            positions = prefill["positions"].reshape(-1)
            assert torch.equal(
                positions, torch.arange(len(positions), device=target_device)
            )
            rotary_positions = prefill["rotary_positions"]
            sampling = torch.load(
                trace_directory / "prefill-sampling-000.pt",
                map_location=target_device,
                weights_only=True,
            )
            assert observed["consumed_decode_embedding"] is None
        else:
            recorded_input = torch.load(
                trace_directory / f"backbone-decode-{frame_index:03d}.pt",
                map_location=target_device,
                weights_only=True,
            )
            sampling = torch.load(
                trace_directory / f"decode-sampling-{frame_index:03d}.pt",
                map_location=target_device,
                weights_only=True,
            )
            positions = recorded_input["position"].reshape(1)
            rotary_positions = recorded_input["rotary_positions"]
            assert (
                positions.item()
                == cache.get_seq_length()
                == sampling["position"].item()
            )
            assert main_codec_token == sampling["sampled_code"].item()
            assert torch.equal(
                previous_feedback, recorded_input["feedback_row"].reshape(-1)
            )
            text_row = recorded_input["text_row"].reshape(-1)
            if recorded_input["uses_pad"]:
                assert torch.equal(text_row, finish["pad_embedding"])
            else:
                pass
            combined = previous_feedback + text_row
            assert torch.equal(
                combined, observed["consumed_decode_embedding"].reshape(-1)
            )
            input_embeddings = combined.view(1, 1, -1)
        assert input_embeddings.dtype == reference.dtype
        linear_positions = positions.view(1, 1, -1)
        if rotary_positions is None:
            position_ids = linear_positions.expand(4, 1, -1)
        else:
            position_ids = torch.cat(
                (linear_positions, rotary_positions.view(3, 1, -1))
            )
        result = reference.model(
            inputs_embeds=input_embeddings,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        cache = result.past_key_values
        assert cache.get_seq_length() == len(prefill["positions"]) + frame_index
        reference_query = result.last_hidden_state[:, -1:]
        native_query = sampling["predictor_query"].view(1, 1, -1)
        reference_logits = reference.codec_head(reference_query).float().reshape(-1)
        native_logits = sampling["masked_logits" if frame_index == 0 else "raw_logits"]
        native_head_logits = reference.codec_head(native_query).float().reshape(-1)
        finite = native_logits.isfinite()
        assert finite[main_codec_token] and reference_logits.isfinite().all()
        difference = native_logits[finite] - reference_logits[finite]
        report = {
            "frame": frame_index,
            "native_code": main_codec_token,
            "eos": is_eos,
            "maximum_absolute_logit_difference": float(difference.abs().max()),
            "relative_logit_l2": float(
                difference.norm() / reference_logits[finite].norm()
            ),
            "head_reprojection_maximum_difference": float(
                (native_logits[finite] - native_head_logits[finite]).abs().max()
            ),
            "filtered_distribution_total_variation": None,
            "reference_sample": None,
            "reference_probability_of_native_code": None,
        }
        if frame_index > 0:
            native_distribution, repeated_native_code = filtered_distribution(
                native_logits, sampling
            )
            assert repeated_native_code == main_codec_token
            reference_distribution, reference_code = filtered_distribution(
                reference_logits, sampling
            )
            report.update(
                filtered_distribution_total_variation=float(
                    0.5 * (native_distribution - reference_distribution).abs().sum()
                ),
                reference_sample=reference_code,
                reference_probability_of_native_code=float(
                    reference_distribution[0, main_codec_token]
                ),
            )
        else:
            pass
        if is_eos:
            frames.append(report)
            break
        else:
            pass
        code_embeddings = [reference.model.codec_embedding(codec_ids[:, :1])]
        for group_index, embedding in enumerate(
            reference.code_predictor.model.codec_embedding
        ):
            code_embeddings.append(
                embedding(codec_ids[:, group_index + 1 : group_index + 2])
            )
        feedback = code_embeddings[0]
        for code_embedding in code_embeddings[1:]:
            feedback = feedback + code_embedding
        previous_feedback = observed["feedback_embedding"].reshape(-1)
        report["same_held_codes_feedback_exact"] = torch.equal(
            feedback.reshape(-1), previous_feedback
        )
        for source_name, predictor_query in (
            ("native_query", native_query),
            ("reference_query", reference_query),
        ):
            predictor_inputs = torch.cat(
                (predictor_query, *code_embeddings[:-1]), dim=1
            )
            hidden_states = reference.code_predictor.model(
                inputs_embeds=predictor_inputs, use_cache=False
            ).last_hidden_state
            residual_logits = torch.stack(
                [
                    head(hidden_states[:, group_index + 1]).float()
                    for group_index, head in enumerate(reference.code_predictor.lm_head)
                ],
                dim=1,
            )
            selected_logits = residual_logits.gather(-1, codec_ids[:, 1:, None])
            report[f"{source_name}_residual_ranks"] = (
                1 + (residual_logits > selected_logits).sum(-1)
            )[0].tolist()
            report[f"{source_name}_residual_logit_gaps"] = (
                residual_logits.max(-1).values - selected_logits.squeeze(-1)
            )[0].tolist()
        frames.append(report)
    return frames


def compare_prefix_cache(
    reference: Qwen3OmniMoeTalkerForConditionalGeneration,
    trace_directory: Path,
) -> dict[str, int | list[dict[str, int | float | str]]]:
    target_device = reference.device
    recorded = {
        name: torch.load(
            trace_directory / f"{name}.pt",
            map_location=target_device,
            weights_only=True,
        )
        for name in (
            "backbone-prefill-000",
            "backbone-decode-001",
            "decode-sampling-001",
            "first-decode-prefix-cache-000",
            "codec-001",
        )
    }
    prefill = recorded["backbone-prefill-000"]
    decode = recorded["backbone-decode-001"]
    sampling = recorded["decode-sampling-001"]
    native_prefix = recorded["first-decode-prefix-cache-000"]
    native_code = int(sampling["sampled_code"])
    position = sampling["position"].reshape(1, 1, 1)
    assert (
        decode["rotary_positions"] is None
        or (decode["rotary_positions"] == position).all()
    )
    assert int(position) == native_prefix["position"] == len(prefill["positions"])
    prefill_positions = prefill["positions"].reshape(1, 1, -1)
    assert prefill["rotary_positions"] is None or torch.equal(
        prefill["rotary_positions"], prefill_positions.reshape(1, -1).expand(3, -1)
    )
    reference_cache = reference.model(
        inputs_embeds=prefill["embeddings"].unsqueeze(0),
        position_ids=prefill_positions.expand(4, 1, -1),
        use_cache=True,
    ).past_key_values
    assert reference_cache.get_seq_length() == len(prefill["positions"])
    decode_embeddings = decode["feedback_row"].reshape(-1) + decode["text_row"].reshape(
        -1
    )
    assert torch.equal(
        decode_embeddings,
        recorded["codec-001"]["consumed_decode_embedding"].reshape(-1),
    )
    native_logits = sampling["raw_logits"]
    native_distribution, repeated_code = filtered_distribution(native_logits, sampling)
    assert native_code == repeated_code
    results = []
    for source in ("reference", "native_keys", "native_values", "native_both"):
        cache = DynamicCache(config=reference.config.text_config)
        for layer_index, reference_layer in enumerate(reference_cache.layers):
            native_keys = (
                native_prefix["key_rows"][layer_index].transpose(0, 1).unsqueeze(0)
            )
            native_values = (
                native_prefix["value_rows"][layer_index].transpose(0, 1).unsqueeze(0)
            )
            assert native_keys.shape == reference_layer.keys.shape
            assert native_values.shape == reference_layer.values.shape
            assert native_keys.dtype == native_values.dtype == reference.dtype
            cache.update(
                (
                    native_keys
                    if source in ("native_keys", "native_both")
                    else reference_layer.keys
                ),
                (
                    native_values
                    if source in ("native_values", "native_both")
                    else reference_layer.values
                ),
                layer_index,
            )
        result = reference.model(
            inputs_embeds=decode_embeddings.view(1, 1, -1),
            position_ids=position.expand(4, 1, 1),
            past_key_values=cache,
            use_cache=True,
        )
        logits = (
            reference.codec_head(result.last_hidden_state[:, -1:]).float().reshape(-1)
        )
        distribution, code = filtered_distribution(logits, sampling)
        results.append(
            {
                "prefill_cache_source": source,
                "selected_code": code,
                "probability_of_native_code": float(distribution[0, native_code]),
                "total_variation_from_native_distribution": float(
                    0.5 * (native_distribution - distribution).abs().sum()
                ),
                "maximum_absolute_logit_difference_from_native": float(
                    (logits - native_logits).abs().max()
                ),
            }
        )
    return {"native_code": native_code, "position": int(position), "arms": results}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--trace-directory", type=Path)
    mode.add_argument("--prefix-trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"], loading
    reference.eval()
    if arguments.prefix_trace is not None:
        reports = compare_prefix_cache(reference, arguments.prefix_trace)
        assert reports == compare_prefix_cache(reference, arguments.prefix_trace)
    else:
        reports = {}
        for trace_directory in sorted(arguments.trace_directory.iterdir()):
            frames = replay_history(reference, trace_directory)
            repeated_frames = replay_history(reference, trace_directory)
            assert frames == repeated_frames
            reports[trace_directory.name] = frames
    with arguments.output.open("x") as output:
        json.dump(reports, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
else:
    pass
