"""Run the checkpoint Talker with saved native conditioning and sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
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
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
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
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    assert not reference.code_predictor.generation_config.do_sample
    reference.eval()
    if arguments.dtype == "float32":
        reference.float()
    else:
        pass
    captures = {}
    for path in arguments.capture_directory.glob("request-*.pt"):
        capture = torch.load(path, map_location="cpu", weights_only=True)
        captures[capture["request_id"]] = capture
    reports = []
    with torch.inference_mode():
        for request_id in arguments.request_id:
            snapshot = captures[request_id]["snapshots"][0]
            conditioning = torch.load(
                arguments.capture_directory
                / "talker_state"
                / f"{request_id}.conditioning.pt",
                map_location="cpu",
                weights_only=True,
            )
            native_codes = torch.load(
                arguments.capture_directory / "talker_state" / f"{request_id}.pt",
                map_location="cpu",
                weights_only=True,
            )
            prefill = snapshot["prefill_input_embeddings"].to(
                reference.device, reference.dtype
            )
            positions = snapshot["prefill_positions"].to(reference.device)
            assert torch.equal(
                positions, torch.arange(len(prefill), device=reference.device)
            )
            text_rows = conditioning["future_text_rows"].to(
                reference.device, reference.dtype
            )
            pad_embedding = conditioning["pad_embedding"].to(
                reference.device, reference.dtype
            )
            generated_ids = snapshot["prefill_input_ids"][None].to(reference.device)
            top_k = torch.tensor([conditioning["top_k"]], device=reference.device)
            top_p = torch.tensor([conditioning["top_p"]], device=reference.device)
            minimum_probability = torch.zeros(1, device=reference.device)
            seed = torch.tensor(
                [conditioning["sampling_seed"]], device=reference.device
            )
            token_ids = torch.arange(
                config.text_config.vocab_size, device=reference.device
            )
            allowed = (token_ids < config.text_config.vocab_size - 1024) | (
                token_ids == config.codec_eos_token_id
            )
            repeated = torch.zeros_like(allowed)[None]
            prepared_inputs = {
                "inputs_embeds": prefill[None],
                "position_ids": positions.view(1, 1, -1).expand(4, 1, -1),
                "use_cache": True,
                "output_hidden_states": True,
            }
            frames = []
            termination = "max_new_tokens"
            for frame_index in range(conditioning["max_new_tokens"]):
                output = reference(**prepared_inputs)
                assert (
                    output.past_key_values.get_seq_length()
                    == len(prefill) + frame_index
                )
                assert output.generation_step == frame_index
                logits = output.logits[:, -1].float()
                penalty = conditioning["repetition_penalty"]
                penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
                logits = torch.where(repeated, penalized, logits).masked_fill(
                    ~allowed, float("-inf")
                )
                sampling_position = len(prefill) - 1 + frame_index
                sampled = top_k_top_p_min_p_sampling_from_probs_torch(
                    torch.softmax(logits / conditioning["temperature"], dim=-1),
                    top_k,
                    top_p,
                    minimum_probability,
                    False,
                    seed,
                    torch.tensor([sampling_position], device=reference.device),
                ).long()
                repeated[:, sampled] = True
                generated_ids = torch.cat((generated_ids, sampled[:, None]), dim=1)
                if int(sampled[0]) == config.codec_eos_token_id:
                    codes = torch.zeros(
                        (1, config.num_code_groups),
                        dtype=torch.long,
                        device=reference.device,
                    )
                    codes[:, 0] = sampled
                    termination = "codec_eos"
                    frames.append(codes[0].cpu())
                    break
                else:
                    prepared_inputs = reference.prepare_inputs_for_generation(
                        generated_ids,
                        next_sequence_length=1,
                        past_key_values=output.past_key_values,
                        attention_mask=torch.ones_like(generated_ids),
                        is_first_iteration=False,
                        position_ids=torch.tensor(
                            [[[sampling_position + 1]]], device=reference.device
                        ).expand(4, 1, 1),
                        hidden_states=output.hidden_states,
                        generation_step=output.generation_step,
                        trailing_text_hidden=text_rows[None],
                        tts_pad_embed=pad_embedding[None, None],
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    codes = prepared_inputs["residual_codes"]
                    assert int(codes[0, 0]) == int(sampled[0])
                    frames.append(codes[0].cpu())
            generated_codes = torch.stack(frames)
            compared = min(len(generated_codes), len(native_codes))
            main_differences = (
                (generated_codes[:compared, 0] != native_codes[:compared, 0])
                .nonzero()
                .flatten()
            )
            differences = (
                (generated_codes[:compared] != native_codes[:compared])
                .any(dim=1)
                .nonzero()
                .flatten()
            )
            torch.save(generated_codes, arguments.output_directory / f"{request_id}.pt")
            report = {
                "request_id": request_id,
                "dtype": arguments.dtype,
                "seed": conditioning["sampling_seed"],
                "prefill_tokens": len(prefill),
                "future_text_rows": len(text_rows),
                "termination": termination,
                "frames": len(generated_codes),
                "native_frames": len(native_codes),
                "first_main_difference": (
                    int(main_differences[0]) if len(main_differences) else None
                ),
                "first_codec_difference": (
                    int(differences[0]) if len(differences) else None
                ),
            }
            reports.append(report)
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
