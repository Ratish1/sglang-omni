# SPDX-License-Identifier: Apache-2.0
"""Generate checkpoint Talker codes with captured conditioning and native sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import xxhash
from huggingface_hub import hf_hub_download
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    load_thinker_embedding_rows,
)
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--trace-directory", action="append", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-frames", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    model_config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path)
    config = model_config.talker_config
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
    assert not reference.code_predictor.generation_config.do_sample
    reference.eval()
    checkpoint_directory = Path(
        hf_hub_download(arguments.model_path, "config.json")
    ).parent
    special_embeddings = load_thinker_embedding_rows(
        str(checkpoint_directory),
        [
            model_config.tts_bos_token_id,
            model_config.tts_eos_token_id,
            model_config.tts_pad_token_id,
        ],
    ).to(device=reference.device, dtype=torch.bfloat16)
    reports = []
    with torch.inference_mode():
        projected_special_embeddings = reference.text_projection(special_embeddings)
        eos_embedding = projected_special_embeddings[1:2]
        pad_embedding = projected_special_embeddings[2:3]
        token_ids = torch.arange(config.text_config.vocab_size, device=reference.device)
        allowed_tokens = (token_ids < config.text_config.vocab_size - 1024) | (
            token_ids == config.codec_eos_token_id
        )
        top_k = torch.tensor([arguments.top_k], device=reference.device)
        top_p = torch.tensor([arguments.top_p], device=reference.device)
        minimum_probability = torch.zeros(1, device=reference.device)
        for trace_directory in arguments.trace_directory:
            prefill_file = next(trace_directory.glob("*-before_prefill.pt"))
            before = torch.load(prefill_file, map_location="cpu", weights_only=True)
            prefill = before["prefill_embeds"].to(reference.device)
            text_rows = before["trailing_text_embeds"].to(reference.device)
            assert torch.equal(
                before["positions"], torch.arange(len(prefill))
            ), "this experiment requires captured linear audio-only positions"
            assert torch.equal(text_rows[-1:], eos_embedding), "incomplete text queue"
            native_files = sorted(trace_directory.glob("*-post_*.pt"))
            native_frames = []
            for filename in native_files:
                captured = torch.load(filename, map_location="cpu", weights_only=True)
                native_frames.append(
                    {
                        name: captured[name]
                        for name in ("codec_ids", "codec_feedback", "logits")
                    }
                )
            seed = (
                xxhash.xxh64_intdigest(trace_directory.name.encode())
                & SAMPLING_SEED_MASK
            )
            sampling_seed = torch.tensor([seed], device=reference.device)
            checked_feedback_rows = 0
            for frame_index, filename in enumerate(
                sorted(trace_directory.glob("*-before_decode.pt"))
            ):
                captured = torch.load(filename, map_location="cpu", weights_only=True)
                assert captured["sampling_seed"].tolist() == [seed]
                assert captured["positions"].tolist() == [len(prefill) + frame_index]
                next_text = (
                    text_rows[frame_index : frame_index + 1]
                    if frame_index < len(text_rows)
                    else pad_embedding
                )
                previous_feedback = native_frames[frame_index]["codec_feedback"].to(
                    reference.device
                )
                assert torch.equal(
                    previous_feedback + next_text,
                    captured["feedback_input"].to(reference.device),
                ), f"uncaptured text handoff at {filename}"
                checked_feedback_rows += 1
            captured_prefill_probabilities = native_frames[0]["logits"].to(
                reference.device
            )
            reproduced_prefill_choice = top_k_top_p_min_p_sampling_from_probs_torch(
                captured_prefill_probabilities,
                top_k,
                top_p,
                minimum_probability,
                False,
                sampling_seed,
                before["positions"][-1:].to(reference.device),
            )
            assert int(reproduced_prefill_choice[0]) == int(
                native_frames[0]["codec_ids"][0, 0]
            )
            case_directory = (
                arguments.output_directory
                / trace_directory.parent.name
                / trace_directory.name
            )
            case_directory.mkdir(parents=True)
            generated_ids = before["input_ids"][None].to(reference.device)
            prepared_inputs = {
                "inputs_embeds": prefill[None],
                "position_ids": before["positions"]
                .to(reference.device)
                .view(1, 1, -1)
                .expand(4, 1, -1),
                "use_cache": True,
                "output_hidden_states": True,
            }
            repeated_tokens = torch.zeros_like(token_ids, dtype=torch.bool)[None]
            first_codec_divergence = None
            first_main_code_divergence = None
            frame_reports = []
            termination = "frame_limit"
            for frame_index in range(arguments.max_frames):
                output = reference(**prepared_inputs)
                assert (
                    output.past_key_values.get_seq_length()
                    == len(prefill) + frame_index
                )
                assert output.generation_step == frame_index
                logits = output.logits[:, -1].float()
                penalized_logits = torch.where(
                    logits > 0,
                    logits / arguments.repetition_penalty,
                    logits * arguments.repetition_penalty,
                )
                sampling_logits = torch.where(
                    repeated_tokens, penalized_logits, logits
                ).masked_fill(~allowed_tokens, float("-inf"))
                sampling_position = len(prefill) - 1 + frame_index
                sampled_code = top_k_top_p_min_p_sampling_from_probs_torch(
                    torch.softmax(sampling_logits / arguments.temperature, dim=-1),
                    top_k,
                    top_p,
                    minimum_probability,
                    False,
                    sampling_seed,
                    torch.tensor([sampling_position], device=reference.device),
                ).long()
                repeated_tokens[:, sampled_code] = True
                is_eos = int(sampled_code[0]) == config.codec_eos_token_id
                generated_ids = torch.cat((generated_ids, sampled_code[:, None]), dim=1)
                if is_eos:
                    codes = torch.zeros(
                        (1, config.num_code_groups),
                        dtype=torch.long,
                        device=reference.device,
                    )
                    codes[:, 0] = sampled_code
                    termination = "codec_eos"
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
                        tts_pad_embed=pad_embedding[None],
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    codes = prepared_inputs["residual_codes"]
                    assert int(codes[0, 0]) == int(sampled_code[0])
                    assert prepared_inputs["inputs_embeds"].shape == (
                        1,
                        1,
                        prefill.shape[-1],
                    )
                if frame_index < len(native_frames):
                    native_codes = native_frames[frame_index]["codec_ids"].to(
                        reference.device
                    )
                    same_main_code = int(native_codes[0, 0]) == int(sampled_code[0])
                    same_codec_frame = torch.equal(native_codes, codes)
                    if not same_main_code and first_main_code_divergence is None:
                        first_main_code_divergence = frame_index
                    else:
                        pass
                    if not same_codec_frame and first_codec_divergence is None:
                        first_codec_divergence = frame_index
                    else:
                        pass
                else:
                    same_main_code = None
                    same_codec_frame = None
                phase = "post_prefill" if frame_index == 0 else "post_decode"
                torch.save(
                    {
                        "codec_ids": codes.cpu(),
                        "hidden_states": output.hidden_states[0][-1][:, -1].cpu(),
                        "logits": logits.cpu(),
                    },
                    case_directory / f"{frame_index:04d}-{phase}.pt",
                )
                frame_reports.append(
                    {
                        "frame_index": frame_index,
                        "sampling_position": sampling_position,
                        "main_code": int(sampled_code[0]),
                        "same_native_main_code": same_main_code,
                        "same_native_codec_frame": same_codec_frame,
                    }
                )
                if is_eos:
                    break
                else:
                    pass
            report = {
                "native_trace_directory": str(trace_directory),
                "reference_trace_directory": str(case_directory),
                "prefill_file_sha256": hashlib.sha256(
                    prefill_file.read_bytes()
                ).hexdigest(),
                "verified_sampling_seed": seed,
                "captured_feedback_rows_verified": checked_feedback_rows,
                "complete_text_queue_rows": len(text_rows),
                "first_codec_divergence": first_codec_divergence,
                "first_main_code_divergence": first_main_code_divergence,
                "native_frames_including_eos": len(native_frames),
                "reference_frames_including_eos": len(frame_reports),
                "termination": termination,
                "frames": frame_reports,
            }
            reports.append(report)
            (arguments.output_directory / "results.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(
                json.dumps(
                    {name: value for name, value in report.items() if name != "frames"}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
