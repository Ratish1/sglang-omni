# SPDX-License-Identifier: Apache-2.0
"""Compare real native speech conditioning with the checkpoint producer."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from sglang_omni.serve.protocol import ChatCompletionRequest


def prompt_hidden(
    model: Qwen3OmniMoeForConditionalGeneration,
    prompt_ids: torch.Tensor,
    audio_embeddings: torch.Tensor,
    feature_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = model.thinker.get_input_embeddings()(prompt_ids)
    audio_mask = prompt_ids == model.config.thinker_config.audio_token_id
    assert int(audio_mask.sum()) == len(audio_embeddings)
    embeddings[audio_mask] = audio_embeddings.to(embeddings)
    attention_mask = torch.ones_like(prompt_ids)
    positions = model.thinker.compute_3d_position_ids(
        input_ids=prompt_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        past_key_values=None,
        audio_feature_lengths=feature_mask.sum(dim=1),
    )
    output = model.thinker(
        input_ids=prompt_ids,
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        position_ids=positions,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return (
        output.hidden_states[model.config.talker_config.accept_hidden_layer],
        positions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--request-directory", type=Path, required=True)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--frontend-directory", type=Path)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    torch.set_float32_matmul_precision("highest")
    model, loading = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        arguments.model_path,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    assert not loading["unexpected_keys"], loading["unexpected_keys"]
    model.eval()
    processor = Qwen3OmniMoeProcessor.from_pretrained(arguments.model_path)
    reports = []
    with torch.inference_mode():
        for request_id in arguments.request_id:
            request_file = arguments.request_directory / f"{request_id}.request.json"
            request_bytes = request_file.read_bytes()
            request = ChatCompletionRequest.model_validate_json(request_bytes)
            assert request.request_id == request_id
            assert request.audios and len(request.audios) == 1
            messages = [
                message.model_dump(exclude_none=True) for message in request.messages
            ]
            assert messages[-1]["role"] == "user"
            assert isinstance(messages[-1]["content"], str)
            messages[-1]["content"] = [
                {"type": "audio", "path": request.audios[0]},
                {"type": "text", "text": messages[-1]["content"]},
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)
            directory = arguments.native_directory / request_id
            native = torch.load(
                directory / "conditioning.pt", map_location="cpu", weights_only=True
            )
            native.update(
                torch.load(
                    directory / "audio-conditioning.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            prompt_ids = inputs["input_ids"]
            assert torch.equal(prompt_ids[0].cpu(), native["prompt_input_ids"])
            audio_mask = prompt_ids[0] == model.config.thinker_config.audio_token_id
            native_hidden = native["prompt_hidden"].to(model.device)
            native_audio = native["audio_embeds"].to(model.device)
            checkpoint_audio = model.thinker.get_audio_features(
                inputs["input_features"],
                inputs["feature_attention_mask"],
                return_dict=True,
            ).last_hidden_state
            checkpoint_hidden, checkpoint_positions = prompt_hidden(
                model, prompt_ids, checkpoint_audio, inputs["feature_attention_mask"]
            )
            native_audio_hidden, native_audio_positions = prompt_hidden(
                model,
                prompt_ids,
                native_audio,
                native["feature_attention_mask"].to(model.device),
            )
            frontend_report = {}
            if arguments.frontend_directory is not None:
                frontend = torch.load(
                    arguments.frontend_directory / request_id / "audio-features.pt",
                    map_location=model.device,
                    weights_only=True,
                )
                assert torch.equal(
                    frontend["feature_attention_mask"].cpu(),
                    native["feature_attention_mask"],
                )
                native_features_audio = model.thinker.get_audio_features(
                    frontend["input_features"].to(dtype=model.dtype),
                    frontend["feature_attention_mask"],
                    return_dict=True,
                ).last_hidden_state
                frontend_report = {
                    "feature_difference": asdict(
                        compare_tensors(
                            frontend["input_features"].to(dtype=model.dtype),
                            inputs["input_features"],
                        )
                    ),
                    "encoder_with_identical_features": asdict(
                        compare_tensors(native_audio, native_features_audio)
                    ),
                }
            else:
                pass
            prefix_proofs = []
            for filename in sorted(directory.glob("thinker-prefill-*.pt")):
                producer = torch.load(filename, map_location="cpu", weights_only=True)
                prefix = int(producer["prefix_length"])
                length = len(producer["positions"])
                assert torch.equal(
                    producer["prompt_input_ids"][~audio_mask.cpu()],
                    native["prompt_input_ids"][~audio_mask.cpu()],
                )
                assert (
                    producer["prompt_input_ids"][audio_mask.cpu()].unique().numel() == 1
                )
                active_audio = audio_mask[prefix : prefix + length].cpu()
                assert torch.equal(
                    producer["selected_hidden"][active_audio],
                    native["prompt_hidden"][prefix : prefix + length][active_audio],
                )
                prefix_proofs.append(
                    {
                        "prefix_length": prefix,
                        "rows": length,
                        "selected_hidden_delivery_exact": True,
                        "mrope_positions_exact": torch.equal(
                            producer["mrope_positions"],
                            native_audio_positions[
                                :, 0, prefix : prefix + length
                            ].cpu(),
                        ),
                    }
                )
            assert prefix_proofs
            positions = torch.arange(prompt_ids.shape[1], device=model.device)[None]
            starts = torch.where(
                prompt_ids == model.config.im_start_token_id,
                positions,
                positions.new_full((), -1),
            )
            role_positions = (starts.cummax(dim=-1).values.clamp_min(0) + 1).clamp_max(
                prompt_ids.shape[1] - 1
            )
            user_mask = (
                prompt_ids.gather(1, role_positions)[0] == model.config.user_token_id
            )
            user_ids = prompt_ids[0, user_mask]
            assert torch.equal(
                native["prefill_input_ids"][: len(user_ids)], user_ids.cpu()
            )
            user_audio = user_ids == model.config.thinker_config.audio_token_id
            assert int(user_audio.sum()) == int(audio_mask.sum())
            audio_rows = user_audio.nonzero(as_tuple=True)[0]
            native_prefill = native["prefill_embeddings"].to(model.device)
            checkpoint_projection = model.talker.hidden_projection(
                checkpoint_hidden[0, audio_mask]
            )
            native_audio_projection = model.talker.hidden_projection(
                native_audio_hidden[0, audio_mask]
            )
            native_projection = model.talker.hidden_projection(
                native_hidden[audio_mask]
            )
            checkpoint_prefill = native_prefill.clone()
            checkpoint_prefill[audio_rows] = checkpoint_projection
            native_audio_prefill = native_prefill.clone()
            native_audio_prefill[audio_rows] = native_audio_projection
            report = {
                "request_id": request_id,
                "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
                "accept_hidden_layer": model.config.talker_config.accept_hidden_layer,
                "prompt_tokens": prompt_ids.shape[1],
                "audio_rows": len(audio_rows),
                "producer_delivery": prefix_proofs,
                "audio_encoder": asdict(
                    compare_tensors(native_audio, checkpoint_audio)
                ),
                "frontend_control": frontend_report,
                "hidden_with_checkpoint_audio": asdict(
                    compare_tensors(
                        native_hidden[audio_mask], checkpoint_hidden[0, audio_mask]
                    )
                ),
                "hidden_with_native_audio": asdict(
                    compare_tensors(
                        native_hidden[audio_mask], native_audio_hidden[0, audio_mask]
                    )
                ),
                "native_projection_reproduction": asdict(
                    compare_tensors(native_prefill[audio_rows], native_projection)
                ),
                "checkpoint_positions_match_native_audio_positions": torch.equal(
                    checkpoint_positions, native_audio_positions
                ),
            }
            torch.save(
                {
                    "prompt_input_ids": prompt_ids.cpu(),
                    "checkpoint_audio": checkpoint_audio.cpu(),
                    "checkpoint_hidden": checkpoint_hidden.cpu(),
                    "native_audio_reference_hidden": native_audio_hidden.cpu(),
                    "checkpoint_audio_prefill": checkpoint_prefill.cpu(),
                    "native_audio_prefill": native_audio_prefill.cpu(),
                    "native_prefill": native_prefill.cpu(),
                    "audio_rows": audio_rows.cpu(),
                },
                arguments.output_directory / f"{request_id}.pt",
            )
            reports.append(report)
            (arguments.output_directory / "results.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
