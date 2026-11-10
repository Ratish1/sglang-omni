"""Check complete captured speech conditioning against checkpoint construction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.debug_qwen3_omni_conditioning import prompt_hidden
from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.debug_qwen3_omni_prefix_provenance import recover_audio_prefix
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights
from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    load_thinker_embedding_rows,
)
from sglang_omni.serve.protocol import ChatCompletionRequest


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--request-directory", type=Path)
    parser.add_argument("--thinker-directory", type=Path)
    parser.add_argument("--cached-prefix-proof", type=Path)
    parser.add_argument("--audit-loaded-weights", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    register_checkpoint_conversion_mapping(
        config.talker_config.model_type,
        get_checkpoint_conversion_mapping("qwen3_omni_moe"),
    )
    if arguments.request_directory is not None:
        owner, loading = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            arguments.model_path,
            revision=arguments.model_revision,
            config=config,
            dtype=torch.bfloat16,
            device_map={"": arguments.gpu},
            attn_implementation="sdpa",
            experts_implementation="eager",
            use_kernels=False,
            output_loading_info=True,
        )
        talker = owner.talker
        processor = Qwen3OmniMoeProcessor.from_pretrained(
            arguments.model_path, revision=arguments.model_revision
        )
    else:
        with torch.device("meta"):
            owner = Qwen3OmniMoeForConditionalGeneration(config)
        talker, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
            arguments.model_path,
            revision=arguments.model_revision,
            config=config.talker_config,
            dtype=torch.bfloat16,
            device_map={"": arguments.gpu},
            attn_implementation="sdpa",
            experts_implementation="eager",
            use_kernels=False,
            output_loading_info=True,
        )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    owner.talker = talker.eval()
    owner.eval()
    checkpoint_directory = Path(
        hf_hub_download(
            arguments.model_path, "config.json", revision=arguments.model_revision
        )
    ).parent
    special_ids = [
        config.tts_bos_token_id,
        config.tts_eos_token_id,
        config.tts_pad_token_id,
    ]
    special_embeddings = load_thinker_embedding_rows(
        str(checkpoint_directory), special_ids
    ).to(talker.device, dtype=talker.dtype)
    bos, eos, pad = talker.text_projection(special_embeddings[None]).chunk(3, dim=1)
    reports = []
    for request_id in arguments.request_id:
        directory = arguments.capture_directory / "conditioning" / request_id
        prompt = torch.load(directory / "prompt.pt", weights_only=True)
        before = torch.load(directory / "0000-before_prefill.pt", weights_only=True)
        finished = torch.load(directory / "finished.pt", weights_only=True)
        assert bool(finished["thinker_done"])
        prompt_ids = prompt["prompt_input_ids"].to(talker.device)
        assistant_ids = finished["assistant_token_ids"].to(talker.device)
        if int(assistant_ids[-1]) == config.im_end_token_id:
            assistant_ids = assistant_ids[:-1]
        else:
            pass
        all_ids = torch.cat([prompt_ids, assistant_ids])
        embeddings = load_thinker_embedding_rows(
            str(checkpoint_directory), all_ids.tolist()
        ).to(talker.device, dtype=talker.dtype)[None]
        hidden = torch.cat(
            [
                prompt["prompt_hidden"].to(talker.device),
                embeddings[0, len(prompt_ids) :],
            ]
        )[None]
        multimodal = torch.zeros_like(all_ids, dtype=torch.bool)
        for token_id in (
            config.thinker_config.audio_token_id,
            config.thinker_config.image_token_id,
            config.thinker_config.video_token_id,
        ):
            multimodal |= all_ids == token_id
        positions = torch.arange(len(prompt_ids), device=talker.device)
        starts = torch.where(prompt_ids == config.im_start_token_id, positions, -1)
        role_positions = (starts.cummax(0).values.clamp_min(0) + 1).clamp_max(
            len(prompt_ids) - 1
        )
        user_mask = prompt_ids[role_positions] == config.user_token_id
        user = owner._get_talker_user_parts(
            0, len(prompt_ids), multimodal[None], hidden, embeddings
        )[:, user_mask]
        assistant_starts = (
            (prompt_ids[:-1] == config.im_start_token_id)
            & (prompt_ids[1:] == config.assistant_token_id)
        ).nonzero(as_tuple=True)[0]
        assistant, assistant_input_ids, future_text = owner._get_talker_assistant_parts(
            int(assistant_starts[-1]),
            len(all_ids),
            int(prompt["speaker_id"]),
            embeddings,
            pad,
            bos,
            eos,
        )
        expected_prefix = torch.cat([user, assistant], dim=1)[0]
        expected_ids = torch.cat([prompt_ids[user_mask], assistant_input_ids[0]])
        assert torch.equal(expected_ids.cpu(), before["input_ids"])
        consumed = [
            record["text_input"]
            for filename in sorted(directory.glob("*-before_decode.pt"))
            for record in [torch.load(filename, weights_only=True)]
            if bool(record["queued_text"])
        ]
        consumed.extend(finished["pending_text"])
        native_future = torch.stack(consumed).to(talker.device)
        report = {
            "request_id": request_id,
            "model_revision": arguments.model_revision,
            "prompt_rows": len(prompt_ids),
            "assistant_token_ids": assistant_ids.cpu().tolist(),
            "user_rows": int(user_mask.sum()),
            "multimodal_rows": int(multimodal.sum()),
            "input_ids_exact": True,
            "prefix": asdict(
                compare_tensors(
                    before["prefill_embeds"].to(talker.device), expected_prefix
                )
            ),
            "future_text": asdict(compare_tensors(native_future, future_text[0])),
            "scope": "Checkpoint input methods with actual native prompt hidden rows. Does not independently validate the multimodal Thinker hidden producer.",
        }
        saved_tensors = {
            "checkpoint_prefix": expected_prefix.cpu(),
            "checkpoint_future_text": future_text[0].cpu(),
            "native_prefix": before["prefill_embeds"],
            "native_future_text": native_future.cpu(),
        }
        if arguments.request_directory is not None:
            request = ChatCompletionRequest.model_validate_json(
                (
                    arguments.request_directory / f"{request_id}.request.json"
                ).read_bytes()
            )
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
            ).to(talker.device, dtype=talker.dtype)
            assert torch.equal(inputs["input_ids"][0], prompt_ids)
            audio_embeddings = owner.thinker.get_audio_features(
                inputs["input_features"],
                inputs["feature_attention_mask"],
                return_dict=True,
            ).last_hidden_state
            checkpoint_hidden, checkpoint_positions = prompt_hidden(
                owner,
                inputs["input_ids"],
                audio_embeddings,
                inputs["feature_attention_mask"],
            )
            audio_mask = prompt_ids == config.thinker_config.audio_token_id
            projected_audio = talker.hidden_projection(checkpoint_hidden[0, audio_mask])
            checkpoint_audio_prefill = (
                before["prefill_embeds"].to(talker.device).clone()
            )
            audio_rows = audio_mask[user_mask].nonzero(as_tuple=True)[0]
            checkpoint_audio_prefill[audio_rows] = projected_audio
            report["independent_audio_hidden"] = asdict(
                compare_tensors(
                    hidden[0, : len(prompt_ids)][audio_mask],
                    checkpoint_hidden[0, audio_mask],
                )
            )
            if arguments.thinker_directory is not None:
                prefix_inputs = [
                    (
                        int(filename.stem.removeprefix("thinker-input-")),
                        torch.load(filename, weights_only=True),
                    )
                    for filename in sorted(
                        (arguments.thinker_directory / request_id).glob(
                            "thinker-input-*.pt"
                        )
                    )
                ]
                actual_prefix = torch.empty_like(embeddings[0])
                covered = torch.zeros(len(all_ids), dtype=torch.bool)
                for prefix, actual_inputs in prefix_inputs:
                    length = len(actual_inputs["positions"])
                    assert prefix + length <= len(all_ids)
                    selected = slice(prefix, prefix + length)
                    actual_prefix[selected] = actual_inputs["input_embeds"].to(
                        talker.device
                    )
                    covered[selected] = True
                if not bool(covered[: len(prompt_ids)][audio_mask.cpu()].all()):
                    assert (
                        arguments.cached_prefix_proof is not None
                    ), "cached audio requires its observed source proof"
                    owners = recover_audio_prefix(
                        arguments.thinker_directory,
                        arguments.cached_prefix_proof,
                        request_id,
                        audio_mask.cpu().nonzero(as_tuple=True)[0],
                        prompt["prompt_hidden"],
                        actual_prefix,
                        covered,
                        checkpoint_positions,
                    )
                    report["cached_prefix_sources"] = [
                        owner.model_dump() for owner in owners
                    ]
                else:
                    pass
                native_audio = actual_prefix[: len(prompt_ids)][audio_mask]
                assert torch.equal(
                    prompt["audio_feature_lengths"].to(talker.device),
                    inputs["feature_attention_mask"].sum(dim=1),
                )
                native_audio_hidden, native_audio_positions = prompt_hidden(
                    owner,
                    inputs["input_ids"],
                    native_audio,
                    inputs["feature_attention_mask"],
                )
                assert torch.equal(native_audio_positions, checkpoint_positions)
                expected_inputs = embeddings[0].clone()
                expected_inputs[audio_mask.nonzero(as_tuple=True)[0]] = native_audio
                complete_positions = owner.thinker.compute_3d_position_ids(
                    input_ids=all_ids[None],
                    image_grid_thw=None,
                    video_grid_thw=None,
                    inputs_embeds=expected_inputs[None],
                    attention_mask=torch.ones_like(all_ids[None]),
                    past_key_values=None,
                    audio_feature_lengths=inputs["feature_attention_mask"].sum(dim=1),
                )
                assert torch.equal(
                    complete_positions[:, :, : len(prompt_ids)], checkpoint_positions
                )
                complete_audio_mask = all_ids == config.thinker_config.audio_token_id
                prefix_proofs = []
                for prefix, actual_inputs in prefix_inputs:
                    length = len(actual_inputs["positions"])
                    selected = slice(prefix, prefix + length)
                    active_audio = complete_audio_mask[selected].cpu()
                    assert torch.equal(
                        actual_inputs["input_ids"][~active_audio],
                        all_ids[selected].cpu()[~active_audio],
                    )
                    delivered_exact = None
                    if bool(active_audio.any()):
                        actual_hidden = torch.load(
                            arguments.thinker_directory
                            / request_id
                            / f"thinker-hidden-{prefix}.pt",
                            weights_only=True,
                        )
                        delivered_exact = torch.equal(
                            actual_hidden[active_audio],
                            prompt["prompt_hidden"][selected][active_audio],
                        )
                        assert delivered_exact
                    else:
                        pass
                    prefix_proofs.append(
                        {
                            "prefix": prefix,
                            "rows": length,
                            "inputs": asdict(
                                compare_tensors(
                                    actual_inputs["input_embeds"].to(talker.device),
                                    expected_inputs[selected],
                                )
                            ),
                            "mrope_positions_exact": torch.equal(
                                actual_inputs["mrope_positions"],
                                complete_positions[:, 0, selected].cpu(),
                            ),
                            "hidden_delivery_exact": delivered_exact,
                        }
                    )
                assert prefix_proofs
                report["producer_delivery"] = prefix_proofs
                report["audio_encoder"] = asdict(
                    compare_tensors(native_audio, audio_embeddings)
                )
                report["hidden_with_identical_audio"] = asdict(
                    compare_tensors(
                        hidden[0, : len(prompt_ids)][audio_mask],
                        native_audio_hidden[0, audio_mask],
                    )
                )
                native_audio_prefill = checkpoint_audio_prefill.clone()
                native_audio_prefill[audio_rows] = talker.hidden_projection(
                    native_audio_hidden[0, audio_mask]
                )
                saved_tensors["native_audio_prefill"] = native_audio_prefill.cpu()
                saved_tensors["native_audio_reference_hidden"] = (
                    native_audio_hidden.cpu()
                )
            else:
                pass
            saved_tensors.update(
                {
                    "native_prefill": before["prefill_embeds"],
                    "checkpoint_audio_prefill": checkpoint_audio_prefill.cpu(),
                    "checkpoint_prompt_hidden": checkpoint_hidden.cpu(),
                    "checkpoint_positions": checkpoint_positions.cpu(),
                    "checkpoint_audio_embeddings": audio_embeddings.cpu(),
                }
            )
        else:
            pass
        torch.save(
            saved_tensors,
            arguments.output_directory / f"{request_id}.pt",
        )
        reports.append(report)
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(reports, indent=2)
        )
        print(json.dumps(report), flush=True)
    if arguments.audit_loaded_weights:
        audit_checkpoint_weights(
            owner if arguments.request_directory is not None else talker,
            checkpoint_directory,
            arguments.output_directory / "loaded-weight-audit.json",
        )
    else:
        pass


if __name__ == "__main__":
    main()
