"""Check complete captured speech conditioning against checkpoint construction."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from sglang_omni.models.qwen3_omni.components.talker_prefill import (
    load_thinker_embedding_rows,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
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
        torch.save(
            {
                "checkpoint_prefix": expected_prefix.cpu(),
                "checkpoint_future_text": future_text[0].cpu(),
                "native_prefix": before["prefill_embeds"],
                "native_future_text": native_future.cpu(),
            },
            arguments.output_directory / f"{request_id}.pt",
        )
        reports.append(report)
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(reports, indent=2)
        )
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
