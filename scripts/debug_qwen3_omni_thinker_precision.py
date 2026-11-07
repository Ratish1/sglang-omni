# SPDX-License-Identifier: Apache-2.0
"""Compare actual multimodal conditioning with a full FP32 Thinker control."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors


@dataclass(kw_only=True)
class ConditioningCase:
    directory: Path
    prompt_ids: torch.Tensor
    embeddings: torch.Tensor
    positions: torch.Tensor
    native_hidden: torch.Tensor
    reference_hidden: torch.Tensor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--case", nargs=2, action="append", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda", arguments.gpu)
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
    model.talker.cpu()
    model.code2wav.cpu()
    model.thinker.audio_tower.cpu()
    model.thinker.visual.cpu()
    model.thinker.lm_head.cpu()
    torch.cuda.empty_cache()
    cases: list[ConditioningCase] = []
    with torch.inference_mode():
        for directory, reference_file in arguments.case:
            native = torch.load(directory / "conditioning.pt", weights_only=True)
            audio = torch.load(directory / "audio-conditioning.pt", weights_only=True)
            reference = torch.load(reference_file, weights_only=True)
            prompt_ids = native["prompt_input_ids"][None].to(device)
            assert torch.equal(reference["prompt_input_ids"], prompt_ids.cpu())
            embeddings = model.thinker.model.embed_tokens(prompt_ids)
            audio_mask = prompt_ids == model.config.thinker_config.audio_token_id
            embeddings[audio_mask] = audio["audio_embeds"].to(embeddings)
            positions = model.thinker.compute_3d_position_ids(
                input_ids=prompt_ids,
                image_grid_thw=None,
                video_grid_thw=None,
                inputs_embeds=embeddings,
                attention_mask=torch.ones_like(prompt_ids),
                past_key_values=None,
                audio_feature_lengths=audio["feature_attention_mask"]
                .sum(dim=1)
                .to(device),
            )
            cases.append(
                ConditioningCase(
                    directory=directory,
                    prompt_ids=prompt_ids,
                    embeddings=embeddings,
                    positions=positions,
                    native_hidden=native["prompt_hidden"].to(device),
                    reference_hidden=reference["native_audio_reference_hidden"].to(
                        device
                    ),
                )
            )
        for case in cases:
            output = model.thinker.model(
                inputs_embeds=case.embeddings,
                attention_mask=torch.ones_like(case.prompt_ids),
                position_ids=case.positions,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = output.hidden_states[
                model.config.talker_config.accept_hidden_layer
            ]
            assert torch.equal(hidden, case.reference_hidden)
            print(
                f"Checkpoint BF16 conditioning reproduced exactly: {case.directory.name}",
                flush=True,
            )
        model.thinker.model.float()
        torch.cuda.empty_cache()
        reports = []
        for case in cases:
            output = model.thinker.model(
                inputs_embeds=case.embeddings.float(),
                attention_mask=torch.ones_like(case.prompt_ids),
                position_ids=case.positions,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = output.hidden_states[
                model.config.talker_config.accept_hidden_layer
            ]
            audio_mask = (
                case.prompt_ids[0] == model.config.thinker_config.audio_token_id
            )
            report = {
                "request_id": case.directory.name,
                "checkpoint_bfloat16_reproduced": True,
                "native_to_float32": asdict(
                    compare_tensors(
                        case.native_hidden[audio_mask], hidden[0, audio_mask]
                    )
                ),
                "checkpoint_bfloat16_to_float32": asdict(
                    compare_tensors(
                        case.reference_hidden[0, audio_mask], hidden[0, audio_mask]
                    )
                ),
            }
            torch.save(
                hidden.cpu(), arguments.output_directory / f"{case.directory.name}.pt"
            )
            reports.append(report)
            (arguments.output_directory / "results.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
