"""Compare the checkpoint backbone along complete captured native histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights


class StepMetadata(BaseModel):
    step: int
    phase: str
    request_ids: list[str]
    frame_indices: list[int]
    projection_rows: dict[str, int]


class RequestTrace(BaseModel):
    steps: list[int]
    codes: list[list[int]]
    main_codes_match: bool


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
    ).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
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
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    steps = {
        metadata.step: metadata
        for line in (arguments.capture_directory / "steps.jsonl")
        .read_text()
        .splitlines()
        for metadata in [StepMetadata.model_validate_json(line)]
    }
    reports = []
    for request_id in arguments.request_id:
        trace = RequestTrace.model_validate_json(
            (
                arguments.capture_directory / "requests" / f"{request_id}.json"
            ).read_bytes()
        )
        assert trace.main_codes_match
        cache = None
        frames = []
        for frame_index, step_index in enumerate(trace.steps):
            metadata = steps[step_index]
            row = metadata.request_ids.index(request_id)
            assert metadata.frame_indices[row] == frame_index
            phase = "before_prefill" if frame_index == 0 else "before_decode"
            before = torch.load(
                arguments.capture_directory
                / "conditioning"
                / request_id
                / f"{frame_index:04d}-{phase}.pt",
                map_location="cpu",
                weights_only=True,
            )
            inputs = (
                before["prefill_embeds"]
                if frame_index == 0
                else (before["feedback_input"] + before["text_input"])[None]
            ).cuda()
            positions = before["positions"].cuda()
            assert inputs.shape[0] == positions.numel()
            if frame_index == 0:
                assert torch.equal(positions, torch.arange(len(inputs), device="cuda"))
            else:
                assert positions.tolist() == [cache.get_seq_length()]
            observed = torch.load(
                arguments.capture_directory / f"step-{step_index:06d}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            assert observed["codes"][row].tolist() == trace.codes[frame_index]
            assert torch.equal(
                observed["snapshots"]["backbone_input"][row], inputs[-1].cpu()
            )
            output = reference.model(
                inputs_embeds=inputs[None],
                position_ids=positions.view(1, 1, -1).expand(4, 1, -1),
                past_key_values=cache,
                use_cache=True,
            )
            cache = output.past_key_values
            hidden = output.last_hidden_state[:, -1]
            logits = reference.codec_head(hidden).float()
            native_hidden = observed["predictor_hidden"][row : row + 1].cuda()
            native_logits = observed["snapshots"]["codec.output"][row : row + 1].cuda()
            projected = reference.codec_head(native_hidden).float()
            assert hidden.isfinite().all() and logits.isfinite().all()
            selected_code = trace.codes[frame_index][0]
            codes = observed["codes"][row : row + 1].long().cuda()
            predictor_inputs = [
                native_hidden[:, None],
                reference.model.codec_embedding(codes[:, :1]),
            ]
            for group in range(config.num_code_groups - 2):
                predictor_inputs.append(
                    reference.code_predictor.model.codec_embedding[group](
                        codes[:, group + 1 : group + 2]
                    )
                )
            predictor_hidden = reference.code_predictor.model(
                inputs_embeds=torch.cat(predictor_inputs, dim=1), use_cache=False
            ).last_hidden_state
            predictor_logits = torch.stack(
                [
                    head(predictor_hidden[:, group + 1])
                    for group, head in enumerate(reference.code_predictor.lm_head)
                ],
                dim=1,
            ).float()
            assert predictor_logits.isfinite().all()
            chosen_logits = predictor_logits.gather(-1, codes[:, 1:, None])
            frames.append(
                {
                    "frame_index": frame_index,
                    "step": step_index,
                    "phase": metadata.phase,
                    "native_projection_rows": metadata.projection_rows["codec"],
                    "native_main_code": selected_code,
                    "is_codec_eos": selected_code == config.codec_eos_token_id,
                    "native_residual_codes": codes[0, 1:].tolist(),
                    "checkpoint_forced_prefix_argmax": predictor_logits.argmax(-1)[
                        0
                    ].tolist(),
                    "checkpoint_forced_prefix_native_code_ranks": (
                        (predictor_logits > chosen_logits).sum(-1)[0] + 1
                    ).tolist(),
                    "checkpoint_forced_prefix_native_code_logit_gaps": (
                        predictor_logits.max(-1).values - chosen_logits.squeeze(-1)
                    )[0].tolist(),
                    "head_reprojection_exact": torch.equal(
                        projected, native_logits.float()
                    ),
                    "head_reprojection_maximum_absolute": float(
                        (projected - native_logits.float()).abs().max()
                    ),
                    "reference_rank_of_native_main_code": int(
                        (logits[0] > logits[0, selected_code]).sum()
                    )
                    + 1,
                    "hidden_relative_l2": float(
                        (hidden.float() - native_hidden.float()).norm()
                        / hidden.float().norm()
                    ),
                    "maximum_absolute_logit_difference": float(
                        (logits - native_logits.float()).abs().max()
                    ),
                }
            )
        report = {"request_id": request_id, "frames": frames}
        reports.append(report)
        (arguments.output_directory / "results.json").write_text(
            json.dumps(
                {
                    "model_revision": arguments.model_revision,
                    "scope": "Every captured native input and frame, including its actual feedback and text handoff. The checkpoint backbone builds its own cache along that fixed input history. Separately the checkpoint predictor receives the native backbone query, main code and each native residual-code prefix, isolating predictor error from changed backbone inputs or earlier code choices. Reference geometry is one request, not the original padded native batch. No new sampling or waveform generation. Main-code ranks use raw logits before penalties and suppression. Residual codes on EOS frames are not consumed audio.",
                    "requests": reports,
                },
                indent=2,
            )
        )
        print(json.dumps({"request_id": request_id, "frames": len(frames)}), flush=True)
    checkpoint_directory = Path(
        snapshot_download(
            arguments.model_path,
            revision=arguments.model_revision,
            local_files_only=True,
        )
    )
    audit_checkpoint_weights(
        reference,
        checkpoint_directory,
        arguments.output_directory / "loaded-weight-audit.json",
    )


if __name__ == "__main__":
    main()
