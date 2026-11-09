"""Verify complete native histories and prepare identical checkpoint conditioning."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from pydantic import BaseModel
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

from sglang_omni.models.qwen3_omni.corpus_diagnostics import RequestTrace, StepMetadata


class SamplingFacts(BaseModel):
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float
    sampling_seed: int
    max_new_tokens: int
    stop_token_ids: list[int]
    suppress_tokens: list[int]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    steps = {
        metadata.step: metadata
        for line in (arguments.capture_directory / "steps.jsonl")
        .read_text()
        .splitlines()
        for metadata in [StepMetadata.model_validate_json(line)]
    }
    reports = []
    for request_id in arguments.request_id:
        source = arguments.capture_directory / "conditioning" / request_id
        destination = arguments.output_directory / request_id
        destination.mkdir()
        trace = RequestTrace.model_validate_json(
            (
                arguments.capture_directory / "requests" / f"{request_id}.json"
            ).read_bytes()
        )
        parameters = SamplingFacts.model_validate_json(
            (source / "sampling.json").read_bytes()
        )
        assert parameters.min_p == 0
        prefill = torch.load(source / "0000-before_prefill.pt", weights_only=True)
        finished = torch.load(source / "finished.pt", weights_only=True)
        assert bool(finished["thinker_done"])
        decode_paths = sorted(source.glob("*-before_decode.pt"))
        decode_inputs = [torch.load(path, weights_only=True) for path in decode_paths]
        queued_rows = [
            record["text_input"]
            for record in decode_inputs
            if bool(record["queued_text"])
        ]
        queued_rows.extend(finished["pending_text"])
        assert queued_rows
        text_rows = torch.stack(queued_rows)
        assert torch.equal(text_rows[-1], finished["tts_eos_embed"])
        prefill["trailing_text_embeds"] = text_rows
        torch.save(prefill, destination / "0000-before_prefill.pt")
        shutil.copyfile(source / "sampling.json", destination / "sampling.json")
        assert trace.main_codes_match
        assert len(trace.codes) == len(trace.steps) == len(decode_inputs) + 1
        history: list[int] = []
        for frame_index, step_index in enumerate(trace.steps):
            metadata = steps[step_index]
            row = metadata.request_ids.index(request_id)
            assert metadata.frame_indices[row] == frame_index
            observed = torch.load(
                arguments.capture_directory / f"step-{step_index:06d}.pt",
                weights_only=True,
                map_location="cpu",
            )
            logits = observed["snapshots"]["codec.output"][row : row + 1].float().cuda()
            if history:
                selected = torch.tensor(history, device="cuda", dtype=torch.long)
                repeated = logits[:, selected]
                logits[:, selected] = torch.where(
                    repeated > 0,
                    repeated / parameters.repetition_penalty,
                    repeated * parameters.repetition_penalty,
                )
            else:
                pass
            suppressed = [
                token
                for token in parameters.suppress_tokens
                if 0 <= token < logits.shape[-1]
            ]
            logits[:, suppressed] = -torch.inf
            probabilities = (logits / parameters.temperature).softmax(-1)
            position = observed["positions"][row : row + 1].cuda()
            sampled = top_k_top_p_min_p_sampling_from_probs_torch(
                probabilities,
                torch.tensor([parameters.top_k], device="cuda", dtype=torch.int32),
                torch.tensor([parameters.top_p], device="cuda", dtype=torch.float32),
                torch.zeros(1, device="cuda"),
                False,
                torch.tensor(
                    [parameters.sampling_seed], device="cuda", dtype=torch.int64
                ),
                position,
            )
            assert int(sampled[0]) == trace.codes[frame_index][0], (
                request_id,
                frame_index,
            )
            assert observed["codes"][row].tolist() == trace.codes[frame_index]
            if frame_index > 0:
                before = decode_inputs[frame_index - 1]
                assert before["sampling_seed"].tolist() == [parameters.sampling_seed]
                assert torch.equal(before["positions"], position.cpu())
                expected_text = (
                    text_rows[frame_index - 1]
                    if frame_index - 1 < len(text_rows)
                    else finished["tts_pad_embed"]
                )
                assert torch.equal(before["text_input"], expected_text)
                assert torch.equal(before["feedback_input"], previous_feedback)
                combined = before["feedback_input"] + before["text_input"]
                assert torch.equal(
                    combined, observed["snapshots"]["backbone_input"][row]
                )
                before["feedback_input"] = combined.unsqueeze(0)
                torch.save(before, destination / decode_paths[frame_index - 1].name)
                phase = "post_decode"
            else:
                phase = "post_prefill"
            previous_feedback = observed["feedback"][row]
            torch.save(
                {
                    "codec_ids": observed["codes"][row : row + 1],
                    "codec_feedback": previous_feedback.unsqueeze(0),
                    "logits": probabilities.cpu(),
                },
                destination / f"{frame_index:04d}-{phase}.pt",
            )
            history.append(trace.codes[frame_index][0])
        report = {
            "request_id": request_id,
            "frames_including_eos": len(history),
            "future_text_rows": len(text_rows),
            "all_draws_exact": True,
            "all_feedback_text_handoffs_exact": True,
            "sampling": parameters.model_dump(),
            "scope": "Complete consumed and pending text reconstructed from actual queue receipts. Legacy logits fields contain independently replayed sampling probabilities, not raw logits.",
        }
        reports.append(report)
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(reports, indent=2)
        )
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
