"""Check actual conditioning, sampling history and cache continuity through EOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch
from transformers import Qwen3OmniMoeConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--reference-logits-directory", type=Path)
    arguments = parser.parse_args()
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision, local_files_only=True
    )
    reports = []
    for directory in sorted(
        path for path in arguments.capture_directory.iterdir() if path.is_dir()
    ):
        prompt = torch.load(directory / "prompt.pt", weights_only=True)
        finished = torch.load(directory / "finished.pt", weights_only=True)
        parameters = json.loads((directory / "sampling.json").read_text())
        initial = torch.load(directory / "0000-before_prefill.pt", weights_only=True)
        prefix = initial["prefill_embeds"]
        history = finished["decode_input_history"]
        enqueued = finished["enqueued_text_rows"]
        pad = finished["tts_pad_embed"]
        eos = finished["tts_eos_embed"]
        output_files = sorted(directory.glob("*-output.pt"))
        outputs = [torch.load(filename, weights_only=True) for filename in output_files]
        assert output_files
        assert [filename.name for filename in output_files] == [
            f"{frame:04d}-output.pt" for frame in range(len(outputs))
        ]
        assert len(outputs) == len(finished["output_ids"])
        assert len(history) == len(outputs) - 1
        assert torch.equal(
            torch.stack([output["codec_ids"][0] for output in outputs]),
            finished["output_ids"],
        )
        assert finished["finished_reason"]["type"] == "stop"
        assert finished["finished_reason"]["matched"] in parameters["stop_token_ids"]
        assert bool(finished["thinker_done"]) and not len(finished["pending_text"])
        spoken_tokens = finished["assistant_token_ids"]
        if int(spoken_tokens[-1]) == configuration.im_end_token_id:
            spoken_tokens = spoken_tokens[:-1]
        else:
            pass
        assert len(enqueued) == len(spoken_tokens)
        assert int((enqueued == eos).all(dim=-1).sum()) == 1
        frame_reports = []
        top_k = torch.tensor([parameters["top_k"]], device=device)
        top_p = torch.tensor([parameters["top_p"]], device=device)
        minimum_probability = torch.tensor([parameters["min_p"]], device=device)
        sampling_seed = torch.tensor([parameters["sampling_seed"]], device=device)
        repetition_penalty = torch.tensor(
            [[parameters["repetition_penalty"]]], dtype=torch.float32, device=device
        )
        temperature = torch.tensor(
            [[parameters["temperature"]]], dtype=torch.float32, device=device
        )
        logit_sources = (
            ("observed", "checkpoint", "native")
            if arguments.reference_logits_directory is not None
            else ("observed",)
        )
        for frame, output in enumerate(outputs):
            phase = "before_prefill" if bool(output["is_extend"]) else "before_decode"
            before = torch.load(
                directory / f"{frame:04d}-{phase}.pt", weights_only=True
            )
            assert torch.equal(before["output_ids"], finished["output_ids"][:frame])
            expected_position = len(prefix) + frame - 1
            position_exact = int(before["positions"][-1]) == expected_position
            sequence_length_exact = (
                int(output["sequence_length"]) == expected_position + 1
            )
            if phase == "before_prefill":
                start = int(before["extend_prefix_length"])
                count = len(before["prefill_embeds"])
                expected_inputs = torch.cat((prefix, history))
                input_exact = torch.equal(
                    before["prefill_embeds"], expected_inputs[start : start + count]
                )
                position_exact = position_exact and torch.equal(
                    before["positions"], torch.arange(start, start + count)
                )
                repetition_mask_exact = None
                seed_exact = None
            else:
                expected_text = (
                    enqueued[frame - 1] if frame - 1 < len(enqueued) else pad
                )
                input_exact = torch.equal(
                    before["feedback_input"], outputs[frame - 1]["codec_feedback"]
                )
                input_exact = input_exact and torch.equal(
                    before["text_input"], expected_text
                )
                input_exact = input_exact and bool(before["queued_text"]) == (
                    frame - 1 < len(enqueued)
                )
                expected_mask = torch.zeros_like(before["repetition_mask"])
                expected_mask[finished["output_ids"][:frame]] = True
                repetition_mask_exact = torch.equal(
                    before["repetition_mask"], expected_mask
                )
                seed_exact = (
                    int(before["sampling_seed"][0]) == parameters["sampling_seed"]
                )
            if frame < len(history):
                expected_text = enqueued[frame] if frame < len(enqueued) else pad
                combined_exact = torch.equal(
                    history[frame], output["codec_feedback"] + expected_text
                )
            else:
                combined_exact = None
            main_code_choices: dict[str, int] = {}
            for source in logit_sources:
                if source == "observed" and frame == 0:
                    probabilities = output["logits"].unsqueeze(0).to(device)
                else:
                    if source == "observed":
                        logits = output["logits"].unsqueeze(0).to(device)
                    else:
                        reference = torch.load(
                            arguments.reference_logits_directory
                            / directory.name
                            / source
                            / output_files[frame].name,
                            weights_only=True,
                        )
                        reference_inputs = (
                            prefix.unsqueeze(0)
                            if frame == 0
                            else history[frame - 1].view(1, 1, -1)
                        )
                        assert torch.equal(
                            reference["hidden_states"][0], reference_inputs
                        )
                        logits = reference["logits"].to(device)
                    repeated = torch.zeros_like(logits, dtype=torch.bool)
                    repeated[:, finished["output_ids"][:frame].to(device)] = True
                    penalized = torch.where(
                        logits > 0,
                        logits / repetition_penalty,
                        logits * repetition_penalty,
                    )
                    logits = torch.where(repeated, penalized, logits)
                    logits[:, parameters["suppress_tokens"]] = float("-inf")
                    probabilities = torch.softmax(logits / temperature, dim=-1)
                sampled = top_k_top_p_min_p_sampling_from_probs_torch(
                    probabilities,
                    top_k,
                    top_p,
                    minimum_probability,
                    False,
                    sampling_seed,
                    torch.tensor([expected_position], device=device),
                )
                main_code_choices[source] = int(sampled[0])
            frame_reports.append(
                {
                    "frame": frame,
                    "phase": phase,
                    "position_exact": position_exact,
                    "sequence_length_exact": sequence_length_exact,
                    "input_exact": input_exact,
                    "combined_feedback_text_exact": combined_exact,
                    "repetition_mask_exact": repetition_mask_exact,
                    "sampling_seed_exact": seed_exact,
                    "previous_cache_exact": bool(output["past_exact"]),
                    "sampled_code_exact": main_code_choices["observed"]
                    == int(output["codec_ids"][0]),
                    "main_code_choices": main_code_choices,
                }
            )
        report = {
            "request_id": directory.name,
            "has_prompt_hidden": bool(prompt["has_prompt_hidden"]),
            "prompt_rows": len(prefix),
            "frames_including_eos": len(outputs),
            "initial_assistant_tokens": prompt["initial_assistant_token_ids"].tolist(),
            "all_assistant_tokens": finished["assistant_token_ids"].tolist(),
            "enqueued_rows": len(enqueued),
            "last_enqueued_row_is_eos": torch.equal(enqueued[-1], eos),
            "decode_history_rows": len(history),
            "mixed_or_recomputed_prefills": sum(
                bool(output["is_extend"]) for output in outputs[1:]
            ),
            "reference_main_code_matches": {
                source: sum(
                    row["main_code_choices"][source]
                    == int(finished["output_ids"][row["frame"]])
                    for row in frame_reports
                )
                for source in logit_sources
                if source != "observed"
            },
            "failures": [
                row
                for row in frame_reports
                if any(value is False for value in row.values())
            ],
            "frames": frame_reports,
        }
        reports.append(report)
        print(
            json.dumps(
                {name: value for name, value in report.items() if name != "frames"}
            ),
            flush=True,
        )
    assert reports
    with arguments.output.open("x") as destination:
        json.dump(reports, destination, indent=2)
    assert all(
        not report["failures"] and report["last_enqueued_row_is_eos"]
        for report in reports
    )


if __name__ == "__main__":
    main()
else:
    pass
