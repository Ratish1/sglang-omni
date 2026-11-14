"""Replay filtered stop-token choices on frozen diagnostic speech histories."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_probs_torch

from scripts.debug_qwen3_omni_sampling_alignment import SamplingParameters


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-pair", action="append", nargs=2, type=Path, required=True
    )
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    summaries = []
    for captured_root, reference_root in arguments.history_pair:
        for directory in sorted(captured_root.iterdir()):
            parameters = SamplingParameters.model_validate_json(
                (directory / "sampling.json").read_bytes()
            )
            initial = torch.load(
                directory / "0000-before_prefill.pt", weights_only=True
            )
            finished = torch.load(directory / "finished.pt", weights_only=True)
            files = sorted(directory.glob("*-output.pt"))
            assert len(files) == len(finished["output_ids"])
            assert [filename.name for filename in files] == [
                f"{frame:04d}-output.pt" for frame in range(len(files))
            ]
            top_k = torch.tensor([parameters.top_k], device=device)
            top_p = torch.tensor([parameters.top_p], device=device)
            minimum_probability = torch.tensor([parameters.min_p], device=device)
            seed = torch.tensor([parameters.sampling_seed], device=device)
            temperature = torch.tensor(
                [[parameters.temperature]], dtype=torch.float32, device=device
            )
            repetition_penalty = torch.tensor(
                [[parameters.repetition_penalty]], dtype=torch.float32, device=device
            )
            prefill_logits = torch.load(
                directory / "0000-prefill-logits.pt", weights_only=True
            )
            repeated = torch.zeros_like(prefill_logits, dtype=torch.bool, device=device)
            rows = []
            for frame, filename in enumerate(files):
                captured = torch.load(filename, weights_only=True)
                expected = torch.load(
                    reference_root / directory.name / "checkpoint" / filename.name,
                    weights_only=True,
                )
                assert torch.equal(
                    expected["hidden_states"][0],
                    (
                        initial["prefill_embeds"].unsqueeze(0)
                        if frame == 0
                        else finished["decode_input_history"][frame - 1].view(1, 1, -1)
                    ),
                )
                raw_logits = (
                    prefill_logits if frame == 0 else captured["logits"]
                ).unsqueeze(0)
                position = len(initial["prefill_embeds"]) + frame - 1
                assert int(captured["sequence_length"]) == position + 1
                choices = {}
                row = {"frame": frame, "sampling_position": position}
                for owner, logits in (
                    ("native", raw_logits),
                    ("checkpoint", expected["logits"]),
                ):
                    logits = logits.to(device)
                    penalized = torch.where(
                        logits > 0,
                        logits / repetition_penalty,
                        logits * repetition_penalty,
                    )
                    processed = torch.where(repeated, penalized, logits)
                    processed[:, parameters.suppress_tokens] = float("-inf")
                    probabilities = (processed / temperature).softmax(-1)
                    chosen, weights, permutation, _ = (
                        top_k_top_p_min_p_sampling_from_probs_torch(
                            probabilities,
                            top_k,
                            top_p,
                            minimum_probability,
                            False,
                            seed,
                            torch.tensor([position], device=device),
                            return_filtered_probs=True,
                        )
                    )
                    by_token = torch.zeros_like(weights).scatter_(
                        1, permutation.long(), weights
                    )
                    assert by_token.isfinite().all() and by_token.sum() > 0
                    normalized = by_token / by_token.sum(-1, keepdim=True)
                    choices[owner] = int(chosen[0])
                    row[owner] = {
                        "chosen_code": choices[owner],
                        "eos_filtered_probability": float(
                            normalized[0, arguments.eos_token_id]
                        ),
                        "actual_code_filtered_probability": float(
                            normalized[0, int(captured["codec_ids"][0])]
                        ),
                    }
                actual_code = int(finished["output_ids"][frame])
                assert choices["native"] == int(captured["codec_ids"][0]) == actual_code
                assert (actual_code == arguments.eos_token_id) == (
                    frame == len(files) - 1
                )
                repeated[actual_code] = True
                rows.append(row)
            destination = (
                arguments.output_directory / captured_root.name / directory.name
            )
            destination.mkdir(parents=True)
            (destination / "frames.json").write_text(json.dumps(rows, indent=2))
            summary = {
                "history": str(directory),
                "request_id": directory.name,
                "all_actual_draws_exact": True,
                "frames": len(rows),
                "checkpoint_earlier_eos_frames": [
                    row["frame"]
                    for row in rows[:-1]
                    if row["checkpoint"]["chosen_code"] == arguments.eos_token_id
                ],
                "native_prefinal_eos_max_filtered_probability": max(
                    row["native"]["eos_filtered_probability"] for row in rows[:-1]
                ),
                "checkpoint_prefinal_eos_max_filtered_probability": max(
                    row["checkpoint"]["eos_filtered_probability"] for row in rows[:-1]
                ),
                "actual_code_outside_checkpoint_support_frames": [
                    row["frame"]
                    for row in rows
                    if row["checkpoint"]["actual_code_filtered_probability"] == 0
                ],
            }
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    sampler_source = Path(
        inspect.getfile(inspect.unwrap(top_k_top_p_min_p_sampling_from_probs_torch))
    )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(
            {
                "requests": summaries,
                "sampler_source_sha256": hashlib.sha256(
                    sampler_source.read_bytes()
                ).hexdigest(),
                "scope": "Frozen reconstructed-native diagnostic trajectories, not new endpoint captures. Raw logits replay the original sampler, full repetition mask, seed and position. Checkpoint-owned cache is conditioned on actual native inputs. Counterfactual draws are not free-running recovery or native-KV integrity proof.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
