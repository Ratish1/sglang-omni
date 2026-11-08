"""Compare output-head rounding on the actual frozen hidden states and weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig
from transformers.utils.hub import cached_file

from scripts.debug_qwen3_omni_sampling_coupling import sampling_draw


def project_head(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    recorded_logits: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, bool | int | float]]:
    hidden_states = hidden_states.reshape(-1, weight.shape[1])
    recorded_logits = recorded_logits.reshape(-1, weight.shape[0]).float()
    assert hidden_states.shape[0] == 1
    ordinary_logits = torch.mm(hidden_states, weight.T).float()
    assert torch.equal(ordinary_logits, recorded_logits)
    projected_logits = torch.mm(hidden_states, weight.T, out_dtype=torch.float32)
    reference_logits = torch.mm(hidden_states.float(), weight.float().T)
    report = {
        "recorded_logits_exact": True,
        "recorded_argmax": int(recorded_logits.argmax()),
        "float32_output_argmax": int(projected_logits.argmax()),
        "recorded_tied_maxima": int((recorded_logits == recorded_logits.max()).sum()),
        "float32_output_top_two_gap": float(
            projected_logits.topk(2).values.diff().neg()[0, 0]
        ),
        "float32_output_rounds_to_recorded": bool(
            torch.equal(projected_logits.to(weight.dtype).float(), recorded_logits)
        ),
        "ordinary_float32_relative_l2": float(
            (ordinary_logits - reference_logits).norm() / reference_logits.norm()
        ),
        "output_float32_relative_l2": float(
            (projected_logits - reference_logits).norm() / reference_logits.norm()
        ),
    }
    return projected_logits, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--predictor-operands-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    assert not arguments.output_file.exists()
    torch.set_float32_matmul_precision("highest")
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    checkpoint_index = Path(
        cached_file(arguments.model_path, "model.safetensors.index.json")
    )
    weight_name = "talker.codec_head.weight"
    shard = json.loads(checkpoint_index.read_text())["weight_map"][weight_name]
    with safe_open(
        cached_file(arguments.model_path, shard), framework="pt", device="cuda:0"
    ) as checkpoint:
        weight = checkpoint.get_tensor(weight_name)
    reports = []
    summary = json.loads((arguments.reference_directory / "summary.json").read_text())
    with torch.inference_mode():
        for observation in summary:
            if observation["history"] != "native_past":
                continue
            else:
                pass
            filename = (
                arguments.reference_directory
                / f"{observation['request_id']}.native_past.{observation['frame']}.pt"
            )
            capture = torch.load(filename, map_location="cuda", weights_only=True)
            projected_logits, report = project_head(
                capture["native_hidden_states"], weight, capture["native_logits"]
            )
            draws = [
                sampling_draw(
                    logits,
                    capture["repetition_mask"],
                    capture["seed"],
                    capture["positions"],
                    capture["temperature"],
                    capture["repetition_penalty"],
                    capture["top_k"],
                    capture["top_p"],
                    config.code_predictor_config.vocab_size,
                    config.codec_eos_token_id,
                )
                for logits in (capture["native_logits"], projected_logits)
            ]
            ordinary, projected = draws
            assert ordinary.sampled_token_id == observation["native_token_id"]
            report.update(
                native_sample=ordinary.sampled_token_id,
                projected_sample=projected.sampled_token_id,
                projected_probabilities_original_noise=int(
                    (projected.token_log_weights + ordinary.token_noise).argmax()
                ),
                original_probabilities_projected_noise=int(
                    (ordinary.token_log_weights + projected.token_noise).argmax()
                ),
                supported_tokens_with_changed_noise=int(
                    (
                        (ordinary.token_noise != projected.token_noise)
                        & (
                            torch.isfinite(ordinary.token_log_weights)
                            | torch.isfinite(projected.token_log_weights)
                        )
                    ).sum()
                ),
            )
            reports.append(
                {
                    "request_id": observation["request_id"],
                    "frame": observation["frame"],
                    **report,
                }
            )
        operands = torch.load(
            arguments.predictor_operands_file, map_location="cuda", weights_only=True
        )
        _, predictor_report = project_head(
            operands["hidden_states"], operands["weight"], operands["native_logits"]
        )
    result = {"main_head": reports, "predictor_head": predictor_report}
    arguments.output_file.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
