"""Check every captured native text decision against checkpoint head arithmetic."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from pydantic import BaseModel
from safetensors import safe_open
from transformers import AutoTokenizer


class CheckpointIndex(BaseModel):
    weight_map: dict[str, str]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--response-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    index = CheckpointIndex.model_validate_json(
        (arguments.checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    parameter_name = "thinker.lm_head.weight"
    with safe_open(
        arguments.checkpoint_directory / index.weight_map[parameter_name],
        framework="pt",
        device="cpu",
    ) as checkpoint:
        head_weight = checkpoint.get_tensor(parameter_name).to("cuda", torch.float64)
    counts: Counter[str] = Counter()
    maximum_logit_error = 0.0
    seen: set[tuple[str, tuple[int, ...]]] = set()
    output_tokens: dict[str, list[int]] = {}
    with (
        (arguments.capture_directory / "steps.jsonl").open() as steps,
        (arguments.output_directory / "decisions.jsonl").open("w") as output,
        torch.inference_mode(),
    ):
        for line in steps:
            metadata = json.loads(line)
            tensors = torch.load(
                arguments.capture_directory / metadata["tensor_file"],
                map_location="cpu",
                weights_only=True,
            )
            assert torch.equal(tensors["token_ids"], tensors["greedy_ids"])
            assert bool(tensors["raw_logits_finite"].all())
            head_inputs = tensors["head_inputs"].to("cuda", torch.float64)
            expected_logits = head_inputs @ head_weight.T
            assert bool(expected_logits.isfinite().all())
            token_ids = tensors["token_ids"].to("cuda")
            selected_logits = expected_logits.gather(1, token_ids[:, None])
            selected_ranks = (expected_logits > selected_logits).sum(-1).cpu()
            expected_ids = expected_logits.argmax(-1).cpu()
            expected_top_logits = expected_logits.gather(
                1, tensors["top_ids"].to("cuda")
            ).cpu()
            rounded_top_logits = expected_top_logits.to(torch.bfloat16).double()
            observed_top_logits = tensors["top_logits"].double()
            for row, request_id in enumerate(metadata["request_ids"]):
                if metadata["remaining_prefill_chunks"][row] > 0:
                    counts["nonfinal_prefill_rows"] += 1
                    continue
                else:
                    pass
                prefix = metadata["output_prefixes"][row]
                assert prefix == output_tokens.get(request_id, []), request_id
                output_tokens[request_id] = prefix + [int(tensors["token_ids"][row])]
                identity = (request_id, tuple(prefix))
                assert identity not in seen, identity
                seen.add(identity)
                changed = int(expected_ids[row]) != int(tensors["token_ids"][row])
                tied = bool(observed_top_logits[row, 0] == observed_top_logits[row, 1])
                rounded_equal = bool(
                    torch.equal(rounded_top_logits[row], observed_top_logits[row])
                )
                error = float(
                    (expected_top_logits[row] - observed_top_logits[row]).abs().max()
                )
                maximum_logit_error = max(maximum_logit_error, error)
                counts["decisions"] += 1
                counts["requests_first_decision"] += not prefix
                counts[f"phase_{metadata['phase']}"] += 1
                counts["native_top_ties"] += tied
                counts["fp64_changes_argmax"] += changed
                counts["changes_on_native_ties"] += changed and tied
                counts["changes_without_native_ties"] += changed and not tied
                counts["top_eight_equal_rounded_fp64"] += rounded_equal
                output.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "output_position": len(prefix),
                            "phase": metadata["phase"],
                            "native_token_id": int(tensors["token_ids"][row]),
                            "fp64_token_id": int(expected_ids[row]),
                            "fp64_selected_rank": int(selected_ranks[row]),
                            "native_top_tie": tied,
                            "top_eight_equal_rounded_fp64": rounded_equal,
                            "maximum_top_logit_error": error,
                            "fp64_selected_margin": float(
                                selected_logits[row, 0] - expected_logits[row].max()
                            ),
                        }
                    )
                    + "\n"
                )
    expected_requests = json.loads(
        (arguments.capture_directory / "request_ids.json").read_text()
    )
    assert set(output_tokens) == set(expected_requests)
    tokenizer = AutoTokenizer.from_pretrained(arguments.checkpoint_directory)
    for request_id, token_ids in output_tokens.items():
        response = json.loads(
            (arguments.response_directory / f"{request_id}.speech.json").read_text()
        )
        text = tokenizer.decode(
            [token_id for token_id in token_ids if token_id != tokenizer.eos_token_id],
            skip_special_tokens=True,
        )
        assert text == response["text"], (request_id, text, response["text"])
    report = {
        "counts": dict(counts),
        "verified_response_texts": len(output_tokens),
        "maximum_top_logit_error": maximum_logit_error,
        "interpretation": "FP64 differences measure rounding sensitivity, not a serving failure.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
