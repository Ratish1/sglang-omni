"""Bind identity and checkpoint-producer controls to one reproduced native request."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-directory", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--hold-native-prefix-cache", action="store_true")
    parser.add_argument(
        "--producer-precision",
        choices=("chained", "checkpoint", "router", "experts", "router-and-experts"),
        default="chained",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    summary = json.loads((arguments.reference_directory / "summary.json").read_text())
    assert summary["request_id"] == arguments.request_id
    assert summary["model_revision"] == arguments.model_revision
    weights = json.loads(
        (arguments.reference_directory / "weight-audit.json").read_text()
    )
    assert not weights["mismatches"] and not weights["unvisited_tensors"]
    if arguments.hold_native_prefix_cache:
        assert summary["native_prefix_cache_held_for_local_controls"]
        if arguments.producer_precision != "chained":
            controls = json.loads(
                (arguments.reference_directory / "precision-controls.json").read_text()
            )
            assert arguments.producer_precision in controls["variants"]
        else:
            pass
        reference_file = (
            arguments.reference_directory
            / f"native-prefix-{arguments.producer_precision}-operands.pt"
        )
    else:
        assert arguments.producer_precision == "chained"
        reference_file = arguments.reference_directory / "producer-operands.pt"
    reference = torch.load(reference_file, map_location="cpu", weights_only=True)
    input_files = sorted(
        (arguments.delivery_directory / "inputs").glob(f"{arguments.request_id}-*.pt")
    )
    assert len(input_files) == 1
    inputs = torch.load(input_files[0], map_location="cpu", weights_only=True)
    assert inputs["request_id"] == arguments.request_id
    prefix_token_count = inputs["prefix_length"]
    assert torch.equal(
        reference["embedded_input"][prefix_token_count:], inputs["embedded_input"]
    )
    assert torch.equal(
        reference["mrope_positions"][:, prefix_token_count:], inputs["mrope_positions"]
    )
    handoff_file = arguments.delivery_directory / arguments.request_id / "handoff.pt"
    handoff = torch.load(handoff_file, map_location="cpu", weights_only=True)
    assert torch.equal(handoff["causal_token_ids"], inputs["causal_token_ids"])
    positions = handoff["prompt_positions"]
    assert bool((positions >= prefix_token_count).all())
    accepted = reference[f"checkpoint_hidden_{summary['accepted_layer']}"]
    if arguments.hold_native_prefix_cache:
        assert len(accepted) == len(handoff["causal_token_ids"]) - prefix_token_count
        checkpoint_hidden = accepted[positions - prefix_token_count]
    else:
        assert len(accepted) == len(handoff["causal_token_ids"])
        checkpoint_hidden = accepted[positions]
    native_hidden = handoff["outgoing_hidden"]
    assert checkpoint_hidden.shape == native_hidden.shape
    assert checkpoint_hidden.dtype == native_hidden.dtype == torch.bfloat16
    assert bool(torch.isfinite(checkpoint_hidden).all())
    assert not torch.equal(checkpoint_hidden, native_hidden)
    selected_writes = []
    for path in sorted((arguments.delivery_directory / "writes").glob("*.pt")):
        written = torch.load(path, map_location="cpu", weights_only=True)
        if written["request_id"] == arguments.request_id:
            selected_writes.append(written)
        else:
            pass
    assert len(selected_writes) == 1
    written = selected_writes[0]
    assert torch.equal(written["prompt_positions"], positions)
    assert torch.equal(written["producer_hidden"], native_hidden)
    assert torch.equal(
        written["accepted_full_hidden"], reference["native_accepted_hidden"]
    )
    provenance = {
        "model_revision": arguments.model_revision,
        "accepted_layer": summary["accepted_layer"],
        "reference_prefix_recomputed": not arguments.hold_native_prefix_cache,
        "producer_precision": arguments.producer_precision,
        "native_input_sha256": hashlib.sha256(input_files[0].read_bytes()).hexdigest(),
        "reference_operands_sha256": hashlib.sha256(
            reference_file.read_bytes()
        ).hexdigest(),
        "native_handoff_sha256": hashlib.sha256(handoff_file.read_bytes()).hexdigest(),
    }
    for name, replacement in (
        ("identity", native_hidden),
        ("checkpoint", checkpoint_hidden),
    ):
        torch.save(
            {
                "request_id": arguments.request_id,
                "prompt_positions": positions,
                "causal_token_ids": handoff["causal_token_ids"],
                "native_hidden": native_hidden,
                "replacement_hidden": replacement,
                "mode": name,
                **provenance,
            },
            arguments.output_directory / f"{name}.pt",
        )
    report = {
        "request_id": arguments.request_id,
        "native_hidden_shape": list(native_hidden.shape),
        "reference_loaded_weights_checked": weights["checked_checkpoint_tensors"],
        "scope": "Only the selected multimodal producer rows change. Native cached rows, returned text, sampling settings and all Talker operators stay unchanged. The identity control must reproduce the original complete trajectory before interpreting the checkpoint control.",
        **provenance,
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
