"""Compare every saved native and reference Talker boundary through EOS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def compare_tensors(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, bool | int | float]:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    return {
        "exact": torch.equal(actual, expected),
        "bytes_exact": torch.equal(
            actual.contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        ),
        "changed_values": int(torch.count_nonzero(actual != expected)),
        "maximum_absolute_error": float(
            (actual.float() - expected.float()).abs().max()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-directory", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    finished = torch.load(arguments.native_directory / "finished.pt", weights_only=True)
    prefill = torch.load(
        arguments.native_directory / "0000-before_prefill.pt", weights_only=True
    )
    assert int(prefill["extend_prefix_length"]) == 0
    expected_frames = len(finished["output_ids"])
    eos_token_id = finished["finished_reason"]["matched"]
    assert isinstance(eos_token_id, int)
    reference_files = sorted(arguments.reference_directory.glob("*-post_*.pt"))
    assert len(reference_files) == expected_frames
    exact_counts: dict[str, int] = {}
    byte_exact_counts: dict[str, int] = {}
    comparisons: dict[str, int] = {}
    with (arguments.output_directory / "frames.jsonl").open("x") as destination:
        for frame, reference_file in enumerate(reference_files):
            observed = torch.load(
                arguments.native_directory / f"{frame:04d}-output.pt", weights_only=True
            )
            operators = torch.load(
                arguments.native_directory / f"{frame:04d}-operators.pt",
                weights_only=True,
            )
            reference = torch.load(reference_file, weights_only=True)
            layers = torch.load(
                arguments.reference_directory / f"{frame:04d}-layers.pt",
                weights_only=True,
            )
            assert bool(observed["past_exact"])
            assert len(operators) == len(layers["keys"]) == len(layers["values"])
            assert len(layers["hidden_states"]) == len(operators) + 1
            pairs = {
                "main_code": (
                    observed["codec_ids"].reshape(-1)[:1],
                    reference["codec_ids"].reshape(-1)[:1],
                ),
                "hidden": (
                    observed["hidden_states"].reshape(-1),
                    reference["hidden_states"].reshape(-1),
                ),
                "inputs": (
                    (
                        prefill["prefill_embeds"]
                        if frame == 0
                        else finished["decode_input_history"][frame - 1].reshape(1, -1)
                    ),
                    layers["input_embeddings"].reshape(
                        -1, layers["input_embeddings"].shape[-1]
                    ),
                ),
            }
            if frame == expected_frames - 1:
                assert int(observed["codec_ids"][0]) == eos_token_id
                assert int(reference["codec_ids"][0, 0]) == eos_token_id
            else:
                assert int(observed["codec_ids"][0]) != eos_token_id
                pairs["audio_codes"] = (
                    observed["codec_ids"].reshape(-1),
                    reference["codec_ids"].reshape(-1),
                )
            frame_report = {
                name: compare_tensors(*tensors) for name, tensors in pairs.items()
            }
            layer_reports = []
            for layer, snapshot in enumerate(operators):
                pairs = {
                    "residual": (
                        (
                            snapshot["prefix_attention_residual"]
                            if frame == 0
                            else snapshot["attention_residual"].reshape(1, -1)
                        ),
                        layers["hidden_states"][layer].reshape(
                            -1, snapshot["attention_residual"].shape[-1]
                        ),
                    ),
                    "keys": (
                        observed["keys"][layer],
                        layers["keys"][layer].squeeze(0).transpose(0, 1),
                    ),
                    "values": (
                        observed["values"][layer],
                        layers["values"][layer].squeeze(0).transpose(0, 1),
                    ),
                }
                layer_reports.append(
                    {name: compare_tensors(*tensors) for name, tensors in pairs.items()}
                )
            for report in (frame_report, *layer_reports):
                for name, comparison in report.items():
                    comparisons[name] = comparisons.get(name, 0) + 1
                    exact_counts[name] = exact_counts.get(name, 0) + int(
                        comparison["exact"]
                    )
                    byte_exact_counts[name] = byte_exact_counts.get(name, 0) + int(
                        comparison["bytes_exact"]
                    )
            destination.write(
                json.dumps(
                    {
                        "frame": frame,
                        "boundaries": frame_report,
                        "layers": layer_reports,
                    }
                )
                + "\n"
            )
    report = {
        "request_id": arguments.native_directory.name,
        "frames_including_eos": expected_frames,
        "comparisons": comparisons,
        "exact_counts": exact_counts,
        "byte_exact_counts": byte_exact_counts,
        "all_exact": comparisons == exact_counts,
        "all_bytes_exact": comparisons == byte_exact_counts,
        "scope": "Every consumed input, composed layer residual, new KV row (complete prefill), final hidden, audio-bearing full code frame and EOS main code. Unconsumed EOS residual codes are excluded. Does not substitute captured outputs into generation.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
