# SPDX-License-Identifier: Apache-2.0
"""Replay native decode draws and locate duration changes in the codec witnesses."""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import torch

from scripts.debug_qwen3_omni_norm_sampler import readout
from scripts.debug_qwen3_omni_rounding_witnesses import WitnessResult, compare_witnesses


class DecodeDecision(TypedDict):
    frame_index: int
    position: int
    code: int
    eos_rank: int
    eos_probability: float
    uses_pad: bool
    batch_rows: int


def replay_decode(
    trace_directory: Path,
    codec_frames: list[list[int]],
    eos_token_id: int,
    codebook_count: int,
    sampling_seed: int,
    device: torch.device,
) -> list[DecodeDecision]:
    prefill = torch.load(trace_directory / "backbone-prefill-000.pt", weights_only=True)
    assert 1 < len(codec_frames) < prefill["max_new_tokens"], trace_directory
    assert all(len(frame) == codebook_count for frame in codec_frames)
    assert [i for i, frame in enumerate(codec_frames) if frame[0] == eos_token_id] == [
        len(codec_frames) - 1
    ], trace_directory
    decode_files = sorted(trace_directory.glob("decode-sampling-*.pt"))
    assert len(decode_files) == len(codec_frames) - 1, trace_directory
    decisions: list[DecodeDecision] = []
    for frame_index, sampling_path in enumerate(decode_files, 1):
        assert sampling_path.name == f"decode-sampling-{frame_index:03d}.pt"
        sampling = torch.load(sampling_path, map_location=device, weights_only=True)
        recorded_input = torch.load(
            trace_directory / f"backbone-decode-{frame_index:03d}.pt",
            weights_only=True,
        )
        assert int(sampling["seed"]) == sampling_seed
        position = int(sampling["position"])
        assert position == int(recorded_input["position"])
        if decisions:
            assert position == decisions[-1]["position"] + 1
        else:
            pass
        sampled_code, weights, ranks = readout(sampling)
        assert sampled_code == codec_frames[frame_index][0], sampling_path
        assert not bool(sampling["suppress_mask"][eos_token_id])
        decisions.append(
            {
                "frame_index": frame_index,
                "position": position,
                "code": sampled_code,
                "eos_rank": int((ranks[0] == eos_token_id).nonzero().item()) + 1,
                "eos_probability": float(weights[0, eos_token_id] / weights.sum()),
                "uses_pad": bool(recorded_input["uses_pad"]),
                "batch_rows": len(recorded_input["batch_request_ids"]),
            }
        )
    return decisions


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("witness_root", type=Path)
    parser.add_argument("--eos-token-id", required=True, type=int)
    parser.add_argument("--codebook-count", required=True, type=int)
    parser.add_argument("--sampling-seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    root = arguments.witness_root
    runs = {
        name: root / name
        for name in ("1-baseline", "2-baseline-eager", "3-prefill-gate")
    }
    manifests: dict[str, dict[str, WitnessResult]] = {}
    histories: dict[str, dict[str, list[list[int]]]] = {}
    decisions: dict[str, dict[str, list[DecodeDecision]]] = {}
    for name, directory in runs.items():
        records = json.loads((directory / "scored-witnesses.json").read_bytes())
        manifests[name] = {row["directory"]: row for row in records}
        histories[name] = json.loads((directory / "codec-histories.json").read_bytes())
        assert set(manifests[name]) == histories[name].keys()
        assert len(manifests[name]) == len(records)
        decisions[name] = {
            request_id: replay_decode(
                directory / request_id,
                frames,
                arguments.eos_token_id,
                arguments.codebook_count,
                arguments.sampling_seed,
                device,
            )
            for request_id, frames in histories[name].items()
        }
    assert all(
        histories[name].keys() == histories["1-baseline"].keys() for name in runs
    )
    report = {}
    for request_id in histories["1-baseline"]:
        comparisons = []
        for stock, candidate in (
            ("1-baseline", "2-baseline-eager"),
            ("1-baseline", "3-prefill-gate"),
            ("2-baseline-eager", "3-prefill-gate"),
        ):
            stock_frames = histories[stock][request_id]
            candidate_frames = histories[candidate][request_id]
            common_frames = zip(stock_frames, candidate_frames)
            differing_groups = {
                i: [
                    index
                    for index, (left, right) in enumerate(
                        zip(stock_frame, candidate_frame, strict=True)
                    )
                    if left != right
                ]
                for i, (stock_frame, candidate_frame) in enumerate(common_frames)
                if stock_frame != candidate_frame
            }
            earlier_last_frame = min(len(stock_frames), len(candidate_frames)) - 1
            stock_decision = decisions[stock][request_id][earlier_last_frame - 1]
            candidate_decision = decisions[candidate][request_id][
                earlier_last_frame - 1
            ]
            comparisons.append(
                {
                    "stock": stock,
                    "candidate": candidate,
                    "returned_text_equal": manifests[stock][request_id]["returned_text"]
                    == manifests[candidate][request_id]["returned_text"],
                    "waveform_equal": manifests[stock][request_id]["waveform_sha256"]
                    == manifests[candidate][request_id]["waveform_sha256"],
                    "first_differing_frame": next(iter(differing_groups), None),
                    "differing_groups": differing_groups,
                    "stock_at_earlier_eos": stock_decision,
                    "candidate_at_earlier_eos": candidate_decision,
                    "boundaries": compare_witnesses(
                        runs[stock] / request_id, runs[candidate] / request_id
                    ),
                }
            )
        report[request_id] = {
            "comparisons": comparisons,
            "decisions": {name: decisions[name][request_id] for name in runs},
        }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
else:
    pass
