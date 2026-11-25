# SPDX-License-Identifier: Apache-2.0
"""Check request-local frontend arithmetic and whole-encoder latency on native captures."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import difference
from torch import nn

from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder


class RequestLocalFrontend(nn.Module):
    def __init__(
        self, operation: nn.Conv2d | nn.Linear, chunk_counts: list[int]
    ) -> None:
        super().__init__()
        self.operation = operation
        self.chunk_counts = chunk_counts

    @property
    def weight(self) -> nn.Parameter:
        return self.operation.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[0] == sum(self.chunk_counts)
        return torch.cat(
            [self.operation(chunk) for chunk in inputs.split(self.chunk_counts)], dim=0
        )


@torch.inference_mode()
def main(root: Path, checkpoint: Path, sample_id: str, repetitions: int) -> None:
    assert repetitions > 0
    torch.set_num_threads(1)
    encoder = Qwen3OmniAudioEncoder(
        str(checkpoint),
        device="cuda:0",
        dtype=torch.bfloat16,
        enable_layer_cuda_graph=True,
    ).eval()
    tower = encoder.audio_tower
    native_operations = {
        name: getattr(tower, name)
        for name in ("conv2d1", "conv2d2", "conv2d3", "conv_out")
    }
    reports = {}
    selected_outputs = {}
    for arm in ("first", "second"):
        directory = root / arm / "captures" / "provenance" / sample_id
        snapshot = torch.load(
            directory / "audio-encoder.pt", map_location="cpu", weights_only=True
        )
        metadata = json.loads((directory / "audio-encoder.json").read_text())
        target_slice = slice(metadata["token_start"], metadata["token_end"])
        chunk_frames = tower.n_window * 2
        chunk_counts = (
            (snapshot["audio_feature_lengths"] + chunk_frames - 1) // chunk_frames
        ).tolist()
        assert sum(chunk_counts) <= tower.conv_chunksize
        inputs = {
            name: snapshot[name]
            for name in (
                "input_features",
                "feature_attention_mask",
                "audio_feature_lengths",
            )
        }
        native_output = encoder(**inputs)["audio_embeds"].cpu()
        assert torch.equal(native_output, snapshot["audio_embeds"])
        local_operations = {
            name: RequestLocalFrontend(operation, chunk_counts)
            for name, operation in native_operations.items()
        }
        timings = {policy: [] for policy in ("native", "request_local")}
        expected_outputs = {"native": native_output}
        for round_index in range(4):
            policies = ("native", "request_local")
            if round_index % 2:
                policies = tuple(reversed(policies))
            else:
                pass
            for policy in policies:
                operations = (
                    native_operations if policy == "native" else local_operations
                )
                for name, operation in operations.items():
                    setattr(tower, name, operation)
                output = encoder(**inputs)["audio_embeds"].cpu()
                if policy in expected_outputs:
                    assert torch.equal(output, expected_outputs[policy])
                else:
                    expected_outputs[policy] = output
                for _ in range(5):
                    encoder(**inputs)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                for _ in range(repetitions):
                    encoder(**inputs)
                end.record()
                end.synchronize()
                timings[policy].append(
                    {
                        "device_milliseconds": start.elapsed_time(end) / repetitions,
                        "wall_milliseconds": (time.perf_counter() - wall_start)
                        * 1000
                        / repetitions,
                    }
                )
        for name, operation in native_operations.items():
            setattr(tower, name, operation)
        restored = encoder(**inputs)["audio_embeds"].cpu()
        assert torch.equal(restored, native_output)
        selected_outputs[arm] = expected_outputs["request_local"][target_slice]
        reports[arm] = {
            "chunk_counts": chunk_counts,
            "request_local_versus_native": difference(
                expected_outputs["request_local"], native_output
            ),
            "native_restore_exact": True,
            "four_repeated_outputs_exact": True,
            "latency_rounds": timings,
            "median_wall_milliseconds": {
                policy: statistics.median(
                    trial["wall_milliseconds"] for trial in trials
                )
                for policy, trials in timings.items()
            },
        }
    reports["selected_cross_batch"] = difference(
        selected_outputs["first"], selected_outputs["second"]
    )
    with (root / "request-local-frontend.json").open("x") as output:
        json.dump(reports, output, indent=2)
        output.write("\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("sample_id")
    parser.add_argument("--repetitions", type=int, default=20)
    arguments = parser.parse_args()
    main(
        arguments.root, arguments.checkpoint, arguments.sample_id, arguments.repetitions
    )
else:
    pass
