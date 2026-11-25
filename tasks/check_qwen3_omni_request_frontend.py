# SPDX-License-Identifier: Apache-2.0
"""Verify request-local audio computation against native captures and singleton calls."""

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import difference
from qwen3_omni_request_frontend import (
    batched_request_frontend,
    request_frontend_forward,
)
from torch import nn

from sglang_omni.models.qwen3_omni.components.audio_encoder import (
    Qwen3OmniAudioEncoder,
    pack_padded_audio_features,
)


@dataclass(kw_only=True)
class FrontendCapture:
    inputs: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    def __call__(
        self, module: nn.Module, inputs: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        self.inputs.append((inputs[0].detach().cpu(), inputs[1].detach().cpu()))


@torch.inference_mode()
def main(root: Path, checkpoint: Path, sample_id: str, output: Path) -> None:
    assert not output.exists()
    torch.set_num_threads(1)
    encoder = Qwen3OmniAudioEncoder(
        str(checkpoint),
        device="cuda:0",
        dtype=torch.bfloat16,
        enable_layer_cuda_graph=True,
    ).eval()
    assert encoder.layer_graph_runner.has_graphs
    original_chunk_limit = encoder.audio_tower.conv_chunksize
    captures = {}
    snapshots = {}
    for arm in ("first", "second"):
        directory = root / arm / "captures/provenance" / sample_id
        tensor_path = directory / "audio-encoder.pt"
        snapshot = torch.load(tensor_path, map_location="cpu", weights_only=True)
        metadata = json.loads((directory / "audio-encoder.json").read_text())
        lengths = snapshot["audio_feature_lengths"].long().cpu()
        assert len(metadata["batch_request_ids"]) == lengths.numel()
        assert metadata["row_count"] == 1
        packed_features = pack_padded_audio_features(
            snapshot["input_features"].to(torch.bfloat16),
            snapshot["feature_attention_mask"],
            lengths,
        )
        native = encoder(
            input_features=packed_features,
            audio_feature_lengths=lengths,
        )["audio_embeds"].cpu()
        assert torch.equal(native, snapshot["audio_embeds"])
        snapshots[arm] = (packed_features, lengths, metadata)
        captures[arm] = sha256(tensor_path.read_bytes()).hexdigest()
    features, lengths, metadata = snapshots["first"]
    assert lengths.numel() == 3
    clips = list(features.split(lengths.tolist(), dim=-1))
    short_clip = clips[0][:, : encoder.audio_tower.n_window // 2]
    cases = [
        ("native_cohort", [[clip] for clip in clips], original_chunk_limit),
        (
            "reordered_cohort",
            [[clip] for clip in reversed(clips)],
            original_chunk_limit,
        ),
        ("multi_clip_request", [clips[:2], clips[2:]], original_chunk_limit),
        ("short_with_long_peer", [[short_clip], [clips[1]]], original_chunk_limit),
        (
            "multi_clip_short_support",
            [[short_clip, clips[1]], [clips[2]]],
            original_chunk_limit,
        ),
        ("convolution_split_boundary", [clips[:2], clips[2:]], 3),
    ]
    reports = {}
    for name, requests, chunk_limit in cases:
        encoder.audio_tower.conv_chunksize = chunk_limit
        request_clip_counts = [len(request) for request in requests]
        packed_features = torch.cat(
            [clip for request in requests for clip in request], dim=-1
        )
        clip_lengths = torch.tensor(
            [clip.shape[-1] for request in requests for clip in request]
        )
        singleton_outputs = []
        expected_frontends = []
        expected_segments = []
        for request in requests:
            request_features = torch.cat(request, dim=-1)
            request_lengths = torch.tensor([clip.shape[-1] for clip in request])
            capture = FrontendCapture()
            hook = encoder.audio_tower.layers.register_forward_pre_hook(capture)
            try:
                native_output = encoder(
                    input_features=request_features,
                    audio_feature_lengths=request_lengths,
                )["audio_embeds"].cpu()
            finally:
                hook.remove()
            assert len(capture.inputs) == 1
            expected_frontends.append(capture.inputs[0][0])
            expected_segments.append(capture.inputs[0][1].diff())
            local_output = request_frontend_forward(
                encoder,
                request_features,
                request_lengths,
                [len(request)],
            ).cpu()
            assert torch.equal(local_output, native_output), name
            singleton_outputs.append(native_output)
        frontend = batched_request_frontend(
            encoder,
            packed_features,
            clip_lengths,
            request_clip_counts,
        )
        assert torch.equal(
            frontend.hidden_states.cpu(), torch.cat(expected_frontends)
        ), name
        assert torch.equal(
            frontend.cumulative_sequence_lengths.diff(), torch.cat(expected_segments)
        ), name
        local_output = request_frontend_forward(
            encoder,
            packed_features,
            clip_lengths,
            request_clip_counts,
        ).cpu()
        native_output = encoder(
            input_features=packed_features,
            audio_feature_lengths=clip_lengths,
        )["audio_embeds"].cpu()
        assert local_output.shape == native_output.shape
        assert torch.isfinite(local_output).all()
        timings = {"native": [], "request_local": []}
        for round_index in range(4):
            policies = (
                ("native", "request_local")
                if round_index % 2 == 0
                else ("request_local", "native")
            )
            for policy in policies:
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    if policy == "native":
                        result = encoder(
                            input_features=packed_features,
                            audio_feature_lengths=clip_lengths,
                        )["audio_embeds"]
                    else:
                        result = request_frontend_forward(
                            encoder, packed_features, clip_lengths, request_clip_counts
                        )
                torch.cuda.synchronize()
                timings[policy].append((time.perf_counter() - start) * 1000 / 20)
                expected = native_output if policy == "native" else local_output
                assert torch.equal(result.cpu(), expected), (name, policy)
        reports[name] = {
            "clip_lengths": clip_lengths.tolist(),
            "request_clip_counts": request_clip_counts,
            "convolution_chunk_limit": chunk_limit,
            "singleton_native_exact": True,
            "frontend_matches_native_singletons": True,
            "segments_match_native_singletons": True,
            "repeated_whole_encoder_exact": True,
            "batched_tail_versus_singletons": difference(
                local_output, torch.cat(singleton_outputs)
            ),
            "native_versus_request_local": difference(native_output, local_output),
            "wall_milliseconds_rounds": timings,
            "median_wall_milliseconds": {
                policy: statistics.median(values) for policy, values in timings.items()
            },
        }
    encoder.audio_tower.conv_chunksize = original_chunk_limit
    witness_outputs = {}
    for arm, (packed_features, clip_lengths, metadata) in snapshots.items():
        local_output = request_frontend_forward(
            encoder, packed_features, clip_lengths, [1] * clip_lengths.numel()
        ).cpu()
        witness_outputs[arm] = local_output[
            metadata["token_start"] : metadata["token_end"]
        ]
    assert torch.equal(witness_outputs["first"], witness_outputs["second"])
    report = {
        "capture_sha256": captures,
        "witness_cross_cohort_exact": True,
        "cases": reports,
    }
    with output.open("x") as destination:
        json.dump(report, destination, indent=2)
        destination.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("sample_id")
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    main(arguments.root, arguments.checkpoint, arguments.sample_id, arguments.output)
else:
    pass
