# SPDX-License-Identifier: Apache-2.0
"""Replay captured encoder batches and isolate frontend batch-shape effects."""

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from compare_qwen3_omni_native_trajectories import difference
from torch import nn
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder


@dataclass(kw_only=True)
class EncoderLaunch:
    inputs: torch.Tensor
    outputs: torch.Tensor
    selected_chunks: slice


@torch.inference_mode()
def main(root: Path, checkpoint: Path, sample_id: str, boundary_carry: bool) -> None:
    torch.set_num_threads(1)
    encoder = Qwen3OmniAudioEncoder(
        str(checkpoint),
        device="cuda:0",
        dtype=torch.bfloat16,
        enable_layer_cuda_graph=True,
    ).eval()
    tower = encoder.audio_tower
    operators = {
        "conv2d1": tower.conv2d1,
        "conv2d2": tower.conv2d2,
        "conv2d3": tower.conv2d3,
        "conv_out": tower.conv_out,
    }
    captured: dict[str, dict[str, EncoderLaunch]] = {}
    for arm in ("first", "second"):
        directory = root / arm / "captures" / "provenance" / sample_id
        snapshot = torch.load(
            directory / "audio-encoder.pt", map_location="cpu", weights_only=True
        )
        metadata = json.loads((directory / "audio-encoder.json").read_text())
        chunk_frames = tower.n_window * 2
        chunk_counts = (
            snapshot["audio_feature_lengths"] + chunk_frames - 1
        ) // chunk_frames
        first_chunk = int(chunk_counts[: metadata["row_start"]].sum())
        last_chunk = first_chunk + int(
            chunk_counts[
                metadata["row_start"] : metadata["row_start"] + metadata["row_count"]
            ].sum()
        )
        offsets = dict.fromkeys(operators, 0)
        launches: dict[str, EncoderLaunch] = {}

        def record(
            module: nn.Module,
            inputs: tuple[torch.Tensor],
            output: torch.Tensor,
            *,
            name: str,
        ) -> None:
            assert module is operators[name]
            start = offsets[name]
            offsets[name] += inputs[0].shape[0]
            if start <= first_chunk and last_chunk <= offsets[name]:
                assert name not in launches
                launches[name] = EncoderLaunch(
                    inputs=inputs[0].clone(),
                    outputs=output.clone(),
                    selected_chunks=slice(first_chunk - start, last_chunk - start),
                )
            else:
                pass

        hooks = [
            module.register_forward_hook(partial(record, name=name))
            for name, module in operators.items()
        ]
        try:
            result = encoder(
                input_features=snapshot["input_features"],
                feature_attention_mask=snapshot["feature_attention_mask"],
                audio_feature_lengths=snapshot["audio_feature_lengths"],
            )
        finally:
            for hook in hooks:
                hook.remove()
        assert result.keys() == {
            "audio_embeds",
            "audio_feature_lengths",
            "audio_output_lengths",
        }
        for name, tensor in result.items():
            assert torch.equal(tensor.cpu(), snapshot[name]), (arm, name)
        assert launches.keys() == operators.keys()
        assert all(count == int(chunk_counts.sum()) for count in offsets.values())
        captured[arm] = launches

    if boundary_carry:
        sources = [
            root / arm / "captures" / "provenance" / sample_id
            for arm in ("first", "second")
        ]
        snapshots = [
            torch.load(
                source / "audio-encoder.pt", map_location="cpu", weights_only=True
            )
            for source in sources
        ]
        metadata = [
            json.loads((source / "audio-encoder.json").read_text())
            for source in sources
        ]
        target_slices = [
            slice(record["token_start"], record["token_end"]) for record in metadata
        ]
        reports = {}
        for name in ("conv2d2", "conv_out"):
            recipient, donor = (captured[arm][name] for arm in ("first", "second"))
            calls = 0

            def carry(
                module: nn.Module, inputs: tuple[torch.Tensor], output: torch.Tensor
            ) -> torch.Tensor:
                nonlocal calls
                assert module is operators[name]
                assert torch.equal(inputs[0], recipient.inputs)
                assert torch.equal(output, recipient.outputs)
                calls += 1
                substituted = output.clone()
                substituted[recipient.selected_chunks].copy_(
                    donor.outputs[donor.selected_chunks]
                )
                return substituted

            hook = operators[name].register_forward_hook(carry)
            try:
                result = encoder(
                    input_features=snapshots[0]["input_features"],
                    feature_attention_mask=snapshots[0]["feature_attention_mask"],
                    audio_feature_lengths=snapshots[0]["audio_feature_lengths"],
                )["audio_embeds"].cpu()
            finally:
                hook.remove()
            assert calls == 1
            peers = torch.ones(result.shape[0], dtype=torch.bool)
            peers[target_slices[0]] = False
            assert torch.equal(result[peers], snapshots[0]["audio_embeds"][peers])
            reports[name] = {
                "peer_outputs_unchanged": True,
                "versus_recipient": difference(
                    result[target_slices[0]],
                    snapshots[0]["audio_embeds"][target_slices[0]],
                ),
                "versus_donor": difference(
                    result[target_slices[0]],
                    snapshots[1]["audio_embeds"][target_slices[1]],
                ),
            }
        with (root / "encoder-boundary-carry.json").open("x") as output:
            json.dump(reports, output, indent=2)
            output.write("\n")
        print(json.dumps(reports, indent=2))
        return
    else:
        pass

    reports = {}
    for name, module in operators.items():
        left, right = (captured[arm][name] for arm in ("first", "second"))
        for launch in (left, right):
            assert torch.equal(module(launch.inputs), launch.outputs), name
        held_inputs = right.inputs.clone()
        held_inputs[right.selected_chunks].copy_(left.inputs[left.selected_chunks])
        held_outputs = module(held_inputs)
        peers = torch.ones(
            held_outputs.shape[0], dtype=torch.bool, device=held_outputs.device
        )
        peers[right.selected_chunks] = False
        assert torch.equal(held_outputs[peers], right.outputs[peers]), name
        selected_left = left.outputs[left.selected_chunks]
        selected_held = held_outputs[right.selected_chunks]
        report = {
            "physical_input_shapes": [
                list(launch.inputs.shape) for launch in (left, right)
            ],
            "native_input": difference(
                left.inputs[left.selected_chunks], right.inputs[right.selected_chunks]
            ),
            "native_output": difference(
                selected_left, right.outputs[right.selected_chunks]
            ),
            "held_input_output": difference(selected_left, selected_held),
            "peer_outputs_unchanged": True,
        }
        if not torch.equal(selected_left, selected_held):
            kernel_names = {}
            for arm, operands in (("first", left.inputs), ("second_held", held_inputs)):
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                ) as profiler:
                    replayed = module(operands)
                    torch.cuda.synchronize()
                expected = left.outputs if arm == "first" else held_outputs
                assert torch.equal(replayed, expected)
                kernel_names[arm] = [
                    event.name
                    for event in profiler.events()
                    if event.device_type == DeviceType.CUDA
                ]
            report["cuda_kernel_names"] = kernel_names
            if isinstance(module, nn.Conv2d):
                raw_outputs = [
                    F.conv2d(
                        operands,
                        module.weight,
                        None,
                        module.stride,
                        module.padding,
                        module.dilation,
                        module.groups,
                    )
                    for operands in (left.inputs, held_inputs)
                ]
                assert module.bias is not None and module.groups == 1
                for raw, expected in zip(
                    raw_outputs, (left.outputs, held_outputs), strict=True
                ):
                    assert torch.equal(raw + module.bias.view(1, -1, 1, 1), expected)
                raw_left = raw_outputs[0][left.selected_chunks]
                raw_right = raw_outputs[1][right.selected_chunks]
                coordinates = (raw_left != raw_right).nonzero()
                assert coordinates.numel() > 0
                chunks, channels, rows, columns = coordinates.T
                patches = F.unfold(
                    left.inputs[left.selected_chunks],
                    module.kernel_size,
                    module.dilation,
                    module.padding,
                    module.stride,
                )
                operands = patches[chunks, :, rows * raw_left.shape[-1] + columns]
                weights = module.weight.flatten(1)[channels]
                reference = (operands.double() * weights.double()).sum(dim=1)
                raw_values = [
                    raw[tuple(coordinates.T)] for raw in (raw_left, raw_right)
                ]
                report["raw_convolution_fp64"] = {
                    "changed_values": len(coordinates),
                    "matches_rounded_reference": [
                        int((values == reference.to(values.dtype)).sum())
                        for values in raw_values
                    ],
                    "mean_absolute_error": [
                        float((values.double() - reference).abs().mean())
                        for values in raw_values
                    ],
                    "bias_addition_exact": True,
                }
            else:
                pass
        else:
            pass
        reports[name] = report
    with (root / "encoder-operator-replay.json").open("x") as output:
        json.dump(
            {"entire_native_encoder_outputs_exact": True, "operators": reports},
            output,
            indent=2,
        )
        output.write("\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("sample_id")
    parser.add_argument("--boundary-carry", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.root,
        arguments.checkpoint,
        arguments.sample_id,
        arguments.boundary_carry,
    )
else:
    pass
