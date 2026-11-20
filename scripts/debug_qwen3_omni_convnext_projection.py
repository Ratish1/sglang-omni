# SPDX-License-Identifier: Apache-2.0
"""Cross the observed ConvNeXt projection while holding the native codec window."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from scripts.debug_qwen3_omni_codec_waveform import (
    capture_cold_window,
    load_native_scheduler,
)
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)


def difference(first: torch.Tensor, second: torch.Tensor) -> dict[str, int | float]:
    assert first.shape == second.shape
    delta = first.double() - second.double()
    return {
        "changed_elements": int(delta.count_nonzero()),
        "max_abs_difference": float(delta.abs().max()),
        "rms_difference": float(delta.square().mean().sqrt()),
    }


def decode_projection(
    scheduler: Code2WavScheduler,
    codes: torch.Tensor,
    replacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codec = scheduler.incremental_codec
    pool = scheduler.codec_pool
    assert codec is not None and pool is not None
    projection = codec.model.upsample[1][1].pwconv2
    operands: list[torch.Tensor] = []

    def record_projection(
        module: torch.nn.Linear,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        assert module is projection
        (hidden_states,) = inputs
        operands.extend((hidden_states.clone(), output.clone()))
        if replacement is not None:
            assert replacement.shape == output[0].shape
            output[0].copy_(replacement)
        else:
            pass

    with projection.register_forward_hook(record_projection):
        waveform = codec.decode_slots(
            codes,
            pool,
            pool.stage_index(list(range(codes.size(0)))),
            first_decode=True,
        )
    assert len(operands) == 2
    return (*operands, waveform)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--witness-directory", type=Path, required=True)
    parser.add_argument("--cold-window-report", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--trace-waveform-residual", action="store_true")
    arguments = parser.parse_args()

    scheduler = load_native_scheduler(
        arguments.server_config, arguments.model_path, arguments.gpu_id
    )
    cold_window = json.loads(arguments.cold_window_report.read_bytes())
    histories = json.loads(
        (arguments.witness_directory / "codec-histories.json").read_bytes()
    )
    frame_count = scheduler.initial_codec_chunk_frames or scheduler.stream_chunk_size
    codes = (
        torch.tensor(
            [
                histories[request_id][:frame_count]
                for request_id in cold_window["peer_request_ids"]
            ],
            device=scheduler.device,
            dtype=torch.long,
        )
        .transpose(1, 2)
        .contiguous()
    )
    assert list(codes.shape) == cold_window["code_shape"]
    solo_input, solo_output, solo_waveform = decode_projection(scheduler, codes[:1])
    peer_input, peer_output, peer_waveform = decode_projection(scheduler, codes)
    assert torch.equal(solo_input[0], peer_input[0])
    expected_waveform = cold_window["boundaries"][-1]
    assert expected_waveform["boundary"] == "waveform"
    observed = difference(solo_waveform, peer_waveform[:1])
    assert observed["changed_elements"] == expected_waveform["different_elements"]
    assert observed["max_abs_difference"] == expected_waveform["max_abs_difference"]
    _, _, restored_peer_waveform = decode_projection(scheduler, codes, solo_output[0])
    _, _, changed_solo_waveform = decode_projection(
        scheduler, codes[:1], peer_output[0]
    )
    _, _, repeated_peer_waveform = decode_projection(scheduler, codes)
    assert torch.equal(peer_waveform, repeated_peer_waveform)
    assert torch.equal(peer_waveform[1:], restored_peer_waveform[1:])

    projection = scheduler.model.upsample[1][1].pwconv2
    if arguments.trace_waveform_residual:
        serial_boundaries = capture_cold_window(scheduler, codes[:1])
        original_projection = projection.forward

        def fixed_projection(hidden_states: torch.Tensor) -> torch.Tensor:
            projected = original_projection(hidden_states)
            projected[0].copy_(solo_output[0])
            return projected

        with patch.object(projection, "forward", new=fixed_projection):
            restored_boundaries = capture_cold_window(scheduler, codes)
        assert list(serial_boundaries) == list(restored_boundaries)
        assert torch.equal(
            restored_boundaries["waveform"], restored_peer_waveform[:1].cpu()
        )
        assert torch.equal(serial_boundaries["waveform"], solo_waveform.cpu())
        print(
            json.dumps(
                {
                    "request_id": cold_window["request_id"],
                    "residual_boundaries": {
                        name: difference(serial_boundaries[name], tensor)
                        for name, tensor in restored_boundaries.items()
                    },
                }
            )
        )
        return
    else:
        pass
    fp64_reference = F.linear(
        solo_input.double(), projection.weight.double(), projection.bias.double()
    )[0]
    rounded_reference = fp64_reference.to(solo_output.dtype)
    matmul = torch.backends.cuda.matmul
    original_policy = matmul.allow_bf16_reduced_precision_reduction
    operators = {}
    try:
        for name, allow_reduced_precision in (
            ("native", original_policy),
            ("fp32_reduction", False),
        ):
            matmul.allow_bf16_reduced_precision_reduction = allow_reduced_precision
            outputs = {}
            operators[name] = {}
            for geometry, hidden_states in (
                ("solo", solo_input),
                ("peers", peer_input),
            ):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                ) as trace:
                    projected = projection(hidden_states)[0]
                    torch.cuda.synchronize(scheduler.device)
                outputs[geometry] = projected
                operators[name][geometry] = {
                    "cuda_kernels": sorted(
                        {
                            event.name
                            for event in trace.events()
                            if event.device_type.name == "CUDA"
                        }
                    ),
                    "vs_fp64": difference(projected, fp64_reference),
                    "vs_rounded_fp64": difference(projected, rounded_reference),
                }
            if name == "native":
                assert torch.equal(outputs["solo"], solo_output[0])
                assert torch.equal(outputs["peers"], peer_output[0])
            else:
                pass
            operators[name]["solo_vs_peers"] = difference(
                outputs["solo"], outputs["peers"]
            )
    finally:
        matmul.allow_bf16_reduced_precision_reduction = original_policy
    print(
        json.dumps(
            {
                "request_id": cold_window["request_id"],
                "native_bf16_reduction_policy": original_policy,
                "operator": operators,
                "waveform_solo_vs_peers": observed,
                "waveform_solo_vs_peers_with_solo_projection": difference(
                    solo_waveform, restored_peer_waveform[:1]
                ),
                "waveform_peers_vs_solo_with_peer_projection": difference(
                    peer_waveform[:1], changed_solo_waveform
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
