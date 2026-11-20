# SPDX-License-Identifier: Apache-2.0
"""Check every cold-window convolution on an exact row of its native peer operands."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.debug_qwen3_omni_codec_waveform import load_native_scheduler
from scripts.debug_qwen3_omni_convnext_projection import decode_projection, difference
from sglang_omni.models.qwen3_omni.components import incremental_codec

RESIDUAL_CONVOLUTION = "decoder.2.block.1.conv"


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--witness-directory", type=Path, required=True)
    parser.add_argument("--cold-window-report", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
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
    native_convolution = incremental_codec.channels_last_conv1d
    native_transposed_convolution = incremental_codec.channels_last_conv_transpose1d
    module_names = {module: name for name, module in scheduler.model.named_modules()}
    operator_differences = {}
    _, solo_activation, _ = decode_projection(scheduler, codes[:1])
    _, _, expected_output = decode_projection(scheduler, codes, solo_activation[0])
    profiler_operands: list[torch.Tensor] = []

    def convolution_with_shadow(
        hidden_states: torch.Tensor,
        convolution: torch.nn.Conv1d,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        samples: int,
    ) -> torch.Tensor:
        output = native_convolution(hidden_states, convolution, weight, bias, samples)
        solo_output = native_convolution(
            hidden_states[:1], convolution, weight, bias, samples
        )
        operator_differences[module_names[convolution]] = difference(
            output[:1], solo_output
        )
        return output

    def transposed_convolution_with_shadow(
        hidden_states: torch.Tensor,
        convolution: torch.nn.ConvTranspose1d,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        output = native_transposed_convolution(hidden_states, convolution, weight, bias)
        solo_output = native_transposed_convolution(
            hidden_states[:1], convolution, weight, bias
        )
        name = module_names[convolution]
        operator_differences[name] = difference(output[:1], solo_output)
        if name == RESIDUAL_CONVOLUTION:
            assert bias is None
            profiler_operands.extend(
                (hidden_states.clone(), output[:1].clone(), solo_output.clone())
            )
        else:
            pass
        return output

    with (
        patch.object(
            incremental_codec, "channels_last_conv1d", new=convolution_with_shadow
        ),
        patch.object(
            incremental_codec,
            "channels_last_conv_transpose1d",
            new=transposed_convolution_with_shadow,
        ),
    ):
        _, _, observed_output = decode_projection(scheduler, codes, solo_activation[0])
    assert torch.equal(observed_output, expected_output)
    peer_input, peer_output, solo_output = profiler_operands
    convolution = scheduler.model.get_submodule(RESIDUAL_CONVOLUTION)
    assert isinstance(convolution, torch.nn.ConvTranspose1d)
    fp64 = native_transposed_convolution(
        peer_input[:1].double(), convolution, convolution.weight.double(), None
    ).to(peer_output.dtype)
    profiler_report = {}
    for name, operands, observed in (
        ("single_row", peer_input[:1], solo_output),
        ("peer_rows", peer_input, peer_output),
    ):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as trace:
            replayed = native_transposed_convolution(
                operands, convolution, convolution.weight, None
            )[:1]
            torch.cuda.synchronize(scheduler.device)
        assert torch.equal(replayed, observed)
        profiler_report[name] = {
            "input_shape": list(operands.shape),
            "vs_rounded_fp64": difference(replayed, fp64),
            "cuda_kernels": sorted(
                {
                    event.name
                    for event in trace.events()
                    if event.device_type.name == "CUDA"
                }
            ),
        }
    print(
        json.dumps(
            {
                "request_id": cold_window["request_id"],
                "conv_with_held_row": operator_differences,
                "profiled_convolution": RESIDUAL_CONVOLUTION,
                "profile": profiler_report,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
else:
    pass
