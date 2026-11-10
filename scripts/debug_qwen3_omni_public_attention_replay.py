# SPDX-License-Identifier: Apache-2.0
"""Replay actual public prefill SDPA operands without model generation."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from scripts.qwen3_omni_reference_observers import SDPAPrefillCapture


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--repeat-count", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("default", "flash", "cudnn", "math"), default="default"
    )
    arguments = parser.parse_args()
    assert arguments.repeat_count > 1
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    capture: SDPAPrefillCapture = torch.load(
        arguments.capture, map_location="cpu", weights_only=True
    )
    tensors = capture["sdpa_prefill"][arguments.layer]
    layouts = capture["sdpa_layouts"][arguments.layer]
    configuration = capture["sdpa_configurations"][arguments.layer]
    assert "attention_mask" not in tensors and configuration["dropout"] == 0
    operands: dict[str, torch.Tensor] = {}
    for name in ("query", "key", "value"):
        layout = layouts[name]
        assert layout["storage_offset_in_elements"] == 0
        tensor = torch.empty_strided(
            layout["shape"],
            layout["strides_in_elements"],
            dtype=tensors[name].dtype,
            device=arguments.gpu,
        )
        tensor.copy_(tensors[name])
        operands[name] = tensor
    query, key, value = (operands[name] for name in ("query", "key", "value"))
    assert key.shape[-1] == value.shape[-1] <= 256
    causal = configuration["is_causal"]
    if causal is None:
        causal = configuration["module_is_causal"]
    else:
        pass
    causal = query.shape[2] > 1 and causal
    grouped_query = configuration["key_value_groups"] > 1
    with sdpa_kernel(SDPBackend.MATH):
        oracle = (
            functional.scaled_dot_product_attention(
                query.double(),
                key.double(),
                value.double(),
                is_causal=causal,
                scale=configuration["scaling"],
                enable_gqa=grouped_query,
            )
            .transpose(1, 2)
            .contiguous()
        )
    backend = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }
    context = (
        nullcontext()
        if arguments.backend == "default"
        else sdpa_kernel(backend[arguments.backend])
    )
    outputs: list[torch.Tensor] = []
    with context:
        for _ in range(arguments.repeat_count):
            outputs.append(
                functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=causal,
                    scale=configuration["scaling"],
                    enable_gqa=grouped_query,
                )
                .transpose(1, 2)
                .contiguous()
            )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            profiled_output = (
                functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=causal,
                    scale=configuration["scaling"],
                    enable_gqa=grouped_query,
                )
                .transpose(1, 2)
                .contiguous()
            )
    profiler.export_chrome_trace(str(arguments.output_directory / "profile.json"))
    captured_output = tensors["output"].to(arguments.gpu)
    reports = []
    for index, output in enumerate([captured_output, *outputs, profiled_output]):
        difference = output.double() - oracle
        reports.append(
            {
                "index": index,
                "kind": (
                    "captured"
                    if index == 0
                    else "profiled" if index == len(outputs) + 1 else "replay"
                ),
                "exact_captured": torch.equal(output, captured_output),
                "exact_first_replay": torch.equal(output, outputs[0]),
                "maximum_absolute_error_to_float64": float(difference.abs().max()),
                "relative_l2_error_to_float64": float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(oracle)
                ),
                "different_values_from_capture": int((output != captured_output).sum()),
            }
        )
    torch.save(
        {
            "outputs": [output.cpu() for output in outputs],
            "profiled_output": profiled_output.cpu(),
            "oracle": oracle.cpu(),
        },
        arguments.output_directory / "outputs.pt",
    )
    report = {
        "capture": str(arguments.capture),
        "layer": arguments.layer,
        "backend": arguments.backend,
        "configuration": configuration,
        "layouts": layouts,
        "comparisons": reports,
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
