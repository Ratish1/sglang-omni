"""Compare every loaded native Talker parameter with the checkpoint model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from sglang_omni.models.qwen3_omni.stages import create_talker_ar_executor_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-running-requests", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    scheduler = create_talker_ar_executor_from_config(
        arguments.model_path,
        gpu_id=0,
        max_running_requests=arguments.max_running_requests,
        total_gpu_memory_fraction=arguments.memory_fraction,
        code2wav_in_process=True,
    )
    native = scheduler.model_runner.model
    assert all(
        layer.mlp.experts.runner_backend.is_flashinfer_cutlass()
        for layer in native.model.layers
    )
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": "cpu"},
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference_parameters = dict(reference.named_parameters())
    consumed_reference_names: set[str] = set()
    comparisons = []
    with torch.inference_mode():
        for name, parameter in native.named_parameters():
            if name.endswith(".experts.w13_weight"):
                reference_names = [name.removesuffix("w13_weight") + "gate_up_proj"]
                gate, up = reference_parameters[reference_names[0]].chunk(2, dim=1)
                expected = torch.cat((up, gate), dim=1)
            elif name.endswith(".experts.w2_weight"):
                reference_names = [name.removesuffix("w2_weight") + "down_proj"]
                expected = reference_parameters[reference_names[0]]
            elif ".qkv_proj." in name:
                reference_names = [
                    name.replace(".qkv_proj.", f".{projection}.")
                    for projection in ("q_proj", "k_proj", "v_proj")
                ]
                expected = torch.cat(
                    [reference_parameters[value] for value in reference_names], dim=0
                )
            elif ".gate_up_proj." in name:
                reference_names = [
                    name.replace(".gate_up_proj.", f".{projection}.")
                    for projection in ("gate_proj", "up_proj")
                ]
                expected = torch.cat(
                    [reference_parameters[value] for value in reference_names], dim=0
                )
            else:
                reference_names = [name]
                expected = reference_parameters[name]
            consumed_reference_names.update(reference_names)
            expected = expected.to(dtype=parameter.dtype)
            actual = parameter.detach().cpu()
            comparison = {
                "name": name,
                "native_shape": list(actual.shape),
                "reference_shape": list(expected.shape),
                "element_count": actual.numel(),
                "equal": torch.equal(actual, expected),
            }
            comparisons.append(comparison)
            print(json.dumps(comparison), flush=True)
    unused_reference_names = sorted(
        reference_parameters.keys() - consumed_reference_names
    )
    report = {
        "model_path": arguments.model_path,
        "native_parameter_count": len(comparisons),
        "native_parameter_elements": sum(row["element_count"] for row in comparisons),
        "unequal_parameter_names": [
            row["name"] for row in comparisons if not row["equal"]
        ],
        "unused_reference_names": unused_reference_names,
        "comparisons": comparisons,
    }
    arguments.output_file.write_text(json.dumps(report, indent=2))
    assert not report["unequal_parameter_names"], report["unequal_parameter_names"]
    assert not unused_reference_names, unused_reference_names


if __name__ == "__main__":
    main()
