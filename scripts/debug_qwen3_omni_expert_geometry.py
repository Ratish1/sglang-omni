"""Isolate request grouping with the same observed native expert algorithms."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from huggingface_hub import hf_hub_download
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights
from scripts.qwen3_omni_talker_expert_runtime import (
    NativeRuntime,
    TacticChoice,
    load_native_expert_weights,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--trace-directory", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config
    register_checkpoint_conversion_mapping(
        configuration.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=configuration,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="eager",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"]
    reference.eval()
    audit_checkpoint_weights(
        reference,
        Path(
            hf_hub_download(
                arguments.model_path, "config.json", revision=arguments.model_revision
            )
        ).parent,
        arguments.output_directory / "weight-audit.json",
    )
    records = []
    with (arguments.output_directory / "layers.jsonl").open("w") as output:
        for directory in arguments.trace_directory:
            runtime = NativeRuntime.model_validate_json(
                (directory.parent / "moe-runtime.json").read_bytes()
            )
            choices = [
                TacticChoice.model_validate(choice)
                for choice in json.loads(
                    (directory.parent / "moe-choices-final.json").read_text()
                )
            ]
            assert choices[: len(runtime.tactic_choices)] == runtime.tactic_choices
            native_weights = load_native_expert_weights(reference, runtime)
            finished = torch.load(directory / "finished.pt", weights_only=True)
            files = sorted(directory.glob("*-operators.pt"))
            assert [filename.name for filename in files] == [
                f"{frame:04d}-operators.pt"
                for frame in range(len(finished["output_ids"]))
            ]
            exact_isolated_batches = 0
            comparisons = 0
            for frame, filename in enumerate(files):
                snapshots = torch.load(filename, weights_only=True)
                assert len(snapshots) == len(native_weights)
                for layer_index, (snapshot, (packed, down)) in enumerate(
                    zip(snapshots, native_weights, strict=True)
                ):
                    hidden_states = snapshot["batch_mlp_normalized"].to(
                        reference.device
                    )
                    indices = snapshot["batch_expert_indices"].to(reference.device)
                    weights = snapshot["batch_expert_weights"].to(reference.device)
                    expected = snapshot["batch_routed_output"].to(reference.device)
                    assert (
                        hidden_states.dtype
                        == packed.dtype
                        == down.dtype
                        == torch.bfloat16
                    )
                    assert (
                        indices.dtype == torch.int32 and weights.dtype == torch.float32
                    )
                    assert all(
                        tensor.is_contiguous()
                        for tensor in (hidden_states, indices, weights, packed, down)
                    )
                    if frame == 0:
                        start, stop = snapshot["prefix_token_rows"].tolist()
                    else:
                        start = int(snapshot["batch_selected_row"])
                        stop = start + 1
                    assert 0 <= start < stop <= len(hidden_states)
                    shapes = [
                        list(hidden_states.shape),
                        list(packed.shape),
                        [],
                        list(down.shape),
                        [],
                    ]
                    profiles = []
                    for operation in (
                        "trtllm::fused_moe::gemm1",
                        "trtllm::fused_moe::gemm2",
                    ):
                        tactics = {
                            choice.tactic
                            for choice in choices
                            if choice.operation == operation
                            and choice.tensor_shapes == shapes
                            and choice.capturing == bool(snapshot["batch_is_graph"])
                            and not choice.tuning
                        }
                        assert len(tactics) == 1, (
                            directory.name,
                            frame,
                            layer_index,
                            operation,
                        )
                        profiles.append(tactics.pop())
                    replayed = []
                    for selected in (slice(None), slice(start, stop)):
                        replay_inputs = hidden_states[selected].contiguous().clone()
                        replayed.append(
                            cutlass_fused_moe(
                                input=replay_inputs,
                                token_selected_experts=indices[selected].contiguous(),
                                token_final_scales=weights[selected].contiguous(),
                                fc1_expert_weights=packed,
                                fc2_expert_weights=down,
                                output_dtype=torch.bfloat16,
                                quant_scales=None,
                                activation_type=ActivationType.Swiglu,
                                tune_max_num_tokens=runtime.tune_max_num_tokens,
                                use_fused_finalize=runtime.use_fused_finalize,
                                profile_ids=profiles,
                            )[0]
                        )
                        assert torch.equal(replay_inputs, hidden_states[selected])
                    assert torch.equal(replayed[0], expected), (
                        directory.name,
                        frame,
                        layer_index,
                    )
                    difference = compare_tensors(expected[start:stop], replayed[1])
                    exact_isolated_batches += difference.exact
                    comparisons += 1
                    output.write(
                        json.dumps(
                            {
                                "request_id": directory.name,
                                "frame": frame,
                                "layer": layer_index,
                                "batch_tokens": len(hidden_states),
                                "request_tokens": stop - start,
                                "profiles": profiles,
                                "whole_native_replay_exact": True,
                                "isolated_vs_whole": asdict(difference),
                            }
                        )
                        + "\n"
                    )
                print(
                    json.dumps(
                        {
                            "request_id": directory.name,
                            "frame": frame,
                            "layer_batches": comparisons,
                            "isolated_exact": exact_isolated_batches,
                        }
                    ),
                    flush=True,
                )
            records.append(
                {
                    "request_id": directory.name,
                    "frames": len(files),
                    "layer_batches": comparisons,
                    "exact_whole_native_batches": comparisons,
                    "exact_isolated_request_batches": exact_isolated_batches,
                }
            )
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(
                    {
                        "records": records,
                        "scope": "Same native weights, held routing ids and FP32 scales, same observed algorithm IDs. Only grouping the selected request alone versus the original whole native batch changes. Complete lifetimes and all captured prefill rows, no free-running quality or invalid-kernel claim.",
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
else:
    pass
