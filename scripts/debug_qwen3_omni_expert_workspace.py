"""Inspect retained native expert activation and down-projection workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType, cutlass_fused_moe_workspace_size
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
from scripts.qwen3_omni_expert_workspace_runtime import (
    read_workspace_stages,
    workspace_prefix,
    workspace_view,
)
from scripts.qwen3_omni_talker_expert_runtime import (
    NativeRuntime,
    ObservedExpertProfileMap,
    load_native_expert_weights,
)
from sglang_omni.models.qwen3_omni.contribution_rounding import (
    FINALIZER_PATH,
    install_contribution_provider,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--trace-directory", action="append", type=Path, required=True)
    parser.add_argument("--profile-map", type=Path, required=True)
    parser.add_argument("--contribution-source-directory", type=Path, required=True)
    parser.add_argument("--layout-source-sha256", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    source_hash = hashlib.sha256(
        (arguments.contribution_source_directory / FINALIZER_PATH).read_bytes()
    ).hexdigest()
    assert source_hash == arguments.layout_source_sha256
    arguments.output_directory.mkdir(exist_ok=False)
    install_contribution_provider(arguments.contribution_source_directory)
    torch.cuda.set_device(arguments.gpu)
    torch.set_float32_matmul_precision("highest")
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
    profile_map = ObservedExpertProfileMap.model_validate_json(
        arguments.profile_map.read_bytes()
    )
    records = []
    with (arguments.output_directory / "layers.jsonl").open("x") as destination:
        for directory in arguments.trace_directory:
            runtime = NativeRuntime.model_validate_json(
                (directory.parent / "moe-runtime.json").read_bytes()
            )
            assert not runtime.use_fused_finalize
            profiles = profile_map.profiles[str(directory.resolve())]
            assert profiles.source_capture == directory
            weights = load_native_expert_weights(reference, runtime)
            filenames = sorted(directory.glob("*-operators.pt"))
            assert len(filenames) == profiles.verified_frames
            comparisons = 0
            maxima: dict[str, float] = {}
            exact_counts: dict[str, int] = {}
            for frame, filename in enumerate(filenames):
                assert filename.name == f"{frame:04d}-operators.pt"
                snapshots = torch.load(filename, weights_only=True)
                assert len(snapshots) == len(weights)
                for layer_index, (snapshot, (packed, down)) in enumerate(
                    zip(snapshots, weights, strict=True)
                ):
                    hidden = snapshot["batch_mlp_normalized"].to(reference.device)
                    indices = snapshot["batch_expert_indices"].to(reference.device)
                    scales = snapshot["batch_expert_weights"].to(reference.device)
                    expected = snapshot["batch_routed_output"].to(reference.device)
                    tokens, hidden_channels = hidden.shape
                    experts, _, intermediate_channels = down.shape
                    ranks = indices.shape[1]
                    assert hidden.dtype == packed.dtype == down.dtype == torch.bfloat16
                    assert (
                        indices.dtype == torch.int32 and scales.dtype == torch.float32
                    )
                    buffers = workspace_prefix(
                        tokens, hidden_channels, intermediate_channels, experts, ranks
                    )
                    workspace_bytes = cutlass_fused_moe_workspace_size(
                        tokens,
                        hidden_channels,
                        intermediate_channels,
                        experts,
                        ranks,
                        x_dtype=hidden.dtype,
                        weight_dtype=packed.dtype,
                        use_fused_finalize=False,
                        device=hidden.device,
                    )
                    workspace = torch.empty(
                        workspace_bytes, dtype=torch.uint8, device=hidden.device
                    )
                    replay_inputs = hidden.clone()
                    profiler = (
                        torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        )
                        if frame < 2 and layer_index == 0
                        else nullcontext()
                    )
                    with profiler as observed_profile:
                        output = cutlass_fused_moe(
                            input=replay_inputs,
                            token_selected_experts=indices,
                            token_final_scales=scales,
                            fc1_expert_weights=packed,
                            fc2_expert_weights=down,
                            output_dtype=torch.bfloat16,
                            quant_scales=None,
                            activation_type=ActivationType.Swiglu,
                            use_fused_finalize=False,
                            tune_max_num_tokens=runtime.tune_max_num_tokens,
                            profile_ids=list(
                                profiles.prefill[layer_index]
                                if frame == 0
                                else profiles.decode[layer_index]
                            ),
                            workspace_buffer=workspace,
                        )[0]
                    if observed_profile is not None:
                        (
                            arguments.output_directory
                            / f"{directory.name}-{frame}-kernels.json"
                        ).write_text(
                            json.dumps(
                                {
                                    "kernels": [
                                        event.name
                                        for event in observed_profile.events()
                                        if event.device_type
                                        == torch.autograd.DeviceType.CUDA
                                    ]
                                },
                                indent=2,
                            )
                        )
                    else:
                        pass
                    assert torch.equal(
                        replay_inputs.view(torch.uint8), hidden.view(torch.uint8)
                    )
                    assert torch.equal(
                        output.view(torch.uint8), expected.view(torch.uint8)
                    )
                    expanded_tokens = tokens * ranks
                    source_rows = workspace_view(
                        workspace,
                        buffers["permuted_row_to_unpermuted_row"],
                        torch.int32,
                        (expanded_tokens,),
                    ).long()
                    assert torch.equal(
                        source_rows.sort().values,
                        torch.arange(expanded_tokens, device=hidden.device),
                    )
                    offsets = workspace_view(
                        workspace,
                        buffers["expert_first_token_offset"],
                        torch.int64,
                        (experts + 1,),
                    )
                    assert int(offsets[0]) == 0 and int(offsets[-1]) == expanded_tokens
                    assert (offsets[1:] >= offsets[:-1]).all()
                    token_rows = source_rows % tokens
                    expert_ranks = source_rows // tokens
                    expected_experts = torch.repeat_interleave(
                        torch.arange(experts, device=hidden.device), offsets.diff()
                    )
                    assert torch.equal(
                        indices[token_rows, expert_ranks], expected_experts
                    )
                    stages = read_workspace_stages(
                        workspace,
                        buffers,
                        indices,
                        scales,
                        token_rows,
                        expert_ranks,
                        intermediate_channels,
                        output,
                    )
                    inverse_rows = torch.empty_like(source_rows)
                    inverse_rows[source_rows] = torch.arange(
                        expanded_tokens, device=hidden.device
                    )
                    row = int(snapshot["batch_selected_row"])
                    selected = inverse_rows[
                        row + torch.arange(ranks, device=hidden.device) * tokens
                    ]
                    native_activation = stages.activated[selected]
                    native_down = stages.projected[selected]
                    ideal_activated = []
                    unrounded_activated = []
                    rounded_activated = []
                    ideal_down = []
                    torch_down = []
                    for rank, expert_index in enumerate(indices[row].tolist()):
                        expert = reference.model.layers[layer_index].mlp.experts
                        gate_projection, up_projection = torch.nn.functional.linear(
                            hidden[row : row + 1].double(),
                            expert.gate_up_proj[expert_index].double(),
                        ).chunk(2, dim=-1)
                        gate = gate_projection.bfloat16()
                        up = up_projection.bfloat16()
                        unrounded_activated.append(
                            (
                                torch.nn.functional.silu(gate_projection)
                                * up_projection
                            ).bfloat16()
                        )
                        ideal_activated.append(
                            (
                                torch.nn.functional.silu(gate.double()) * up.double()
                            ).bfloat16()
                        )
                        rounded_activated.append(
                            (
                                torch.nn.functional.silu(gate.double()).bfloat16() * up
                            ).bfloat16()
                        )
                        ideal_down.append(
                            torch.nn.functional.linear(
                                native_activation[rank : rank + 1].double(),
                                expert.down_proj[expert_index].double(),
                            ).bfloat16()
                        )
                        torch_down.append(
                            torch.nn.functional.linear(
                                native_activation[rank : rank + 1],
                                expert.down_proj[expert_index],
                            )
                        )
                    differences = {
                        "activation_vs_fp64_without_gemm_store": compare_tensors(
                            native_activation, torch.cat(unrounded_activated)
                        ),
                        "activation_vs_fp64_gemm_bf16_store_fp64_silu": compare_tensors(
                            native_activation, torch.cat(ideal_activated)
                        ),
                        "activation_vs_fp64_gemm_bf16_silu_store": compare_tensors(
                            native_activation, torch.cat(rounded_activated)
                        ),
                        "down_vs_fp64_on_native_activation": compare_tensors(
                            native_down, torch.cat(ideal_down)
                        ),
                        "down_vs_torch_on_native_activation": compare_tensors(
                            native_down, torch.cat(torch_down)
                        ),
                    }
                    for name, difference in differences.items():
                        maxima[name] = max(
                            maxima.get(name, 0), difference.maximum_absolute_error
                        )
                        exact_counts[name] = (
                            exact_counts.get(name, 0) + difference.exact
                        )
                    destination.write(
                        json.dumps(
                            {
                                "request_id": directory.name,
                                "frame": frame,
                                "layer": layer_index,
                                "batch_tokens": tokens,
                                "whole_output_bytes_exact": True,
                                "workspace_down_reduction_bytes_exact": True,
                                "has_glu_intermediate": stages.has_glu_intermediate,
                                "differences": {
                                    name: asdict(value)
                                    for name, value in differences.items()
                                },
                            }
                        )
                        + "\n"
                    )
                    comparisons += 1
                destination.flush()
                print(
                    json.dumps(
                        {
                            "request_id": directory.name,
                            "frame": frame,
                            "layers": comparisons,
                        }
                    ),
                    flush=True,
                )
            records.append(
                {
                    "request_id": directory.name,
                    "frames": len(filenames),
                    "layer_batches": comparisons,
                    "exact_counts": exact_counts,
                    "maximum_absolute_errors": maxima,
                }
            )
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(
                    {
                        "records": records,
                        "layout_source_sha256": source_hash,
                        "scope": "Pinned unquantized single-GPU BF16 non-fused-finalizer workspace. Actual native activation/down buffers, complete map validation and full output-byte reconstruction precede comparisons. FP64 is a higher-precision numerical reference, not a speech-quality oracle. GEMM1 operands are overwritten by GEMM2, so an activation difference alone does not separate first-GEMM and SiLU implementation.",
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
else:
    pass
