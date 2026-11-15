"""Compare isolated activation stores on complete captured Talker prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import ExitStack, nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType, cutlass_fused_moe_workspace_size
from safetensors import safe_open

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.qwen3_omni_checkpoint_weights import CheckpointIndex
from scripts.qwen3_omni_expert_workspace_runtime import (
    read_workspace_stages,
    workspace_prefix,
    workspace_view,
)
from scripts.qwen3_omni_talker_expert_runtime import NativeRuntime
from sglang_omni.models.qwen3_omni.contribution_rounding import (
    install_contribution_provider,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-directory", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--routing-directory", type=Path, required=True)
    parser.add_argument("--profile", nargs=2, type=int, action="append", required=True)
    parser.add_argument("--unchanged-profile", nargs=2, type=int, required=True)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--baseline-directory", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    assert (arguments.source_directory is None) == (
        arguments.baseline_directory is None
    )
    assert arguments.unchanged_profile in arguments.profile
    arguments.output_directory.mkdir(exist_ok=False)
    if arguments.source_directory is not None:
        install_contribution_provider(arguments.source_directory)
    else:
        pass
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    torch.set_float32_matmul_precision("highest")
    runtime = NativeRuntime.model_validate_json(arguments.runtime.read_bytes())
    assert not runtime.use_fused_finalize
    index = CheckpointIndex.model_validate_json(
        (arguments.checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    routing_files = sorted(arguments.routing_directory.glob("*/*/routing.pt"))
    assert routing_files
    input_identity = {
        "runtime_sha256": hashlib.sha256(arguments.runtime.read_bytes()).hexdigest(),
        "routing_sha256": {
            str(filename): hashlib.sha256(filename.read_bytes()).hexdigest()
            for filename in routing_files
        },
        "profiles": arguments.profile,
    }
    if arguments.baseline_directory is not None:
        baseline_summary = json.loads(
            (arguments.baseline_directory / "summary.json").read_text()
        )
        assert baseline_summary["input_identity"] == input_identity
    else:
        pass
    captures = {
        f"{filename.parent.parent.name}-{filename.parent.name}": torch.load(
            filename, weights_only=True, map_location="cpu"
        )
        for filename in routing_files
    }
    assert all(len(layers) == len(runtime.layers) for layers in captures.values())
    shard_names = {
        shard
        for name, shard in index.weight_map.items()
        if name.startswith("talker.model.layers.") and ".mlp.experts." in name
    }
    comparisons = 0
    affected_activations = 0
    with ExitStack() as resources:
        checkpoints = {
            shard: resources.enter_context(
                safe_open(arguments.checkpoint_directory / shard, framework="pt")
            )
            for shard in sorted(shard_names)
        }
        destination = resources.enter_context(
            (arguments.output_directory / "layers.jsonl").open("x")
        )
        for layer_index, layer_runtime in enumerate(runtime.layers):
            assert layer_runtime.tp_size == layer_runtime.ep_size == 1
            assert layer_runtime.tp_rank == layer_runtime.ep_rank == 0
            experts = layer_runtime.weights["w13_weight"].shape[0]
            projections: dict[str, list[torch.Tensor]] = {
                "up_proj": [],
                "gate_proj": [],
                "down_proj": [],
            }
            for expert_index in range(experts):
                for projection, tensors in projections.items():
                    name = f"talker.model.layers.{layer_index}.mlp.experts.{expert_index}.{projection}.weight"
                    tensors.append(checkpoints[index.weight_map[name]].get_tensor(name))
            packed = torch.cat(
                (
                    torch.stack(projections["up_proj"]),
                    torch.stack(projections["gate_proj"]),
                ),
                dim=1,
            ).contiguous()
            down = torch.stack(projections["down_proj"]).contiguous()
            for name, tensor in (("w13_weight", packed), ("w2_weight", down)):
                expected = layer_runtime.weights[name]
                assert list(tensor.shape) == expected.shape
                assert str(tensor.dtype) == expected.dtype == "torch.bfloat16"
                assert (
                    hashlib.sha256(
                        tensor.view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    == expected.sha256
                )
            packed, down = packed.to(device), down.to(device)
            intermediate_channels = down.shape[-1]
            for capture_name, layers in captures.items():
                captured = layers[layer_index]
                hidden_states = captured["normalized_inputs"].to(device)
                indices = (
                    captured["indices"]
                    .to(device=device, dtype=torch.int32)
                    .contiguous()
                )
                scales = (
                    captured["weights"]
                    .to(device=device, dtype=torch.float32)
                    .contiguous()
                )
                tokens, hidden_channels = hidden_states.shape
                ranks = indices.shape[1]
                expanded_tokens = tokens * ranks
                assert (
                    hidden_states.dtype == packed.dtype == down.dtype == torch.bfloat16
                )
                assert indices.shape == scales.shape
                reference_unrounded = torch.empty(
                    (tokens, ranks, intermediate_channels),
                    dtype=torch.bfloat16,
                    device=device,
                )
                reference_stores = torch.empty_like(reference_unrounded)
                for expert_index in indices.unique().tolist():
                    token_rows, expert_ranks = (indices == expert_index).nonzero(
                        as_tuple=True
                    )
                    up_projection, gate_projection = torch.nn.functional.linear(
                        hidden_states[token_rows].double(),
                        packed[expert_index].double(),
                    ).chunk(2, dim=-1)
                    reference_unrounded[token_rows, expert_ranks] = (
                        torch.nn.functional.silu(gate_projection) * up_projection
                    ).bfloat16()
                    reference_stores[token_rows, expert_ranks] = (
                        torch.nn.functional.silu(gate_projection.bfloat16()).bfloat16()
                        * up_projection.bfloat16()
                    )
                buffers = workspace_prefix(
                    tokens, hidden_channels, intermediate_channels, experts, ranks
                )
                workspace = torch.empty(
                    cutlass_fused_moe_workspace_size(
                        tokens,
                        hidden_channels,
                        intermediate_channels,
                        experts,
                        ranks,
                        x_dtype=hidden_states.dtype,
                        weight_dtype=packed.dtype,
                        use_fused_finalize=False,
                        device=device,
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
                for profiles in arguments.profile:
                    repeated: dict[str, torch.Tensor] = {}
                    for repetition in range(2):
                        replay_inputs = hidden_states.clone()
                        profiler = (
                            torch.profiler.profile(
                                activities=[
                                    torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA,
                                ]
                            )
                            if layer_index == 0
                            and capture_name == next(iter(captures))
                            and repetition == 0
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
                                profile_ids=profiles,
                                workspace_buffer=workspace,
                            )[0]
                        if observed_profile is not None:
                            observed_profile.export_chrome_trace(
                                str(
                                    arguments.output_directory
                                    / f"kernels-{profiles[0]}-{profiles[1]}.json"
                                )
                            )
                        else:
                            pass
                        assert torch.equal(replay_inputs, hidden_states)
                        source_rows = workspace_view(
                            workspace,
                            buffers["permuted_row_to_unpermuted_row"],
                            torch.int32,
                            (expanded_tokens,),
                        ).long()
                        assert torch.equal(
                            source_rows.sort().values,
                            torch.arange(expanded_tokens, device=device),
                        )
                        offsets = workspace_view(
                            workspace,
                            buffers["expert_first_token_offset"],
                            torch.int64,
                            (experts + 1,),
                        )
                        assert (
                            int(offsets[0]) == 0 and int(offsets[-1]) == expanded_tokens
                        )
                        assert (offsets.diff() >= 0).all()
                        token_rows, expert_ranks = (
                            source_rows % tokens,
                            source_rows // tokens,
                        )
                        assert torch.equal(
                            indices[token_rows, expert_ranks],
                            torch.repeat_interleave(
                                torch.arange(experts, device=device), offsets.diff()
                            ),
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
                            "native",
                        )
                        current = {
                            "activated": stages.activated.clone(),
                            "projected": stages.projected.clone(),
                            "output": output.clone(),
                            "source_rows": source_rows.clone(),
                        }
                        if repetition == 0:
                            repeated = current
                        else:
                            assert all(
                                torch.equal(value, current[name])
                                for name, value in repeated.items()
                            )
                    filename = f"{capture_name}-layer{layer_index}-{profiles[0]}-{profiles[1]}.pt"
                    source_rows = repeated["source_rows"]
                    token_rows, expert_ranks = (
                        source_rows % tokens,
                        source_rows // tokens,
                    )
                    reference_differences = {
                        "fp64_gemm_without_stores": asdict(
                            compare_tensors(
                                repeated["activated"],
                                reference_unrounded[token_rows, expert_ranks],
                            )
                        ),
                        "fp64_gemm_with_bf16_stores": asdict(
                            compare_tensors(
                                repeated["activated"],
                                reference_stores[token_rows, expert_ranks],
                            )
                        ),
                    }
                    differences = {}
                    if arguments.baseline_directory is None:
                        torch.save(
                            {name: tensor.cpu() for name, tensor in repeated.items()},
                            arguments.output_directory / filename,
                        )
                    else:
                        baseline = torch.load(
                            arguments.baseline_directory / filename,
                            weights_only=True,
                            map_location=device,
                        )
                        assert torch.equal(
                            baseline["source_rows"], repeated["source_rows"]
                        )
                        differences = {
                            name: asdict(
                                compare_tensors(repeated[name], baseline[name])
                            )
                            for name in ("activated", "projected", "output")
                        }
                        if profiles == arguments.unchanged_profile:
                            assert all(
                                difference["exact"]
                                for difference in differences.values()
                            )
                        else:
                            affected_activations += not differences["activated"][
                                "exact"
                            ]
                    destination.write(
                        json.dumps(
                            {
                                "capture": capture_name,
                                "layer": layer_index,
                                "tokens": tokens,
                                "profiles": profiles,
                                "unchanged_repetition_exact": True,
                                "complete_output_reconstructed": True,
                                "differences": differences,
                                "activation_references": reference_differences,
                            }
                        )
                        + "\n"
                    )
                    comparisons += 1
                destination.flush()
            print(
                json.dumps({"layer": layer_index, "comparisons": comparisons}),
                flush=True,
            )
        if arguments.baseline_directory is not None:
            assert affected_activations > 0
        else:
            pass
        libraries = sorted(
            {
                line.split()[-1]
                for line in Path("/proc/self/maps").read_text().splitlines()
                if "fused_moe" in line and line.split()[-1].endswith(".so")
            }
        )
        assert libraries
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(
                {
                    "comparisons": comparisons,
                    "changed_activation_cells": affected_activations,
                    "libraries": {
                        filename: hashlib.sha256(
                            Path(filename).read_bytes()
                        ).hexdigest()
                        for filename in libraries
                    },
                    "input_identity": input_identity,
                    "scope": "Complete saved prompt operands, all layers, unchanged selected experts and weights. Fixed profiles are diagnostic controls, not a native batch replay or a speech/performance verdict. FP64 references are not bit-exact emulators or speech-quality oracles.",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
else:
    pass
