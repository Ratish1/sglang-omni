"""Replay original native MoE batches with observed tactics and verified weights."""

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
from scripts.qwen3_omni_talker_expert_rounding import expert_rounding_stages
from scripts.qwen3_omni_talker_expert_runtime import (
    NativeRuntime,
    TacticChoice,
    load_native_expert_weights,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--previous-directory", type=Path)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--rounding-stages", action="store_true")
    parser.add_argument("--verify-profile-alias", type=int, nargs=2)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    runtime = NativeRuntime.model_validate_json(
        (arguments.capture_directory / "moe-runtime.json").read_bytes()
    )
    choices = [
        TacticChoice.model_validate(choice)
        for choice in json.loads(
            (arguments.capture_directory / "moe-choices-final.json").read_text()
        )
    ]
    assert choices[: len(runtime.tactic_choices)] == runtime.tactic_choices
    directory = arguments.capture_directory / arguments.request_id
    finished = torch.load(directory / "finished.pt", weights_only=True)
    initial = torch.load(directory / "0000-before_prefill.pt", weights_only=True)
    output_files = sorted(directory.glob("*-output.pt"))
    assert [filename.name for filename in output_files] == [
        f"{frame:04d}-output.pt" for frame in range(len(finished["output_ids"]))
    ]
    if arguments.previous_directory is not None:
        previous = arguments.previous_directory / arguments.request_id
        earlier = torch.load(previous / "finished.pt", weights_only=True)
        for name in ("output_ids", "decode_input_history", "enqueued_text_rows"):
            assert torch.equal(finished[name], earlier[name]), name
        prior_initial = torch.load(
            previous / "0000-before_prefill.pt", weights_only=True
        )
        assert initial.keys() == prior_initial.keys()
        assert all(
            torch.equal(tensor, prior_initial[name]) for name, tensor in initial.items()
        )
        for filename in output_files:
            observed = torch.load(filename, weights_only=True)
            old = torch.load(previous / filename.name, weights_only=True)
            assert bool(observed["is_extend"]) == (filename.name == "0000-output.pt")
            assert bool(observed["past_exact"]) and bool(old["past_exact"])
            for name in ("codec_ids", "hidden_states", "logits"):
                assert torch.equal(observed[name], old[name]), (filename.name, name)
            for name in ("keys", "values"):
                assert all(
                    torch.equal(current, prior)
                    for current, prior in zip(observed[name], old[name], strict=True)
                ), (filename.name, name)
    else:
        for filename in output_files:
            observed = torch.load(filename, weights_only=True)
            assert bool(observed["is_extend"]) == (filename.name == "0000-output.pt")
            assert bool(observed["past_exact"])
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
    native_weights = load_native_expert_weights(reference, runtime)
    filenames = sorted(directory.glob("*-operators.pt"))
    assert [filename.name for filename in filenames] == [
        f"{frame:04d}-operators.pt" for frame in range(len(finished["output_ids"]))
    ]
    exact_replays = 0
    comparisons = 0
    maximum_replay_error = 0.0
    exact_profile_alias_replays = 0
    exact_router_batches = 0
    exact_shared_gate_projections = 0
    maximum_shared_gate_error = 0.0
    matching_routing_batches = 0
    maximum_checkpoint_expert_error = 0.0
    rounding_stage_exact_batches: dict[str, int] = {}
    rounding_stage_maximum_relative_errors: dict[str, float] = {}
    with (arguments.output_directory / "layers.jsonl").open("w") as output:
        for filename in filenames:
            frame = int(filename.name.split("-")[0])
            phase = "before_prefill" if frame == 0 else "before_decode"
            before = torch.load(
                directory / f"{frame:04d}-{phase}.pt", weights_only=True
            )
            position = (
                before["mrope_positions"][:, -1]
                if frame == 0 and before["mrope_positions"].numel()
                else before["positions"][-1]
            )
            snapshots = torch.load(filename, weights_only=True)
            assert len(snapshots) == len(native_weights)
            for layer_index, (snapshot, (packed, down)) in enumerate(
                zip(snapshots, native_weights, strict=True)
            ):
                row = int(snapshot["batch_selected_row"])
                assert (snapshot["positions"] == position).all()
                capturing = bool(snapshot["batch_is_graph"])
                tensors = {}
                for name in (
                    "mlp_normalized",
                    "router_logits",
                    "expert_indices",
                    "expert_weights",
                    "routed_output",
                ):
                    batch = snapshot[f"batch_{name}"]
                    assert batch.ndim == 2 and torch.isfinite(batch).all()
                    assert torch.equal(batch[row], snapshot[name]), name
                    layout = snapshot[f"layout_{name}"].tolist()
                    assert layout[:2] == list(batch.shape)
                    assert layout[2:4] == list(batch.stride())
                    assert layout[4] == 0
                    tensors[name] = batch.to(reference.device)
                    assert tensors[name].data_ptr() % 16 == 0
                    assert tensors[name].stride() == batch.stride()
                hidden_states = tensors["mlp_normalized"]
                indices = tensors["expert_indices"]
                weights = tensors["expert_weights"]
                assert hidden_states.dtype == torch.bfloat16
                assert indices.dtype == torch.int32 and weights.dtype == torch.float32
                assert indices.shape == weights.shape
                assert indices.shape == (
                    len(hidden_states),
                    configuration.text_config.num_experts_per_tok,
                )
                assert indices.min() >= 0 and indices.max() < len(packed)
                assert weights.min() >= 0
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
                        and choice.capturing == capturing
                        and not choice.tuning
                    }
                    assert len(tactics) == 1, (shapes, capturing, operation, tactics)
                    profiles.append(tactics.pop())
                replayed = []
                profile_choices = [profiles, [-1, -1]]
                if arguments.verify_profile_alias is not None:
                    assert profiles == [-1, -1]
                    profile_choices.append(arguments.verify_profile_alias)
                else:
                    pass
                for selected_profiles in profile_choices:
                    replayed.append(
                        cutlass_fused_moe(
                            input=hidden_states,
                            token_selected_experts=indices,
                            token_final_scales=weights,
                            fc1_expert_weights=packed,
                            fc2_expert_weights=down,
                            output_dtype=torch.bfloat16,
                            quant_scales=None,
                            activation_type=ActivationType.Swiglu,
                            tune_max_num_tokens=runtime.tune_max_num_tokens,
                            use_fused_finalize=runtime.use_fused_finalize,
                            profile_ids=selected_profiles,
                        )[0]
                    )
                assert torch.equal(
                    hidden_states.cpu(), snapshot["batch_mlp_normalized"]
                )
                difference = compare_tensors(tensors["routed_output"], replayed[0])
                if arguments.verify_profile_alias is not None:
                    profile_alias_difference = compare_tensors(
                        tensors["routed_output"], replayed[2]
                    )
                    exact_profile_alias_replays += profile_alias_difference.exact
                else:
                    profile_alias_difference = None
                reference_layer = reference.model.layers[layer_index]
                shared_gate_difference = compare_tensors(
                    snapshot["shared_gate"].to(reference.device),
                    reference_layer.mlp.shared_expert_gate(hidden_states)[row],
                )
                reference_router_logits, reference_router_weights, reference_indices = (
                    reference_layer.mlp.gate(hidden_states)
                )
                router_difference = compare_tensors(
                    tensors["router_logits"], reference_router_logits
                )
                matching_experts = torch.equal(
                    indices.sort(dim=-1).values,
                    reference_indices.to(indices.dtype).sort(dim=-1).values,
                )
                reference_probabilities = reference_router_logits.float().softmax(
                    dim=-1
                )
                reference_native_order_weights = reference_probabilities.gather(
                    1, indices.long()
                )
                if configuration.text_config.norm_topk_prob:
                    reference_native_order_weights /= (
                        reference_native_order_weights.sum(dim=-1, keepdim=True)
                    )
                else:
                    pass
                held_routing_expert_output = reference_layer.mlp.experts(
                    hidden_states, indices.long(), weights
                )
                checkpoint_expert_difference = compare_tensors(
                    tensors["routed_output"], held_routing_expert_output
                )
                if arguments.rounding_stages:
                    stage_outputs = expert_rounding_stages(
                        reference_layer.mlp.experts,
                        hidden_states,
                        indices.long(),
                        weights,
                    )
                    assert torch.equal(
                        stage_outputs["checkpoint_expert_order_bfloat16_sum"],
                        held_routing_expert_output,
                    )
                    rounding_differences = {
                        name: compare_tensors(tensors["routed_output"], value)
                        for name, value in stage_outputs.items()
                    }
                    for name, stage_difference in rounding_differences.items():
                        rounding_stage_exact_batches[name] = (
                            rounding_stage_exact_batches.get(name, 0)
                            + stage_difference.exact
                        )
                        rounding_stage_maximum_relative_errors[name] = max(
                            rounding_stage_maximum_relative_errors.get(name, 0.0),
                            stage_difference.relative_l2_error or 0.0,
                        )
                else:
                    rounding_differences = {}
                record = {
                    "frame": int(filename.name.split("-")[0]),
                    "layer": layer_index,
                    "selected_row": row,
                    "batch_tokens": len(hidden_states),
                    "captured_graph": capturing,
                    "profiles": profiles,
                    "native_replay": asdict(difference),
                    "explicit_profile_alias_replay": (
                        asdict(profile_alias_difference)
                        if profile_alias_difference is not None
                        else None
                    ),
                    "selected_row_replay": asdict(
                        compare_tensors(tensors["routed_output"][row], replayed[0][row])
                    ),
                    "default_vs_observed": asdict(
                        compare_tensors(tensors["routed_output"], replayed[1])
                    ),
                    "checkpoint_router": asdict(router_difference),
                    "shared_gate_captured_batch_geometry": asdict(
                        shared_gate_difference
                    ),
                    "checkpoint_expert_membership_exact": matching_experts,
                    "checkpoint_float32_weights_native_order": asdict(
                        compare_tensors(weights, reference_native_order_weights)
                    ),
                    "checkpoint_bfloat16_weights_native_order": asdict(
                        compare_tensors(
                            weights,
                            reference_native_order_weights.to(
                                reference_router_weights.dtype
                            ),
                        )
                    ),
                    "checkpoint_experts_with_native_routing": asdict(
                        checkpoint_expert_difference
                    ),
                    "selected_checkpoint_experts_with_native_routing": asdict(
                        compare_tensors(
                            tensors["routed_output"][row],
                            held_routing_expert_output[row],
                        )
                    ),
                    "rounding_stages": {
                        name: asdict(stage_difference)
                        for name, stage_difference in rounding_differences.items()
                    },
                }
                output.write(json.dumps(record) + "\n")
                output.flush()
                exact_replays += difference.exact
                comparisons += 1
                maximum_replay_error = max(
                    maximum_replay_error, difference.maximum_absolute_error
                )
                exact_router_batches += router_difference.exact
                exact_shared_gate_projections += shared_gate_difference.exact
                maximum_shared_gate_error = max(
                    maximum_shared_gate_error,
                    shared_gate_difference.maximum_absolute_error,
                )
                matching_routing_batches += matching_experts
                maximum_checkpoint_expert_error = max(
                    maximum_checkpoint_expert_error,
                    checkpoint_expert_difference.maximum_absolute_error,
                )
            print(
                json.dumps(
                    {
                        "frame": filename.name,
                        "comparisons": comparisons,
                        "exact": exact_replays,
                    }
                ),
                flush=True,
            )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(
            {
                "request_id": arguments.request_id,
                "frames": len(filenames),
                "layer_batches": comparisons,
                "exact_native_replays": exact_replays,
                "explicit_profile_alias": arguments.verify_profile_alias,
                "exact_profile_alias_replays": exact_profile_alias_replays,
                "maximum_native_replay_error": maximum_replay_error,
                "paired_trajectory_checked": arguments.previous_directory is not None,
                "exact_checkpoint_router_batches": exact_router_batches,
                "exact_shared_gate_projections": exact_shared_gate_projections,
                "maximum_shared_gate_error": maximum_shared_gate_error,
                "matching_checkpoint_expert_membership_batches": matching_routing_batches,
                "maximum_checkpoint_expert_error_with_native_routing": maximum_checkpoint_expert_error,
                "rounding_stage_exact_batches": rounding_stage_exact_batches,
                "rounding_stage_maximum_relative_errors": rounding_stage_maximum_relative_errors,
                "rounding_scope": "Per-expert cuBLAS GEMMs preserve checkpoint row order and independently separate activation, weighted-store and reduction rounding. They are not the native grouped-GEMM geometry or a bit-exact CUTLASS SiLU implementation. Both FMA and separate multiply/add are reported. The final checkpoint stage must match the original eager expert calculation exactly; other differences are measurements, not proof of an invalid native operation.",
                "scope": "Complete original batches, native packed-weight hashes and observed per-geometry tactic choices. Fallback -1 identifies the same internal heuristic request, not a resolved tactic ID. Default replay is a numerical control, not a quality claim or backend recommendation.",
            },
            indent=2,
        )
    )
    if arguments.verify_profile_alias is not None:
        assert exact_profile_alias_replays == exact_replays == comparisons
    else:
        pass


if __name__ == "__main__":
    main()
else:
    pass
