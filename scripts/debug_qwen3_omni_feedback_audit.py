# SPDX-License-Identifier: Apache-2.0
"""Measure captured Talker feedback reductions and expert arithmetic."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


@dataclass(kw_only=True)
class TensorDifference:
    exact: bool
    equal_fraction: float
    maximum_absolute_error: float
    relative_l2_error: float | None


def compare_tensors(actual: torch.Tensor, expected: torch.Tensor) -> TensorDifference:
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    difference = actual.float() - expected.float()
    expected_norm = expected.float().norm().item()
    return TensorDifference(
        exact=torch.equal(actual, expected),
        equal_fraction=(actual == expected).float().mean().item(),
        maximum_absolute_error=difference.abs().max().item(),
        relative_l2_error=(
            difference.norm().item() / expected_norm if expected_norm else None
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--trace-directory", action="append", type=Path, required=True)
    parser.add_argument("--compare-trace-directories", nargs=2, type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--feedback-only", action="store_true")
    parser.add_argument("--profile-tactics", nargs=2, type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    torch.set_float32_matmul_precision("highest")
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    reports = []
    paired_frames = []
    if arguments.compare_trace_directories is not None:
        native_directory, candidate_directory = arguments.compare_trace_directories
        for native_file in sorted(native_directory.glob("*.pt")):
            candidate_file = candidate_directory / native_file.name
            if not candidate_file.is_file():
                continue
            else:
                pass
            native = torch.load(native_file, weights_only=True)
            candidate = torch.load(candidate_file, weights_only=True)
            paired_frames.append(
                {
                    "trace_file": native_file.name,
                    "fields": {
                        name: asdict(compare_tensors(native[name], candidate[name]))
                        for name in sorted(native.keys() & candidate.keys())
                        if native[name].shape == candidate[name].shape
                    },
                }
            )
        (arguments.output_directory / "paired-frames.json").write_text(
            json.dumps(paired_frames, indent=2)
        )
    else:
        pass
    with torch.inference_mode():
        for trace_directory in arguments.trace_directory:
            frames = []
            expert_layers = []
            trace_pattern = (
                "*-post_decode.pt" if arguments.decode_only else "*-post_*.pt"
            )
            trace_files = sorted(trace_directory.glob(trace_pattern))
            assert trace_files, trace_directory
            for trace_file in trace_files:
                captured = torch.load(trace_file, weights_only=True)
                codes = captured["codec_ids"].to(device)
                embedding_rows = [reference.model.codec_embedding(codes[:, :1])]
                for group, embedding in enumerate(
                    reference.code_predictor.model.codec_embedding
                ):
                    embedding_rows.append(embedding(codes[:, group + 1 : group + 2]))
                sequential_sum = torch.zeros_like(embedding_rows[0][:, 0])
                for embedding in embedding_rows:
                    sequential_sum.add_(embedding[:, 0])
                reference_sum = torch.cat(embedding_rows, dim=1).sum(dim=1)
                feedback = captured["codec_feedback"].to(device)
                frame_report = {
                    "trace_file": trace_file.name,
                    "codes": codes.cpu().tolist(),
                    "sequential_bf16": asdict(
                        compare_tensors(feedback, sequential_sum)
                    ),
                    "single_reduction": asdict(
                        compare_tensors(feedback, reference_sum)
                    ),
                }
                frames.append(frame_report)
                if arguments.feedback_only or (
                    not arguments.decode_only and "post_prefill" not in trace_file.name
                ):
                    continue
                else:
                    pass
                token_count = 1 if arguments.decode_only else 128
                for layer_index, layer in enumerate(reference.model.layers):
                    prefix = f"moe_{layer_index}_"
                    hidden_states = captured[f"{prefix}input_{token_count}"].to(device)
                    selected_experts = (
                        captured[f"{prefix}experts_{token_count}"].to(device).long()
                    )
                    routing_weights = (
                        captured[f"{prefix}weights_{token_count}"].to(device).float()
                    )
                    native_output = captured[f"{prefix}routed_output_{token_count}"].to(
                        device
                    )
                    router_logits = captured[f"{prefix}router_logits_{token_count}"].to(
                        device
                    )
                    selected_logits = router_logits.gather(1, selected_experts)
                    topk_threshold = router_logits.topk(
                        selected_experts.shape[1], dim=-1
                    ).values[:, -1:]
                    selected_weights = torch.softmax(
                        router_logits.float(), dim=-1
                    ).gather(1, selected_experts)
                    if config.text_config.norm_topk_prob:
                        selected_weights /= selected_weights.sum(dim=-1, keepdim=True)
                    else:
                        pass
                    if arguments.decode_only:
                        normalized_input = captured[
                            f"boundary_{layer_index}_mlp_normalized_1"
                        ].to(device)
                        assert torch.equal(hidden_states, normalized_input)
                        shared_output = captured[f"{prefix}shared_output_1"].to(device)
                        whole_output = captured[
                            f"boundary_{layer_index}_mlp_output_1"
                        ].to(device)
                        assert torch.equal(native_output + shared_output, whole_output)
                    else:
                        pass
                    reference_output = layer.mlp.experts(
                        hidden_states, selected_experts, routing_weights
                    )
                    float_output = torch.zeros_like(hidden_states, dtype=torch.float32)
                    for expert_index in selected_experts.unique().tolist():
                        token_rows, selected_columns = (
                            selected_experts == expert_index
                        ).nonzero(as_tuple=True)
                        projections = torch.nn.functional.linear(
                            hidden_states[token_rows].float(),
                            layer.mlp.experts.gate_up_proj[expert_index].float(),
                        )
                        gate, up = projections.chunk(2, dim=-1)
                        expert_output = torch.nn.functional.linear(
                            torch.nn.functional.silu(gate) * up,
                            layer.mlp.experts.down_proj[expert_index].float(),
                        )
                        float_output.index_add_(
                            0,
                            token_rows,
                            expert_output
                            * routing_weights[token_rows, selected_columns, None],
                        )
                    tactic_replays = []
                    if arguments.profile_tactics is not None and layer_index == 0:
                        gate_weights, up_weights = layer.mlp.experts.gate_up_proj.chunk(
                            2, dim=1
                        )
                        packed_weights = torch.cat(
                            (up_weights, gate_weights), dim=1
                        ).contiguous()
                        for first_profile in (-1, arguments.profile_tactics[0]):
                            for second_profile in (-1, arguments.profile_tactics[1]):
                                with torch.profiler.profile(
                                    activities=[
                                        torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA,
                                    ]
                                ) as profile:
                                    replay = cutlass_fused_moe(
                                        input=hidden_states.clone(),
                                        token_selected_experts=selected_experts.to(
                                            torch.int32
                                        ),
                                        token_final_scales=routing_weights,
                                        fc1_expert_weights=packed_weights,
                                        fc2_expert_weights=layer.mlp.experts.down_proj.contiguous(),
                                        output_dtype=hidden_states.dtype,
                                        quant_scales=None,
                                        output=torch.empty_like(hidden_states),
                                        activation_type=ActivationType.Swiglu,
                                        use_fused_finalize=False,
                                        profile_ids=[first_profile, second_profile],
                                    )[0]
                                profile.export_chrome_trace(
                                    str(
                                        arguments.output_directory
                                        / f"{trace_directory.parent.name}-{trace_directory.name}-{first_profile}-{second_profile}.trace.json"
                                    )
                                )
                                tactic_replays.append(
                                    {
                                        "profiles": [first_profile, second_profile],
                                        "native_vs_replay": asdict(
                                            compare_tensors(native_output, replay)
                                        ),
                                        "replay_vs_float32": asdict(
                                            compare_tensors(replay, float_output)
                                        ),
                                    }
                                )
                                torch.save(
                                    replay.detach().cpu(),
                                    arguments.output_directory
                                    / f"{trace_directory.parent.name}-{trace_directory.name}-{first_profile}-{second_profile}.pt",
                                )
                    else:
                        pass
                    expert_layers.append(
                        {
                            "trace_file": trace_file.name,
                            "layer": layer_index,
                            "expert_ids": selected_experts.cpu().tolist(),
                            "routing_weights_dtype": str(
                                captured[f"{prefix}weights_{token_count}"].dtype
                            ),
                            "selected_experts_are_topk": bool(
                                (selected_logits >= topk_threshold).all()
                            ),
                            "router_logits_vs_checkpoint": asdict(
                                compare_tensors(
                                    router_logits,
                                    torch.nn.functional.linear(
                                        hidden_states, layer.mlp.gate.weight
                                    ),
                                )
                            ),
                            "routing_weights_vs_softmax": asdict(
                                compare_tensors(routing_weights, selected_weights)
                            ),
                            "native_vs_reference": asdict(
                                compare_tensors(native_output, reference_output)
                            ),
                            "native_vs_float32": asdict(
                                compare_tensors(native_output, float_output)
                            ),
                            "reference_vs_float32": asdict(
                                compare_tensors(reference_output, float_output)
                            ),
                            "tactic_replays": tactic_replays,
                        }
                    )
            report = {
                "trace_directory": str(trace_directory),
                "frames": frames,
                "expert_layers": expert_layers,
                "decode_only": arguments.decode_only,
            }
            reports.append(report)
            destination = (
                arguments.output_directory
                / f"{trace_directory.parent.name}-{trace_directory.name}.json"
            )
            destination.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    {
                        "trace_directory": str(trace_directory),
                        "frames": len(frames),
                        "expert_layers": len(expert_layers),
                    }
                ),
                flush=True,
            )
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(
            {
                "model_path": arguments.model_path,
                "expert_attention": "not executed; fixed expert inputs only",
                "scope": "Measured feedback arithmetic and teacher-forced expert differences, not a speech-quality or speed verdict. FP32 is a diagnostic comparison, not a replacement checkpoint execution contract.",
                "reports": reports,
                "paired_frames": paired_frames,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
