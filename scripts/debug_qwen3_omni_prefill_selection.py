"""Compare both expert projections on the exact complete prefill operands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import cutlass_fused_moe
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeConfig

from scripts.debug_qwen3_omni_expert_corpus import compare_outputs, load_layer_weights
from sglang_omni.models.qwen3_omni.corpus_diagnostics import StepMetadata


def run_experts(
    inputs: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    up_gate_weight: torch.Tensor,
    down_weight: torch.Tensor,
    tune_max_num_tokens: int,
) -> torch.Tensor:
    return cutlass_fused_moe(
        input=inputs.clone(),
        token_selected_experts=expert_ids,
        token_final_scales=expert_weights,
        fc1_expert_weights=up_gate_weight,
        fc2_expert_weights=down_weight,
        output_dtype=inputs.dtype,
        quant_scales=None,
        use_fused_finalize=False,
        tune_max_num_tokens=tune_max_num_tokens,
    )[0]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--tune-max-num-tokens", type=int, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    assert AutoTuner.get().load_configs(str(arguments.autotune_cache))
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config.text_config
    model_directory = Path(
        snapshot_download(
            arguments.model_path,
            revision=arguments.model_revision,
            local_files_only=True,
        )
    )
    weight_map = json.loads(
        (model_directory / "model.safetensors.index.json").read_text()
    )["weight_map"]
    steps = [
        StepMetadata.model_validate_json(line)
        for line in (arguments.capture_directory / "steps.jsonl")
        .read_text()
        .splitlines()
    ]
    prefill_steps = [step for step in steps if step.phase == "EXTEND"]
    summary = {
        "model_revision": arguments.model_revision,
        "autotune_cache_sha256": hashlib.sha256(
            arguments.autotune_cache.read_bytes()
        ).hexdigest(),
        "scope": "Original complete prefill operands, IDs, weights and padding. Cached output must first reproduce the recorded native output exactly. All four cached/default GEMM1/GEMM2 combinations are compared with original inputs cloned independently because fused experts can mutate them.",
        "layers": [],
    }
    with (arguments.output_directory / "comparisons.jsonl").open("w") as reports:
        for layer in range(config.num_hidden_layers):
            weights = load_layer_weights(
                model_directory, layer, config.num_experts, weight_map
            )
            up_gate = torch.stack(
                [
                    torch.cat(
                        (
                            weights[f"experts.{expert}.up_proj.weight"],
                            weights[f"experts.{expert}.gate_proj.weight"],
                        )
                    )
                    for expert in range(config.num_experts)
                ]
            )
            down = torch.stack(
                [
                    weights[f"experts.{expert}.down_proj.weight"]
                    for expert in range(config.num_experts)
                ]
            )
            unchanged = 0
            unchanged_second = 0
            unchanged_second_with_default_first = 0
            for metadata in prefill_steps:
                saved = torch.load(
                    arguments.capture_directory / f"experts-{metadata.step:06d}.pt",
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                snapshots = saved["snapshots"]
                prefix = f"backbone.{layer}."
                inputs = snapshots[prefix + "router.input"].cuda()
                ids = snapshots[prefix + "expert_ids"].cuda()
                scales = snapshots[prefix + "expert_weights"].cuda()
                actual = snapshots[prefix + "routed"].cuda()
                real_rows = int(saved["real_rows"])
                cached = run_experts(
                    inputs, ids, scales, up_gate, down, arguments.tune_max_num_tokens
                )
                assert torch.equal(cached[:real_rows], actual[:real_rows]), (
                    layer,
                    metadata.step,
                )
                with autotune(False, skip_ops={"trtllm::fused_moe::gemm1"}):
                    default = run_experts(
                        inputs,
                        ids,
                        scales,
                        up_gate,
                        down,
                        arguments.tune_max_num_tokens,
                    )
                with autotune(False, skip_ops={"trtllm::fused_moe::gemm2"}):
                    default_second = run_experts(
                        inputs,
                        ids,
                        scales,
                        up_gate,
                        down,
                        arguments.tune_max_num_tokens,
                    )
                with autotune(
                    False,
                    skip_ops={
                        "trtllm::fused_moe::gemm1",
                        "trtllm::fused_moe::gemm2",
                    },
                ):
                    default_both = run_experts(
                        inputs,
                        ids,
                        scales,
                        up_gate,
                        down,
                        arguments.tune_max_num_tokens,
                    )
                metrics = compare_outputs(cached[:real_rows], default[:real_rows])
                assert metrics["finite"].all()
                exact = torch.equal(cached[:real_rows], default[:real_rows])
                unchanged += int(exact)
                second_metrics = compare_outputs(
                    cached[:real_rows], default_second[:real_rows]
                )
                both_metrics = compare_outputs(
                    default[:real_rows], default_both[:real_rows]
                )
                assert second_metrics["finite"].all() and both_metrics["finite"].all()
                second_exact = torch.equal(
                    cached[:real_rows], default_second[:real_rows]
                )
                both_exact = torch.equal(default[:real_rows], default_both[:real_rows])
                unchanged_second += int(second_exact)
                unchanged_second_with_default_first += int(both_exact)
                record = {
                    "layer": layer,
                    "step": metadata.step,
                    "request_ids": metadata.request_ids,
                    "real_rows": real_rows,
                    "graph_rows": inputs.shape[0],
                    "cached_replay_exact": True,
                    "default_matches_cached": exact,
                    "maximum_relative_l2": float(metrics["relative_l2"].max()),
                    "maximum_absolute": float(metrics["max_absolute"].max()),
                    "default_second_matches_cached": second_exact,
                    "default_second_with_default_first_matches_default_first": both_exact,
                    "second_projection_maximum_relative_l2": float(
                        second_metrics["relative_l2"].max()
                    ),
                    "second_projection_with_default_first_maximum_relative_l2": float(
                        both_metrics["relative_l2"].max()
                    ),
                }
                reports.write(json.dumps(record) + "\n")
            summary["layers"].append(
                {
                    "layer": layer,
                    "prefill_steps": len(prefill_steps),
                    "unchanged_steps": unchanged,
                    "unchanged_second_projection_steps": unchanged_second,
                    "unchanged_second_with_default_first_steps": (
                        unchanged_second_with_default_first
                    ),
                }
            )
            reports.flush()
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            print(json.dumps(summary["layers"][-1]), flush=True)


if __name__ == "__main__":
    main()
