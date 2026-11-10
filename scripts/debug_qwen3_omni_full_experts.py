"""Replay every original MoE row from a complete expert capture."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import cutlass_fused_moe
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeConfig

from scripts.debug_qwen3_omni_expert_corpus import compare_outputs, load_layer_weights
from sglang_omni.models.qwen3_omni.corpus_diagnostics import StepMetadata


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--autotune-cache", type=Path, required=True)
    parser.add_argument("--tune-max-num-tokens", type=int, required=True)
    parser.add_argument(
        "--routing-scale-dtype", choices=("bfloat16", "float32"), required=True
    )
    parser.add_argument("--prefill-gemm1-fallback-above-rows", type=int)
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
    summary = {
        "model_revision": arguments.model_revision,
        "autotune_cache_sha256": hashlib.sha256(
            arguments.autotune_cache.read_bytes()
        ).hexdigest(),
        "steps": len(steps),
        "routing_scale_dtype": arguments.routing_scale_dtype,
        "prefill_gemm1_fallback_above_rows": arguments.prefill_gemm1_fallback_above_rows,
        "routing_scale_scope": "The TopK hook precedes the model's routing-weight cast. Replay applies the serving source's selected storage dtype and restores FP32 for the kernel ABI.",
        "layers": [],
        "scope": "All actual prefill, decode and padding inputs, IDs and scales are replayed at their original row count. Exactness checks cover every non-padding row, including all prefill tokens, not only the final query. No missing rows are synthesized.",
    }
    with (arguments.output_directory / "replay.jsonl").open("w") as reports:
        for layer in range(config.num_hidden_layers):
            weights = load_layer_weights(
                model_directory, layer, config.num_experts, weight_map
            )
            w13 = torch.stack(
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
            w2 = torch.stack(
                [
                    weights[f"experts.{expert}.down_proj.weight"]
                    for expert in range(config.num_experts)
                ]
            )
            exact_steps = 0
            real_rows_total = 0
            mismatches = []
            for metadata in steps:
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
                scales = (
                    snapshots[prefix + "expert_weights"]
                    .cuda()
                    .to(
                        {"bfloat16": torch.bfloat16, "float32": torch.float32}[
                            arguments.routing_scale_dtype
                        ]
                    )
                    .float()
                )
                actual = snapshots[prefix + "routed"].cuda()
                assert inputs.shape[0] == metadata.projection_rows[prefix + "router"]
                real_rows = int(saved["real_rows"])
                assert 0 < real_rows <= inputs.shape[0]
                assert int(saved["query_indices"].max()) < real_rows
                if (
                    arguments.prefill_gemm1_fallback_above_rows is not None
                    and metadata.phase == "EXTEND"
                    and inputs.shape[0] > arguments.prefill_gemm1_fallback_above_rows
                ):
                    precision_context = autotune(
                        False, skip_ops={"trtllm::fused_moe::gemm1"}
                    )
                else:
                    precision_context = nullcontext()
                with precision_context:
                    output = cutlass_fused_moe(
                        input=inputs,
                        token_selected_experts=ids,
                        token_final_scales=scales,
                        fc1_expert_weights=w13,
                        fc2_expert_weights=w2,
                        output_dtype=inputs.dtype,
                        quant_scales=None,
                        use_fused_finalize=False,
                        tune_max_num_tokens=arguments.tune_max_num_tokens,
                    )[0]
                metrics = compare_outputs(actual[:real_rows], output[:real_rows])
                assert metrics["finite"].all()
                exact = torch.equal(actual[:real_rows], output[:real_rows])
                exact_steps += int(exact)
                real_rows_total += real_rows
                record = {
                    "layer": layer,
                    "step": metadata.step,
                    "phase": metadata.phase,
                    "request_ids": metadata.request_ids,
                    "frames": metadata.frame_indices,
                    "real_rows": real_rows,
                    "projection_rows": inputs.shape[0],
                    "exact": exact,
                    "max_absolute": float(metrics["max_absolute"].max()),
                }
                reports.write(json.dumps(record) + "\n")
                if not exact:
                    mismatches.append(metadata.step)
                    torch.save(
                        {
                            "actual": actual.cpu(),
                            "replayed": output.cpu(),
                            "metrics": {
                                name: values.cpu() for name, values in metrics.items()
                            },
                        },
                        arguments.output_directory
                        / f"mismatch-layer-{layer:02d}-step-{metadata.step:06d}.pt",
                    )
                else:
                    pass
            summary["layers"].append(
                {
                    "layer": layer,
                    "steps": len(steps),
                    "real_rows": real_rows_total,
                    "exact_steps": exact_steps,
                    "mismatches": mismatches,
                }
            )
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            reports.flush()
            print(json.dumps(summary["layers"][-1]), flush=True)


if __name__ == "__main__":
    main()
