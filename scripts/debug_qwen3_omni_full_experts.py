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
from scripts.qwen3_omni_moe_tile_control import compile_tile_kernel, replace_tile_kernel
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
    parser.add_argument("--layer", type=int, action="append")
    parser.add_argument("--step", type=int, action="append")
    parser.add_argument("--tile-kernel-control", choices=("original", "barrier"))
    parser.add_argument("--benchmark-replays", type=int, default=0)
    parser.add_argument("--benchmark-warmup-replays", type=int, default=0)
    arguments = parser.parse_args()
    assert arguments.benchmark_replays >= 0
    assert arguments.benchmark_warmup_replays >= 0
    assert not arguments.benchmark_replays or arguments.tile_kernel_control is not None
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
    if arguments.step is not None:
        selected_steps = set(arguments.step)
        assert selected_steps <= {metadata.step for metadata in steps}
        steps = [metadata for metadata in steps if metadata.step in selected_steps]
    else:
        pass
    layers = (
        range(config.num_hidden_layers) if arguments.layer is None else arguments.layer
    )
    assert all(0 <= layer < config.num_hidden_layers for layer in layers)
    tile_module, tile_function = None, None
    summary = {
        "model_revision": arguments.model_revision,
        "autotune_cache_sha256": hashlib.sha256(
            arguments.autotune_cache.read_bytes()
        ).hexdigest(),
        "steps": len(steps),
        "selected_steps": arguments.step,
        "selected_layers": arguments.layer,
        "tune_max_num_tokens": arguments.tune_max_num_tokens,
        "tile_kernel_control": arguments.tile_kernel_control,
        "tile_module_handle": int(tile_module) if tile_module is not None else None,
        "routing_scale_dtype": arguments.routing_scale_dtype,
        "prefill_gemm1_fallback_above_rows": arguments.prefill_gemm1_fallback_above_rows,
        "routing_scale_scope": "The TopK hook precedes the model's routing-weight cast. Replay applies the serving source's selected storage dtype and restores FP32 for the kernel ABI.",
        "layers": [],
        "scope": "All actual prefill, decode and padding inputs, IDs and scales are replayed at their original row count. Exactness checks cover every non-padding row, including all prefill tokens, not only the final query. No missing rows are synthesized.",
    }
    with (arguments.output_directory / "replay.jsonl").open("w") as reports:
        for layer in layers:
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
            if arguments.tile_kernel_control is not None and tile_function is None:
                tile_module, tile_function = compile_tile_kernel(
                    arguments.output_directory, arguments.tile_kernel_control
                )
                summary["tile_module_handle"] = int(tile_module)
            else:
                pass
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
                if tile_function is not None:
                    warmup = cutlass_fused_moe(
                        input=inputs.clone(),
                        token_selected_experts=ids,
                        token_final_scales=scales,
                        fc1_expert_weights=w13,
                        fc2_expert_weights=w2,
                        output_dtype=inputs.dtype,
                        quant_scales=None,
                        use_fused_finalize=False,
                        tune_max_num_tokens=arguments.tune_max_num_tokens,
                    )[0]
                    assert torch.equal(actual[:real_rows], warmup[:real_rows])
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    graph_context = torch.cuda.graph(graph)
                else:
                    graph_context = nullcontext()
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
                with (
                    precision_context,
                    torch.cuda.nvtx.range(
                        f"actual-moe/layer-{layer:02d}/step-{metadata.step:06d}/{metadata.phase}"
                    ),
                    graph_context,
                ):
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
                if tile_function is not None:
                    replace_tile_kernel(graph, tile_function)
                    graph.replay()
                else:
                    pass
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
                if arguments.benchmark_replays:
                    for _ in range(arguments.benchmark_warmup_replays):
                        graph.replay()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(arguments.benchmark_replays):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    assert torch.equal(actual[:real_rows], output[:real_rows])
                    record["benchmark_replays"] = arguments.benchmark_replays
                    record["benchmark_warmup_replays"] = (
                        arguments.benchmark_warmup_replays
                    )
                    record["complete_moe_milliseconds_per_replay"] = (
                        start.elapsed_time(end) / arguments.benchmark_replays
                    )
                else:
                    pass
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
