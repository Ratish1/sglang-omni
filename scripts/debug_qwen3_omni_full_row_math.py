"""Check every captured real MoE row against independent expert arithmetic."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig

from scripts.debug_qwen3_omni_expert_corpus import load_layer_weights, measure_layer
from sglang_omni.models.qwen3_omni.corpus_diagnostics import StepMetadata


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--routing-only", action="store_true")
    parser.add_argument(
        "--routing-scale-dtype", choices=("bfloat16", "float32"), required=True
    )
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        local_files_only=True,
    ).talker_config.text_config
    assert config.norm_topk_prob and config.hidden_act == "silu"
    index_path = Path(
        hf_hub_download(
            arguments.model_path,
            "model.safetensors.index.json",
            revision=arguments.model_revision,
            local_files_only=True,
        )
    )
    weight_map = json.loads(index_path.read_text())["weight_map"]
    gate_weights = []
    for layer in range(config.num_hidden_layers):
        weight_name = f"talker.model.layers.{layer}.mlp.gate.weight"
        with safe_open(
            index_path.parent / weight_map[weight_name], framework="pt", device="cpu"
        ) as shard:
            weight = shard.get_tensor(weight_name)
            assert weight.dtype == torch.bfloat16
            gate_weights.append(weight.double())
    steps = [
        StepMetadata.model_validate_json(line)
        for line in (arguments.capture_directory / "steps.jsonl")
        .read_text()
        .splitlines()
    ]
    assert steps and len({metadata.step for metadata in steps}) == len(steps)
    fields = (
        "router.input",
        "router.output",
        "expert_ids",
        "expert_weights",
        "routed",
        "shared",
        "shared_gate.output",
        "combined",
    )
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in fields}
    real_rows_total = 0
    prefill_rows_total = 0
    routing_invalid_rows = 0
    maximum_precast_scale_error = 0.0
    projection_sensitive_rows = 0
    maximum_gate_projection_error = 0.0
    with (
        (arguments.output_directory / "rows.jsonl").open("w") as reports,
        (arguments.output_directory / "routing.jsonl").open("w") as routing_reports,
    ):
        for metadata in steps:
            saved = torch.load(
                arguments.capture_directory / f"experts-{metadata.step:06d}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            real_rows = int(saved["real_rows"])
            lengths = saved["extend_lengths"]
            assert sum(lengths) == real_rows
            assert len(lengths) == len(metadata.request_ids)
            snapshots = saved["snapshots"]
            if not arguments.routing_only:
                for name in fields:
                    values = []
                    for layer in range(config.num_hidden_layers):
                        tensor = snapshots[f"backbone.{layer}.{name}"]
                        assert (
                            tensor.shape[0]
                            == metadata.projection_rows[f"backbone.{layer}.router"]
                        )
                        assert real_rows <= tensor.shape[0]
                        values.append(tensor[:real_rows])
                    chunks[name].append(torch.stack(values))
            else:
                pass
            for layer in range(config.num_hidden_layers):
                prefix = f"backbone.{layer}."
                logits = snapshots[prefix + "router.output"][:real_rows].double()
                expert_ids = snapshots[prefix + "expert_ids"][:real_rows].long()
                assert logits.isfinite().all()
                assert expert_ids.shape == (real_rows, config.num_experts_per_tok)
                assert expert_ids.min() >= 0 and expert_ids.max() < logits.shape[1]
                assert (expert_ids.sort(-1).values.diff(dim=-1) > 0).all()
                selected_logits = logits.gather(1, expert_ids)
                unselected_maximum = logits.scatter(1, expert_ids, -torch.inf).amax(-1)
                margin = selected_logits.amin(-1) - unselected_maximum
                probabilities = logits.softmax(-1).gather(1, expert_ids)
                probabilities /= probabilities.sum(-1, keepdim=True)
                recorded_scales = snapshots[prefix + "expert_weights"][:real_rows]
                assert recorded_scales.isfinite().all() and (recorded_scales >= 0).all()
                scale_errors = (recorded_scales.double() - probabilities).abs().amax(-1)
                reference_logits = (
                    snapshots[prefix + "router.input"][:real_rows].double()
                    @ gate_weights[layer].T
                )
                assert reference_logits.isfinite().all()
                projection_errors = (logits - reference_logits).abs().amax(-1)
                rounded_projection_errors = (
                    (logits - reference_logits.bfloat16().double()).abs().amax(-1)
                )
                reference_boundary_margin = reference_logits.gather(1, expert_ids).amin(
                    -1
                ) - reference_logits.scatter(1, expert_ids, -torch.inf).amax(-1)
                projection_sensitive_rows += int((reference_boundary_margin < 0).sum())
                maximum_gate_projection_error = max(
                    maximum_gate_projection_error, float(projection_errors.max())
                )
                invalid_rows = (margin < 0).nonzero().flatten().tolist()
                routing_invalid_rows += len(invalid_rows)
                maximum_precast_scale_error = max(
                    maximum_precast_scale_error, float(scale_errors.max())
                )
                routing_reports.write(
                    json.dumps(
                        {
                            "step": metadata.step,
                            "layer": layer,
                            "real_rows": real_rows,
                            "invalid_source_rows": invalid_rows,
                            "topk_logit_boundary_margin": margin.tolist(),
                            "precast_scale_max_absolute_to_fp64": scale_errors.tolist(),
                            "gate_projection_max_absolute_to_fp64": projection_errors.tolist(),
                            "gate_projection_max_absolute_to_bf16_rounded_fp64": rounded_projection_errors.tolist(),
                            "selected_boundary_margin_against_fp64_projection": reference_boundary_margin.tolist(),
                        }
                    )
                    + "\n"
                )
            source_row = 0
            for request_id, frame_index, length in zip(
                metadata.request_ids, metadata.frame_indices, lengths, strict=True
            ):
                for query_offset in range(length):
                    reports.write(
                        json.dumps(
                            {
                                "request_id": request_id,
                                "step": metadata.step,
                                "phase": metadata.phase,
                                "frame_index": frame_index,
                                "source_row": source_row,
                                "query_offset": query_offset,
                            }
                        )
                        + "\n"
                    )
                    source_row += 1
            assert source_row == real_rows
            if metadata.phase == "EXTEND":
                prefill_rows_total += real_rows
            else:
                assert metadata.phase == "DECODE"
                assert all(length == 1 for length in lengths)
            real_rows_total += real_rows
    routing_summary = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        "steps": len(steps),
        "row_layer_checks": real_rows_total * config.num_hidden_layers,
        "invalid_topk_rows": routing_invalid_rows,
        "maximum_precast_scale_error_to_fp64": maximum_precast_scale_error,
        "projection_sensitive_rows": projection_sensitive_rows,
        "maximum_gate_projection_error_to_fp64": maximum_gate_projection_error,
        "scope": "Actual recorded logits and selected IDs, with ties accepted at the top-k boundary. The independent gate projection uses actual BF16 inputs and checkpoint weights in FP64. Selection differences against that projection measure rounding sensitivity, not an invalid BF16 top-k. Normalized-weight differences are reported before the serving scale cast, without an invented tolerance.",
    }
    (arguments.output_directory / "routing-summary.json").write_text(
        json.dumps(routing_summary, indent=2)
    )
    if arguments.routing_only:
        print(json.dumps(routing_summary), flush=True)
        return
    else:
        pass
    packed = {name: torch.cat(values, dim=1) for name, values in chunks.items()}
    chunks.clear()
    packed["expert_weights"] = (
        packed["expert_weights"]
        .to(
            {"bfloat16": torch.bfloat16, "float32": torch.float32}[
                arguments.routing_scale_dtype
            ]
        )
        .float()
    )
    assert packed["router.input"].shape == (
        config.num_hidden_layers,
        real_rows_total,
        config.hidden_size,
    )
    assert torch.isfinite(packed["routed"]).all()
    summary = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip(),
        "model_revision": arguments.model_revision,
        "steps": len(steps),
        "layers": config.num_hidden_layers,
        "real_rows_per_layer": real_rows_total,
        "prefill_rows_per_layer": prefill_rows_total,
        "routing_scale_dtype": arguments.routing_scale_dtype,
        "scope": "Every captured real row, including all prefill queries and neighboring requests. Independent FP64 and explicit rounding references reuse actual inputs, expert IDs and consumed scales. Padding is not synthesized. Numerical differences are observations, not an arbitrary tolerance or a speech-quality verdict.",
        "measurements": {},
    }
    for layer in range(config.num_hidden_layers):
        weights = load_layer_weights(
            index_path.parent, layer, config.num_experts, weight_map
        )
        measurements = measure_layer(packed, layer, weights, config.num_experts)
        torch.save(measurements, arguments.output_directory / f"layer-{layer:02d}.pt")
        summary["measurements"][str(layer)] = {
            name: {
                "count": values.numel(),
                "finite_count": int(values.isfinite().sum()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.double().mean()),
            }
            for name, values in measurements.items()
        }
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(json.dumps({"layer": layer, "real_rows": real_rows_total}), flush=True)


if __name__ == "__main__":
    main()
