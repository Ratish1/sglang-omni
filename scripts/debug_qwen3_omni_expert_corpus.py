"""Recompute selected experts over complete captured request lifetimes."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig

from sglang_omni.models.qwen3_omni.corpus_diagnostics import RequestTrace, StepMetadata


class ResponseIdentity(BaseModel):
    request_id: str
    sample_id: str


@dataclass(kw_only=True)
class ExpertReferences:
    outputs: dict[str, torch.Tensor]
    measurements: dict[str, torch.Tensor]


def pack_corpus(
    capture_directory: Path,
    identities: dict[str, str],
    layer_count: int,
    output_directory: Path,
) -> dict[str, torch.Tensor]:
    traces = {
        request_id: RequestTrace.model_validate_json(
            (capture_directory / "requests" / f"{request_id}.json").read_text()
        )
        for request_id in identities
    }
    frame_count = sum(len(trace.codes) for trace in traces.values())
    observed_steps: dict[str, list[int]] = {request_id: [] for request_id in traces}
    packed: dict[str, torch.Tensor] = {}
    offset = 0
    with (output_directory / "rows.jsonl").open("w") as output:
        for line in (capture_directory / "steps.jsonl").read_text().splitlines():
            metadata = StepMetadata.model_validate_json(line)
            rows = [
                row
                for row, request_id in enumerate(metadata.request_ids)
                if request_id in identities
            ]
            if metadata.predictor_batch == 0 or not rows:
                continue
            else:
                pass
            saved = torch.load(
                capture_directory / f"step-{metadata.step:06d}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            snapshots = saved["snapshots"]
            for layer in range(layer_count):
                prefix = f"backbone.{layer}."
                for name in (
                    "router.input",
                    "router.output",
                    "expert_ids",
                    "expert_weights",
                    "routed",
                    "shared",
                    "shared_gate.output",
                    "combined",
                ):
                    tensor = snapshots[prefix + name]
                    assert tensor.shape[0] == metadata.live_rows
                    if name not in packed:
                        packed[name] = torch.empty(
                            layer_count,
                            frame_count,
                            tensor.shape[1],
                            dtype=tensor.dtype,
                        )
                    else:
                        pass
                    packed[name][layer, offset : offset + len(rows)] = tensor[rows]
            for row in rows:
                request_id = metadata.request_ids[row]
                trace = traces[request_id]
                frame = metadata.frame_indices[row]
                assert frame == len(observed_steps[request_id])
                assert trace.steps[frame] == metadata.step
                assert saved["codes"][row].tolist() == trace.codes[frame]
                observed_steps[request_id].append(metadata.step)
                output.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "sample_id": identities[request_id],
                            "ordinal": trace.ordinal,
                            "frame": frame,
                            "frame_count": len(trace.codes),
                            "step": metadata.step,
                            "source_row": row,
                            "phase": metadata.phase,
                            "live_rows": metadata.live_rows,
                            "projection_rows": metadata.projection_rows[
                                "backbone.0.router"
                            ],
                        }
                    )
                    + "\n"
                )
            offset += len(rows)
            if metadata.step % 500 == 0:
                print(
                    json.dumps({"step": metadata.step, "packed_frames": offset}),
                    flush=True,
                )
            else:
                pass
    assert offset == frame_count
    assert all(
        trace.main_codes_match and observed_steps[request_id] == trace.steps
        for request_id, trace in traces.items()
    )
    torch.save(packed, output_directory / "selected-routes.pt")
    return packed


def load_layer_weights(
    model_directory: Path,
    layer: int,
    expert_count: int,
    weight_map: dict[str, str],
) -> dict[str, torch.Tensor]:
    prefix = f"talker.model.layers.{layer}.mlp."
    names = [
        f"{prefix}experts.{expert}.{projection}_proj.weight"
        for expert in range(expert_count)
        for projection in ("gate", "up", "down")
    ] + [
        prefix + name
        for name in (
            "gate.weight",
            "shared_expert.gate_proj.weight",
            "shared_expert.up_proj.weight",
            "shared_expert.down_proj.weight",
            "shared_expert_gate.weight",
        )
    ]
    with ExitStack() as stack:
        shards = {
            name: stack.enter_context(
                safe_open(model_directory / name, framework="pt", device="cpu")
            )
            for name in sorted({weight_map[name] for name in names})
        }
        weights = {
            name.removeprefix(prefix): shards[weight_map[name]].get_tensor(name).cuda()
            for name in names
        }
    assert all(weight.dtype == torch.bfloat16 for weight in weights.values())
    return weights


def reference_mlp(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> ExpertReferences:
    gate = inputs.double() @ gate_weight.double().T
    up = inputs.double() @ up_weight.double().T
    checkpoint_activation = F.silu(gate.bfloat16()) * up.bfloat16()
    fused_activation = (
        F.silu(gate.bfloat16().float()) * up.bfloat16().float()
    ).bfloat16()
    fp32_activation = (F.silu(gate.float()) * up.float()).bfloat16()
    fp64_activation = F.silu(gate) * up
    outputs = {"fp64": fp64_activation @ down_weight.double().T}
    measurements = {}
    for name, observed, reference in (
        ("gate_projection_rounding", gate.bfloat16(), gate),
        ("up_projection_rounding", up.bfloat16(), up),
        ("checkpoint_activation", checkpoint_activation, fp64_activation),
        ("bf16_gemm_activation", fused_activation, fp64_activation),
        ("fp32_gemm_activation", fp32_activation, fp64_activation),
    ):
        for metric, values in compare_outputs(observed, reference).items():
            measurements[f"{name}.{metric}"] = values
    for name, activation in (
        ("checkpoint_steps", checkpoint_activation),
        ("bf16_gemm_fused_activation", fused_activation),
        ("fp32_gemm_fused_activation", fp32_activation),
    ):
        projected = activation.double() @ down_weight.double().T
        outputs[name] = projected.bfloat16()
        for metric, values in compare_outputs(outputs[name], projected).items():
            measurements[f"{name}.down_projection_rounding.{metric}"] = values
    return ExpertReferences(outputs=outputs, measurements=measurements)


def compare_outputs(
    observed: torch.Tensor, reference: torch.Tensor
) -> dict[str, torch.Tensor]:
    difference = observed.double() - reference.double()
    return {
        "relative_l2": difference.norm(dim=-1)
        / reference.double().norm(dim=-1).clamp_min(torch.finfo(torch.float64).tiny),
        "max_absolute": difference.abs().amax(-1),
        "equal_coordinates": (observed == reference).double().mean(-1),
        "finite": reference.isfinite().all(-1),
    }


def measure_layer(
    packed: dict[str, torch.Tensor],
    layer: int,
    weights: dict[str, torch.Tensor],
    expert_count: int,
) -> dict[str, torch.Tensor]:
    inputs = packed["router.input"][layer].cuda()
    ids = packed["expert_ids"][layer].cuda().long()
    scales = packed["expert_weights"][layer].cuda()
    observed = packed["routed"][layer].cuda()
    assert ids.min() >= 0 and ids.max() < expert_count
    assert (ids.sort(-1).values.diff(dim=-1) > 0).all()
    assert torch.isfinite(inputs).all() and torch.isfinite(scales).all()
    frame_count, hidden_size = inputs.shape
    selected_written = torch.zeros_like(ids, dtype=torch.bool)
    expert_measurements: dict[str, torch.Tensor] = {}
    routed = {
        "fp64": torch.zeros(
            frame_count, hidden_size, device="cuda", dtype=torch.float64
        ),
        "checkpoint_steps": torch.zeros_like(inputs),
        "checkpoint_steps_bf16_scales": torch.zeros_like(inputs),
        "bf16_gemm_fused_activation": torch.zeros(
            frame_count, hidden_size, device="cuda", dtype=torch.float32
        ),
        "fp32_gemm_fused_activation": torch.zeros(
            frame_count, hidden_size, device="cuda", dtype=torch.float32
        ),
    }
    for expert in range(expert_count):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        else:
            pass
        references = reference_mlp(
            inputs[rows],
            weights[f"experts.{expert}.gate_proj.weight"],
            weights[f"experts.{expert}.up_proj.weight"],
            weights[f"experts.{expert}.down_proj.weight"],
        )
        selected_scales = scales[rows, slots, None]
        for name, values in references.measurements.items():
            if name not in expert_measurements:
                expert_measurements[name] = torch.empty_like(ids, dtype=values.dtype)
            else:
                pass
            expert_measurements[name][rows, slots] = values
        selected_written[rows, slots] = True
        for name, output in references.outputs.items():
            scaled = output.to(routed[name].dtype) * selected_scales.to(
                routed[name].dtype if name == "fp64" else torch.float32
            )
            # note (ratish): unique rows preserve ascending-expert BF16 accumulation.
            routed[name][rows] = routed[name][rows] + scaled.to(routed[name].dtype)
        rounded_scales = selected_scales.to(inputs.dtype)
        rounded_output = references.outputs["checkpoint_steps"] * rounded_scales
        routed["checkpoint_steps_bf16_scales"][rows] = (
            routed["checkpoint_steps_bf16_scales"][rows] + rounded_output
        )
    assert selected_written.all()
    measurements = {
        f"expert.{name}": values.cpu() for name, values in expert_measurements.items()
    }
    for name, output in routed.items():
        reference = output if name == "fp64" else output.bfloat16()
        for metric, values in compare_outputs(observed, reference).items():
            measurements[f"routed.{name}.{metric}"] = values.cpu()
    for metric, values in compare_outputs(
        routed["checkpoint_steps_bf16_scales"], routed["checkpoint_steps"]
    ).items():
        measurements[f"routing.bf16_scale_boundary.{metric}"] = values.cpu()
    shared = reference_mlp(
        inputs,
        weights["shared_expert.gate_proj.weight"],
        weights["shared_expert.up_proj.weight"],
        weights["shared_expert.down_proj.weight"],
    )
    for name, values in shared.measurements.items():
        measurements[f"shared_internal.{name}"] = values.cpu()
    for name, reference in shared.outputs.items():
        for metric, values in compare_outputs(
            packed["shared"][layer].cuda(), reference
        ).items():
            measurements[f"shared.{name}.{metric}"] = values.cpu()
    expected_gate = inputs.double() @ weights["shared_expert_gate.weight"].double().T
    for metric, values in compare_outputs(
        packed["shared_gate.output"][layer].cuda(), expected_gate
    ).items():
        measurements[f"shared_gate.fp64.{metric}"] = values.cpu()
    actual_shared = packed["shared"][layer].cuda()
    actual_gate = packed["shared_gate.output"][layer].cuda()
    combined = packed["combined"][layer].cuda()
    measurements["combined.actual_branches_exact"] = (
        (combined == observed + actual_shared * actual_gate.sigmoid()).all(-1).cpu()
    )
    normalized = packed["router.output"][layer].cuda().double().softmax(-1)
    normalized = normalized.gather(1, ids)
    normalized /= normalized.sum(-1, keepdim=True)
    measurements["routing.scales_max_absolute"] = (
        (scales.double() - normalized).abs().amax(-1).cpu()
    )
    measurements["routing.scale_sum_error"] = (scales.double().sum(-1) - 1).abs().cpu()
    measurements["routing.bf16_scale_cast_max_absolute"] = (
        (scales - scales.to(inputs.dtype).float()).abs().amax(-1).cpu()
    )
    return measurements


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument(
        "--response-directory", type=Path, action="append", required=True
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--packed-directory", type=Path)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config.text_config
    assert config.norm_topk_prob and config.hidden_act == "silu"
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
    identities = [
        ResponseIdentity.model_validate_json(path.read_text())
        for directory in arguments.response_directory
        for path in directory.glob("*.response.json")
    ]
    assert identities and len({record.request_id for record in identities}) == len(
        identities
    )
    if arguments.packed_directory is None:
        packed = pack_corpus(
            arguments.capture_directory,
            {record.request_id: record.sample_id for record in identities},
            config.num_hidden_layers,
            arguments.output_directory,
        )
    else:
        packed = torch.load(
            arguments.packed_directory / "selected-routes.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        rows = [
            json.loads(line)
            for line in (arguments.packed_directory / "rows.jsonl")
            .read_text()
            .splitlines()
        ]
        assert {row["request_id"] for row in rows} == {
            record.request_id for record in identities
        }
        assert len(rows) == packed["router.input"].shape[1]
    summary = {
        "requests": len(identities),
        "frames": packed["router.input"].shape[1],
        "layers": config.num_hidden_layers,
        "experts_per_token": config.num_experts_per_tok,
        "model_revision": arguments.model_revision,
        "scope": "All captured current-query rows and full generated lifetimes. Actual expert IDs and weights remain fixed. FP64 and explicitly named rounding variants are local mathematical references, not the actual kernel's internal intermediates or free-running quality results.",
        "routing_scale_scope": "checkpoint_steps retains FP32 routing scales for comparison with earlier evidence. checkpoint_steps_bf16_scales rounds those same frozen scales to BF16 before multiplication, keeping expert outputs and ascending-expert accumulation fixed. This isolates one checkpoint boundary, not a complete free-running checkpoint implementation.",
        "reused_packed_directory": (
            str(arguments.packed_directory)
            if arguments.packed_directory is not None
            else None
        ),
        "measurements": {},
    }
    for layer in range(config.num_hidden_layers):
        weights = load_layer_weights(
            model_directory, layer, config.num_experts, weight_map
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
        print(json.dumps({"layer": layer, "frames": summary["frames"]}), flush=True)
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
