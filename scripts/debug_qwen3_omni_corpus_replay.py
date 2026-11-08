"""Replay every captured predictor frame and measure corpus-wide decision margins."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel

from sglang_omni.models.qwen3_omni.components.talker import (
    PredictorDecodeGraph,
    Qwen3OmniTalker,
)
from sglang_omni.models.qwen3_omni.stages import create_talker_ar_executor_from_config
from sglang_omni.vendor.sglang.layers import ReplicatedLinear


class StepMetadata(BaseModel):
    step: int
    phase: str
    captured_graph: bool
    request_ids: list[str]
    request_ordinals: list[int]
    frame_indices: list[int]
    live_rows: int
    predictor_batch: int
    projection_rows: dict[str, int]
    finite_including_padding: dict[str, bool]
    sequence_lengths: list[int]
    predictor_geometry: dict[str, int]
    predictor_stages: dict[str, int]


@dataclass(kw_only=True)
class HeadCapture:
    inputs: torch.Tensor
    outputs: torch.Tensor

    def __call__(
        self,
        module: ReplicatedLinear,
        arguments: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        self.inputs.copy_(arguments[0].reshape_as(self.inputs))
        self.outputs.copy_(output[0].reshape_as(self.outputs))


@dataclass(kw_only=True)
class ReplayGraphs:
    plain: PredictorDecodeGraph
    observed: PredictorDecodeGraph
    heads: list[HeadCapture]


class CorpusReplay:
    def __init__(self, native: Qwen3OmniTalker) -> None:
        self.native = native
        self.graphs: dict[tuple[int, torch.dtype], ReplayGraphs] = {}
        self.head_weights = [
            head.weight.double() for head in native.code_predictor.lm_head
        ]
        self.router_weights = [
            layer.mlp.gate.weight.double() for layer in native.model.layers
        ]

    def replay(
        self,
        metadata: StepMetadata,
        tensors: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> tuple[dict[str, bool], dict[str, torch.Tensor]]:
        layer0_codes = tensors["layer0_codes"]
        hidden_states = tensors["predictor_hidden"]
        code_int32 = tensors["code_int32"]
        assert isinstance(layer0_codes, torch.Tensor)
        assert isinstance(hidden_states, torch.Tensor)
        assert isinstance(code_int32, torch.Tensor)
        code_dtype = torch.int32 if bool(code_int32) else torch.int64
        geometry = (metadata.predictor_batch, code_dtype)
        fused_layers = self.native.predictor_fused_layers
        assert fused_layers is not None
        for stage, count in metadata.predictor_stages.items():
            operation, rows = stage.split(":")
            fused_layers.fitted_stages[(operation, int(rows))] = count
        if geometry not in self.graphs:
            plain = PredictorDecodeGraph(self.native, *geometry)
            heads = []
            handles = []
            for head in self.native.code_predictor.lm_head:
                capture = HeadCapture(
                    inputs=torch.empty(
                        metadata.predictor_batch,
                        head.weight.shape[1],
                        device=head.weight.device,
                        dtype=head.weight.dtype,
                    ),
                    outputs=torch.empty(
                        metadata.predictor_batch,
                        head.weight.shape[0],
                        device=head.weight.device,
                        dtype=head.weight.dtype,
                    ),
                )
                heads.append(capture)
                handles.append(head.register_forward_hook(capture))
            observed = PredictorDecodeGraph(self.native, *geometry)
            for handle in handles:
                handle.remove()
            self.graphs[geometry] = ReplayGraphs(
                plain=plain, observed=observed, heads=heads
            )
        else:
            pass
        graphs = self.graphs[geometry]
        layer0_codes = layer0_codes.to(device="cuda", dtype=code_dtype)[:, None]
        hidden_states = hidden_states.cuda()[:, None]
        expected_codes = tensors["codes"]
        expected_feedback = tensors["feedback"]
        snapshots = tensors["snapshots"]
        counters = tensors["predictor_counters"]
        assert isinstance(expected_codes, torch.Tensor)
        assert isinstance(expected_feedback, torch.Tensor)
        assert isinstance(snapshots, dict)
        assert isinstance(counters, torch.Tensor)
        checks = {"recorded_counters_zero": not bool(counters.any())}
        for name, graph in (("plain", graphs.plain), ("observed", graphs.observed)):
            codes, feedback = graph.replay(layer0_codes, hidden_states)
            checks[f"{name}_codes_exact"] = torch.equal(
                codes[: metadata.live_rows, :, 0].cpu(), expected_codes
            )
            checks[f"{name}_feedback_exact"] = torch.equal(
                feedback[: metadata.live_rows, 0].cpu(), expected_feedback
            )
        checks["head_inputs_exact"] = all(
            torch.equal(head.inputs.cpu(), snapshots[f"predictor.{group}.input"])
            for group, head in enumerate(graphs.heads)
        )
        checks["head_logits_exact"] = all(
            torch.equal(head.outputs.cpu(), snapshots[f"predictor.{group}.output"])
            for group, head in enumerate(graphs.heads)
        )
        sequential_feedback = self.native.model.codec_embedding(
            expected_codes[:, 0].cuda()
        )
        for group, embedding in enumerate(
            self.native.code_predictor.model.codec_embedding
        ):
            sequential_feedback.add_(embedding(expected_codes[:, group + 1].cuda()))
        checks["sequential_feedback_exact"] = torch.equal(
            sequential_feedback.cpu(), expected_feedback
        )
        measurements: dict[str, torch.Tensor] = {}
        for group, weight in enumerate(self.head_weights):
            recorded = snapshots[f"predictor.{group}.output"][
                : metadata.live_rows
            ].cuda()
            projection = (
                snapshots[f"predictor.{group}.input"][: metadata.live_rows]
                .cuda()
                .double()
                @ weight.T
            )
            top_two = recorded.float().topk(2, dim=-1).values
            selected = expected_codes[:, group + 1].cuda()
            measurements[f"head.{group}.tie_count"] = (
                recorded == recorded.max(-1, keepdim=True).values
            ).sum(-1)
            measurements[f"head.{group}.margin"] = top_two[:, 0] - top_two[:, 1]
            measurements[f"head.{group}.selected_is_argmax"] = (
                selected == recorded.argmax(-1)
            )
            measurements[f"head.{group}.fp64_changes_argmax"] = (
                selected != projection.argmax(-1)
            )
            measurements[f"head.{group}.fp64_selected_gap"] = (
                projection.max(-1).values
                - projection.gather(1, selected[:, None])[:, 0]
            )
        for layer, weight in enumerate(self.router_weights):
            prefix = f"backbone.{layer}"
            recorded = snapshots[f"{prefix}.router.output"].cuda()
            projection = snapshots[f"{prefix}.router.input"].cuda().double() @ weight.T
            selected = snapshots[f"{prefix}.expert_ids"].cuda().long()
            selected_count = selected.shape[-1]
            top_values = recorded.float().topk(selected_count + 1, dim=-1).values
            actual_set = selected.sort(-1).values
            measurements[f"router.{layer}.cutoff_margin"] = (
                top_values[:, selected_count - 1] - top_values[:, selected_count]
            )
            measurements[f"router.{layer}.selected_set_valid"] = (
                recorded.gather(1, selected) >= top_values[:, selected_count - 1, None]
            ).all(-1) & (actual_set[:, 1:] != actual_set[:, :-1]).all(-1)
            measurements[f"router.{layer}.fp64_changes_set"] = (
                actual_set
                != projection.topk(selected_count, dim=-1).indices.sort(-1).values
            ).any(-1)
            measurements[f"router.{layer}.weights_finite"] = (
                snapshots[f"{prefix}.expert_weights"].isfinite().all(-1).cuda()
            )
        dense = tensors["dense"]
        assert isinstance(dense, dict)
        for name, tensor in dense.items():
            if name == "seen":
                checks["dense_layers_seen"] = bool(tensor.all())
            elif name.startswith("opening_"):
                rows = tensor[:, : metadata.live_rows].float()
                for layer in range(rows.shape[0]):
                    measurements[f"dense.{name}.{layer}.finite"] = (
                        rows[layer].isfinite().all(-1)
                    )
                    measurements[f"dense.{name}.{layer}.norm"] = rows[layer].norm(
                        dim=-1
                    )
            else:
                rows = tensor[:, :, : metadata.live_rows].float().flatten(3)
                finite = rows.isfinite().all(-1)
                norms = rows.norm(dim=-1)
                for group in range(rows.shape[0]):
                    for layer in range(rows.shape[1]):
                        measurements[f"dense.{name}.{group}.{layer}.finite"] = finite[
                            group, layer
                        ]
                        measurements[f"dense.{name}.{group}.{layer}.norm"] = norms[
                            group, layer
                        ]
        return checks, {name: values.cpu() for name, values in measurements.items()}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-running-requests", type=int, required=True)
    parser.add_argument("--memory-fraction", type=float, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    scheduler = create_talker_ar_executor_from_config(
        arguments.model_path,
        gpu_id=0,
        max_running_requests=arguments.max_running_requests,
        total_gpu_memory_fraction=arguments.memory_fraction,
        code2wav_in_process=True,
    )
    replay = CorpusReplay(scheduler.model_runner.model)
    failures: Counter[str] = Counter()
    request_frames: Counter[str] = Counter()
    steps = 0
    with (
        (arguments.capture_directory / "steps.jsonl").open() as manifest,
        (arguments.output_directory / "replay.jsonl").open("w") as reports,
    ):
        for line in manifest:
            metadata = StepMetadata.model_validate_json(line)
            if metadata.predictor_batch == 0:
                continue
            else:
                pass
            tensors = torch.load(
                arguments.capture_directory / f"step-{metadata.step:06d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            checks, measurements = replay.replay(metadata, tensors)
            failures.update(name for name, passed in checks.items() if not passed)
            for request_id in metadata.request_ids:
                request_frames[request_id] += 1
            torch.save(
                measurements,
                arguments.output_directory / f"step-{metadata.step:06d}.pt",
            )
            reports.write(json.dumps({"step": metadata.step, "checks": checks}) + "\n")
            reports.flush()
            steps += 1
            if steps % 100 == 0:
                print(
                    json.dumps(
                        {
                            "steps": steps,
                            "requests": len(request_frames),
                            "failures": failures,
                        }
                    ),
                    flush=True,
                )
            else:
                pass
    summary = {
        "steps": steps,
        "requests": len(request_frames),
        "request_frames": request_frames,
        "failures": failures,
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
