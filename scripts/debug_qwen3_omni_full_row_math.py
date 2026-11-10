"""Check every captured real MoE row against independent expert arithmetic."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
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
    with (arguments.output_directory / "rows.jsonl").open("w") as reports:
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
