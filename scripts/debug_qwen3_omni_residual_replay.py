"""Reconstruct every selected residual update from actual layer contributions."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from sgl_kernel import fused_add_rmsnorm, rmsnorm
from transformers import Qwen3OmniMoeConfig

from scripts.debug_qwen3_omni_expert_corpus import compare_outputs
from sglang_omni.models.qwen3_omni.corpus_diagnostics import StepMetadata


def checkpoint_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> torch.Tensor:
    values = hidden_states.float()
    normalized = values * (values.square().mean(-1, keepdim=True) + epsilon).rsqrt()
    return normalized.to(hidden_states.dtype) * weight


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config.text_config
    checkpoint_directory = Path(
        snapshot_download(
            arguments.model_path,
            revision=arguments.model_revision,
            local_files_only=True,
        )
    )
    weight_map = json.loads(
        (checkpoint_directory / "model.safetensors.index.json").read_text()
    )["weight_map"]
    names = [
        f"talker.model.layers.{layer}.{name}.weight"
        for layer in range(config.num_hidden_layers)
        for name in ("input_layernorm", "post_attention_layernorm")
    ] + ["talker.model.norm.weight"]
    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(checkpoint_directory / shard, framework="pt", device="cpu")
            )
            for shard in {weight_map[name] for name in names}
        }
        weights = {
            name: handles[weight_map[name]].get_tensor(name).cuda() for name in names
        }
    assert all(weight.dtype == torch.bfloat16 for weight in weights.values())
    requested_ids = set(arguments.request_id)
    checked_steps = {request_id: [] for request_id in requested_ids}
    exact_normalizations = 0
    checked_normalizations = 0
    exact_final_queries = 0
    with (arguments.output_directory / "rows.jsonl").open("w") as reports:
        for line in (
            (arguments.capture_directory / "steps.jsonl").read_text().splitlines()
        ):
            metadata = StepMetadata.model_validate_json(line)
            saved = torch.load(
                arguments.capture_directory / f"experts-{metadata.step:06d}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            snapshots = saved["snapshots"]
            queries = torch.load(
                arguments.capture_directory / f"step-{metadata.step:06d}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            offset = 0
            for row, request_id in enumerate(metadata.request_ids):
                query_rows = saved["extend_lengths"][row]
                selected = slice(offset, offset + query_rows)
                offset += query_rows
                if request_id not in requested_ids:
                    continue
                else:
                    pass
                phase = (
                    "before_prefill" if metadata.phase == "EXTEND" else "before_decode"
                )
                before = torch.load(
                    arguments.capture_directory
                    / "conditioning"
                    / request_id
                    / f"{metadata.frame_indices[row]:04d}-{phase}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                inputs = (
                    before["prefill_embeds"]
                    if metadata.phase == "EXTEND"
                    else (before["feedback_input"] + before["text_input"])[None]
                ).cuda()
                assert inputs.shape == (query_rows, config.hidden_size)
                residual = inputs.clone()
                contribution = inputs
                for layer in range(config.num_hidden_layers):
                    prefix = f"backbone.{layer}."
                    for phase_name, norm_name, expected_name in (
                        ("attention", "input_layernorm", "qkv.input"),
                        ("feed_forward", "post_attention_layernorm", "router.input"),
                    ):
                        weight = weights[
                            f"talker.model.layers.{layer}.{norm_name}.weight"
                        ]
                        if phase_name == "attention" and layer == 0:
                            normalized = rmsnorm(
                                contribution, weight, config.rms_norm_eps
                            )
                        else:
                            normalized = contribution.clone()
                            fused_add_rmsnorm(
                                normalized, residual, weight, config.rms_norm_eps
                            )
                        actual = snapshots[prefix + expected_name][selected].cuda()
                        assert actual.isfinite().all() and normalized.isfinite().all()
                        exact = torch.equal(normalized, actual)
                        exact_normalizations += exact
                        checked_normalizations += 1
                        checkpoint = checkpoint_norm(
                            residual, weight, config.rms_norm_eps
                        )
                        difference = compare_outputs(actual, checkpoint)
                        reports.write(
                            json.dumps(
                                {
                                    "request_id": request_id,
                                    "step": metadata.step,
                                    "frame_index": metadata.frame_indices[row],
                                    "phase": metadata.phase,
                                    "layer": layer,
                                    "boundary": phase_name,
                                    "rows": query_rows,
                                    "native_residual_reconstruction_exact": exact,
                                    "native_maximum_absolute": float(
                                        (normalized.float() - actual.float())
                                        .abs()
                                        .max()
                                    ),
                                    "checkpoint_stored_residual_relative_l2": difference[
                                        "relative_l2"
                                    ]
                                    .cpu()
                                    .tolist(),
                                }
                            )
                            + "\n"
                        )
                        contribution = (
                            snapshots[
                                prefix
                                + (
                                    "attention_projection.output"
                                    if phase_name == "attention"
                                    else "combined"
                                )
                            ][selected]
                            .cuda()
                            .clone()
                        )
                weight = weights["talker.model.norm.weight"]
                fused_add_rmsnorm(contribution, residual, weight, config.rms_norm_eps)
                actual_query = queries["predictor_hidden"][row : row + 1]
                exact_final_queries += torch.equal(
                    contribution[-1:].cpu(), actual_query
                )
                checked_steps[request_id].append(metadata.step)
                reports.flush()
            assert offset == saved["real_rows"]
    for request_id in requested_ids:
        trace = json.loads(
            (
                arguments.capture_directory / "requests" / f"{request_id}.json"
            ).read_text()
        )
        assert checked_steps[request_id] == trace["steps"]
    summary = {
        "model_revision": arguments.model_revision,
        "checked_frames": {
            request_id: len(steps) for request_id, steps in checked_steps.items()
        },
        "checked_normalizations": checked_normalizations,
        "exact_native_normalizations": exact_normalizations,
        "exact_final_queries": exact_final_queries,
        "scope": "All selected real prefill rows and decode frames. Reconstruct each residual from actual attention and MoE contributions using the native norm kernels. Checkpoint comparison normalizes the same stored BF16 residual with the checkpoint's cast-before-weight order, not a different generated trajectory.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
