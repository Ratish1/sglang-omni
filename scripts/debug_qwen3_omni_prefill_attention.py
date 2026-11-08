"""Audit every real prefix row against checkpoint projections and causal attention."""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as functional
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerRotaryEmbedding,
    apply_rotary_pos_emb,
)

from scripts.debug_qwen3_omni_expert_corpus import compare_outputs
from sglang_omni.models.qwen3_omni.corpus_diagnostics import StepMetadata


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config.text_config
    assert config.sliding_window is None
    checkpoint_rotary = Qwen3OmniMoeTalkerRotaryEmbedding(config).cuda()
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
        "prefill_steps": len(prefill_steps),
        "layers": [],
        "scope": "All real prefill rows, actual complete per-request K/V, independent checkpoint projection weights and FP64 causal GQA. Padding is retained in the capture but excluded from real-row metrics. Local numerical differences are not speech-quality verdicts.",
    }
    query_width = config.num_attention_heads * config.head_dim
    key_width = config.num_key_value_heads * config.head_dim
    with (arguments.output_directory / "rows.jsonl").open("w") as reports:
        for layer in range(config.num_hidden_layers):
            prefix = f"talker.model.layers.{layer}."
            names = [
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.shared_expert.gate_proj.weight",
                "mlp.shared_expert.up_proj.weight",
                "mlp.shared_expert.down_proj.weight",
            ]
            with ExitStack() as stack:
                handles = {
                    shard: stack.enter_context(
                        safe_open(model_directory / shard, framework="pt", device="cpu")
                    )
                    for shard in sorted({weight_map[prefix + name] for name in names})
                }
                weights = {
                    name: handles[weight_map[prefix + name]]
                    .get_tensor(prefix + name)
                    .cuda()
                    for name in names
                }
            projections = {
                "qkv": torch.cat(
                    [
                        weights[f"self_attn.{name}_proj.weight"]
                        for name in ("q", "k", "v")
                    ]
                ),
                "attention_projection": weights["self_attn.o_proj.weight"],
                "shared_up": torch.cat(
                    [
                        weights[f"mlp.shared_expert.{name}_proj.weight"]
                        for name in ("gate", "up")
                    ]
                ),
                "shared_down": weights["mlp.shared_expert.down_proj.weight"],
            }
            checked_rows = 0
            exact_projection_rows = {name: 0 for name in projections}
            maximum_attention_relative = 0.0
            maximum_rotated_relative = {name: 0.0 for name in ("query", "key")}
            for metadata in prefill_steps:
                saved = torch.load(
                    arguments.capture_directory / f"experts-{metadata.step:06d}.pt",
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                )
                snapshots = saved["snapshots"]
                real_rows = int(saved["real_rows"])
                assert sum(saved["extend_lengths"]) == real_rows
                observed = f"backbone.{layer}."
                projection_metrics = {}
                for name, weight in projections.items():
                    inputs = snapshots[observed + name + ".input"][:real_rows].cuda()
                    actual = snapshots[observed + name + ".output"][:real_rows].cuda()
                    expected = functional.linear(inputs, weight)
                    metrics = compare_outputs(actual, expected)
                    assert metrics["finite"].all()
                    exact_projection_rows[name] += int(
                        (actual == expected).all(-1).sum()
                    )
                    projection_metrics[name] = {
                        "maximum_relative_l2": float(metrics["relative_l2"].max()),
                        "maximum_absolute": float(metrics["max_absolute"].max()),
                    }
                projected_queries, projected_keys, _ = (
                    snapshots[observed + "qkv.output"][:real_rows]
                    .cuda()
                    .split((query_width, key_width, key_width), dim=-1)
                )
                normalized = []
                for name, projected in (
                    ("q", projected_queries),
                    ("k", projected_keys),
                ):
                    projected = projected.reshape(real_rows, -1, config.head_dim)
                    normalized_float = (
                        projected.float()
                        * (
                            projected.float().square().mean(-1, keepdim=True)
                            + config.rms_norm_eps
                        ).rsqrt()
                    )
                    normalized.append(
                        (
                            normalized_float.to(projected.dtype)
                            * weights[f"self_attn.{name}_norm.weight"]
                        ).transpose(0, 1)[None]
                    )
                positions = saved["mrope_positions"].cuda()
                assert positions.shape == (3, real_rows)
                cosine, sine = checkpoint_rotary(projected_queries, positions[:, None])
                rotated = apply_rotary_pos_emb(*normalized, cosine, sine)
                rotated_metrics = {}
                for name, expected in zip(("query", "key"), rotated, strict=True):
                    expected = expected[0].transpose(0, 1).reshape(real_rows, -1)
                    actual = snapshots[observed + name][:real_rows].cuda()
                    metrics = compare_outputs(actual, expected)
                    assert metrics["finite"].all()
                    maximum_rotated_relative[name] = max(
                        maximum_rotated_relative[name],
                        float(metrics["relative_l2"].max()),
                    )
                    rotated_metrics[name] = metrics
                offset = 0
                for row, request_id in enumerate(metadata.request_ids):
                    extend_length = saved["extend_lengths"][row]
                    prefix_length = saved["prefix_lengths"][row]
                    sequence_length = metadata.sequence_lengths[row]
                    assert extend_length + prefix_length == sequence_length
                    prefix_cache = torch.load(
                        arguments.capture_directory
                        / f"prefix-{request_id}-{metadata.step:06d}.pt",
                        map_location="cpu",
                        weights_only=True,
                        mmap=True,
                    )
                    selected = slice(offset, offset + extend_length)
                    keys = prefix_cache["keys"][layer]
                    values = prefix_cache["values"][layer]
                    assert keys.shape[0] == values.shape[0] == sequence_length
                    assert torch.equal(
                        snapshots[observed + "key"][selected],
                        keys[prefix_length:].reshape(extend_length, key_width),
                    )
                    assert torch.equal(
                        snapshots[observed + "value"][selected],
                        values[prefix_length:].reshape(extend_length, key_width),
                    )
                    assert torch.equal(
                        snapshots[observed + "qkv.output"][
                            selected, query_width + key_width :
                        ],
                        snapshots[observed + "value"][selected],
                    )
                    queries = (
                        snapshots[observed + "query"][selected]
                        .cuda()
                        .double()
                        .reshape(
                            extend_length, config.num_attention_heads, config.head_dim
                        )
                        .transpose(0, 1)
                    )
                    keys = (
                        keys.cuda()
                        .double()
                        .transpose(0, 1)
                        .repeat_interleave(
                            config.num_attention_heads // config.num_key_value_heads,
                            dim=0,
                        )
                    )
                    values = (
                        values.cuda()
                        .double()
                        .transpose(0, 1)
                        .repeat_interleave(
                            config.num_attention_heads // config.num_key_value_heads,
                            dim=0,
                        )
                    )
                    scores = (queries @ keys.transpose(-1, -2)) * config.head_dim**-0.5
                    visible = torch.arange(sequence_length, device="cuda")[None, :] <= (
                        prefix_length
                        + torch.arange(extend_length, device="cuda")[:, None]
                    )
                    probabilities = scores.masked_fill(~visible, float("-inf")).softmax(
                        -1
                    )
                    expected = (
                        (probabilities @ values)
                        .transpose(0, 1)
                        .reshape(extend_length, query_width)
                    )
                    actual = snapshots[observed + "attention_projection.input"][
                        selected
                    ].cuda()
                    metrics = compare_outputs(actual, expected)
                    assert metrics["finite"].all()
                    maximum_attention_relative = max(
                        maximum_attention_relative, float(metrics["relative_l2"].max())
                    )
                    record = {
                        "layer": layer,
                        "step": metadata.step,
                        "request_id": request_id,
                        "real_rows": extend_length,
                        "prefix_rows": prefix_length,
                        "projection_metrics": projection_metrics,
                        "checkpoint_normalized_rotated_relative_l2": {
                            name: metrics["relative_l2"][selected].cpu().tolist()
                            for name, metrics in rotated_metrics.items()
                        },
                        "checkpoint_normalized_rotated_maximum_absolute": {
                            name: metrics["max_absolute"][selected].cpu().tolist()
                            for name, metrics in rotated_metrics.items()
                        },
                        "attention_relative_l2": metrics["relative_l2"].cpu().tolist(),
                        "attention_maximum_absolute": metrics["max_absolute"]
                        .cpu()
                        .tolist(),
                        "all_current_cache_writes_exact": True,
                        "projected_values_match_cache_exactly": True,
                    }
                    reports.write(json.dumps(record) + "\n")
                    checked_rows += extend_length
                    offset += extend_length
                assert offset == real_rows
            summary["layers"].append(
                {
                    "layer": layer,
                    "real_rows": checked_rows,
                    "exact_projection_rows": exact_projection_rows,
                    "maximum_attention_relative_l2": maximum_attention_relative,
                    "maximum_checkpoint_normalized_rotated_relative_l2": (
                        maximum_rotated_relative
                    ),
                }
            )
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            reports.flush()
            print(json.dumps(summary["layers"][-1]), flush=True)


if __name__ == "__main__":
    main()
