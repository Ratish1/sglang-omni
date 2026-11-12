"""Separate native query/key normalization, rotary tables and rotary arithmetic."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    rotate_half,
)

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.qwen3_omni_checkpoint_weights import CheckpointIndex
from scripts.qwen3_omni_thinker_operator_audit import fp32_norm
from sglang_omni.models.qwen3_omni.rotary_precision_control import (
    mrope_without_contraction,
)


def contracted_bf16_rotation(
    normalized: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    first_half, second_half = normalized.chunk(2, dim=-1)
    half_cosine = cosine[..., : first_half.shape[-1]].to(normalized.dtype)
    half_sine = sine[..., : first_half.shape[-1]].to(normalized.dtype)
    first_output = (
        first_half.double() * half_cosine.double() - (second_half * half_sine).double()
    ).to(normalized.dtype)
    second_output = (
        (second_half * half_cosine).double() + first_half.double() * half_sine.double()
    ).to(normalized.dtype)
    return torch.cat((first_output, second_output), dim=-1)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    checkpoint_directory = Path(
        snapshot_download(arguments.model_path, revision=arguments.model_revision)
    )
    config = Qwen3OmniMoeConfig.from_pretrained(
        checkpoint_directory
    ).thinker_config.text_config
    index = CheckpointIndex.model_validate_json(
        (checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    selected = []
    for path in sorted((arguments.delivery_directory / "writes").glob("*.pt")):
        captured = torch.load(path, map_location="cpu", weights_only=True)
        if captured["request_id"] == arguments.request_id:
            selected.append(captured)
        else:
            pass
    assert len(selected) == 1
    captured = selected[0]
    operators = captured["operator_batches"]
    layouts = captured["operator_layouts"]
    positions = captured["operator_positions"].to(device)
    assert positions.shape[1] == 3 and bool((positions >= 0).all())
    token_count = len(positions)
    live = slice(
        captured["batch_offset"], captured["batch_offset"] + captured["extend_length"]
    )
    rotary = Qwen3OmniMoeThinkerTextRotaryEmbedding(config).to(device)
    head_features = config.head_dim
    query_features = config.num_attention_heads * head_features
    key_features = config.num_key_value_heads * head_features
    rotary_lanes = torch.arange(head_features // 2, device=device)
    expected_axes = torch.zeros(head_features // 2, dtype=torch.long, device=device)
    for axis in (1, 2):
        expected_axes[axis : 3 * rotary.mrope_section[axis] : 3] = axis
    angles = positions[..., None].float() * rotary.inv_freq.float()
    expected_rows = torch.cat((angles.cos(), angles.sin()), dim=-1)
    comparisons = []
    layer_count = len(captured["layer_batches"]) - 1
    manifest = json.loads(
        (arguments.delivery_directory / "native-thinker-state.json").read_text()
    )
    for layer_index in range(layer_count):
        prefix = f"operator_{layer_index:02d}"
        runtime = manifest["layers"][layer_index]
        assert runtime["layer_index"] == layer_index
        assert runtime["mrope_interleaved"] and not runtime["mrope_interleaved_glm"]
        assert runtime["is_neox_style"]
        assert runtime["mrope_section"] == rotary.mrope_section
        assert (
            captured["rotary_implementations"][prefix]
            == "MRotaryEmbedding.forward_cuda"
        )
        raw = operators[f"{prefix}_qkv_output"].to(device)
        assert raw.dtype == torch.bfloat16
        raw_layout = layouts[f"{prefix}_qkv_output"].tolist()
        packed = torch.empty_strided(
            raw.shape,
            raw_layout[raw.ndim : 2 * raw.ndim],
            dtype=raw.dtype,
            device=device,
        )
        packed.copy_(raw)
        assert packed.data_ptr() % 16 == raw_layout[-1]
        query, key, _ = packed.split(
            (query_features, key_features, key_features), dim=-1
        )
        query = query.view(token_count, -1, head_features)
        key = key.view(token_count, -1, head_features)
        weights = {}
        for kind in ("q", "k"):
            name = f"thinker.model.layers.{layer_index}.self_attn.{kind}_norm.weight"
            with safe_open(
                checkpoint_directory / index.weight_map[name],
                framework="pt",
                device="cpu",
            ) as checkpoint:
                weights[kind] = checkpoint.get_tensor(name).to(device)
        expected_query = fp32_norm(query, weights["q"], config.rms_norm_eps)
        expected_key = fp32_norm(key, weights["k"], config.rms_norm_eps)
        fused_inplace_qknorm(
            query,
            key,
            weights["q"],
            weights["k"],
            config.rms_norm_eps,
            head_dim=head_features,
        )
        normalized_query = (
            operators[f"{prefix}_query_normalized"].to(device).reshape_as(query)
        )
        normalized_key = (
            operators[f"{prefix}_key_normalized"].to(device).reshape_as(key)
        )
        cache_rows = operators[f"{prefix}_rotary_cache_rows"].to(device)
        axes = operators[f"{prefix}_rotary_axes"].to(device)
        assert torch.equal(axes, expected_axes.expand(token_count, -1))
        cache_cosine = cache_rows[:, expected_axes, rotary_lanes]
        cache_sine = cache_rows[:, expected_axes, rotary_lanes + head_features // 2]
        cosine = torch.cat((cache_cosine, cache_cosine), dim=-1)[:, None].float()
        sine = torch.cat((cache_sine, cache_sine), dim=-1)[:, None].float()
        formula_query = (
            normalized_query.float() * cosine
            + rotate_half(normalized_query).float() * sine
        ).to(raw.dtype)
        formula_key = (
            normalized_key.float() * cosine + rotate_half(normalized_key).float() * sine
        ).to(raw.dtype)
        rounded_query = normalized_query * cosine.to(raw.dtype) + rotate_half(
            normalized_query
        ) * sine.to(raw.dtype)
        rounded_key = normalized_key * cosine.to(raw.dtype) + rotate_half(
            normalized_key
        ) * sine.to(raw.dtype)
        table_rows: dict[int, torch.Tensor] = {}
        for token_positions, rows in zip(
            captured["operator_positions"],
            operators[f"{prefix}_rotary_cache_rows"],
            strict=True,
        ):
            for position, row in zip(token_positions.tolist(), rows, strict=True):
                if position in table_rows:
                    assert torch.equal(row, table_rows[position])
                else:
                    table_rows[position] = row
        table = torch.zeros(
            (max(table_rows) + 1, head_features), dtype=raw.dtype, device=device
        )
        for position, row in table_rows.items():
            table[position].copy_(row)
        replay_values = []
        separate_product_values = []
        for kind in ("query", "key"):
            name = f"{prefix}_{kind}_normalized"
            values = operators[name].to(device)
            layout = layouts[name].tolist()
            for destination in (replay_values, separate_product_values):
                replay = torch.empty_strided(
                    values.shape,
                    layout[values.ndim : 2 * values.ndim],
                    dtype=values.dtype,
                    device=device,
                )
                assert replay.data_ptr() % 16 == layout[-1]
                replay.copy_(values)
                destination.append(replay)
        mrope_without_contraction(
            separate_product_values[0],
            separate_product_values[1],
            table,
            positions.T.contiguous(),
            rotary.mrope_section,
            head_features,
        )
        separate_query = separate_product_values[0].reshape_as(query)
        separate_key = separate_product_values[1].reshape_as(key)
        assert torch.equal(separate_query[live], rounded_query[live]), layer_index
        assert torch.equal(separate_key[live], rounded_key[live]), layer_index
        triton_mrope_fused(
            replay_values[0],
            replay_values[1],
            table,
            positions.T.contiguous(),
            rotary.mrope_section,
            head_features,
            head_features,
            True,
            False,
            True,
            None,
        )
        actual_query = operators[f"{prefix}_query_rotated"].to(device).reshape_as(query)
        actual_key = operators[f"{prefix}_key_rotated"].to(device).reshape_as(key)
        differences = {
            "query_norm_fp32_formula": compare_tensors(
                normalized_query[live], expected_query[live]
            ),
            "key_norm_fp32_formula": compare_tensors(
                normalized_key[live], expected_key[live]
            ),
            "query_norm_paired_kernel_replay": compare_tensors(
                normalized_query[live], query[live]
            ),
            "key_norm_paired_kernel_replay": compare_tensors(
                normalized_key[live], key[live]
            ),
            "rotary_cache_config_formula": compare_tensors(
                cache_rows[live], expected_rows.to(cache_rows.dtype)[live]
            ),
            "query_rotation_native_cache_fp32_formula": compare_tensors(
                actual_query[live], formula_query[live]
            ),
            "key_rotation_native_cache_fp32_formula": compare_tensors(
                actual_key[live], formula_key[live]
            ),
            "query_rotation_checkpoint_bf16_formula": compare_tensors(
                actual_query[live], rounded_query[live]
            ),
            "key_rotation_checkpoint_bf16_formula": compare_tensors(
                actual_key[live], rounded_key[live]
            ),
            "query_rotation_bf16_multiply_fma_formula": compare_tensors(
                actual_query[live],
                contracted_bf16_rotation(normalized_query, cosine, sine)[live],
            ),
            "key_rotation_bf16_multiply_fma_formula": compare_tensors(
                actual_key[live],
                contracted_bf16_rotation(normalized_key, cosine, sine)[live],
            ),
            "query_rotation_native_kernel_replay": compare_tensors(
                actual_query[live], replay_values[0].reshape_as(query)[live]
            ),
            "key_rotation_native_kernel_replay": compare_tensors(
                actual_key[live], replay_values[1].reshape_as(key)[live]
            ),
            "query_rotation_no_contraction_kernel": compare_tensors(
                actual_query[live], separate_query[live]
            ),
            "key_rotation_no_contraction_kernel": compare_tensors(
                actual_key[live], separate_key[live]
            ),
        }
        comparisons.append(
            {
                "layer_index": layer_index,
                "comparisons": {
                    name: asdict(difference) for name, difference in differences.items()
                },
            }
        )
    report = {
        "request_id": arguments.request_id,
        "scope": "Each operation holds actual native operands and layouts. Replaying a native kernel checks execution reproducibility, not independent mathematical correctness. Rotary axis ownership is independently derived from checkpoint configuration.",
        "layers": comparisons,
    }
    (arguments.output_directory / "comparisons.json").write_text(
        json.dumps(report, indent=2)
    )
    print(
        json.dumps(
            {"layers": layer_count, "output_directory": str(arguments.output_directory)}
        )
    )


if __name__ == "__main__":
    main()
