"""Replay the installed paired QK kernel on retained native packed projection rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from sglang.kernels.ops.layernorm.norm import (
    _jit_qknorm_module as load_query_key_norm_module,
)
from sglang.kernels.ops.layernorm.norm import fused_inplace_qknorm
from transformers import Qwen3OmniMoeConfig

from scripts.qwen3_omni_checkpoint_weights import CheckpointIndex


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    ).talker_config.text_config
    checkpoint_directory = Path(
        hf_hub_download(
            arguments.model_path, "config.json", revision=arguments.model_revision
        )
    ).parent
    index = CheckpointIndex.model_validate_json(
        (checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    weights: list[dict[str, torch.Tensor]] = []
    fingerprints: dict[str, str] = {}
    for layer in range(configuration.num_hidden_layers):
        layer_weights = {}
        for projection in ("q", "k"):
            name = f"talker.model.layers.{layer}.self_attn.{projection}_norm.weight"
            with safe_open(
                checkpoint_directory / index.weight_map[name],
                framework="pt",
                device="cpu",
            ) as shard:
                weight = shard.get_tensor(name)
            assert (
                weight.shape == (configuration.head_dim,)
                and weight.dtype == torch.bfloat16
            )
            fingerprints[name] = hashlib.sha256(
                weight.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            layer_weights[projection] = weight.to(device)
        weights.append(layer_weights)
    (arguments.output_directory / "weights.json").write_text(
        json.dumps(fingerprints, indent=2)
    )
    query_dimensions = configuration.num_attention_heads * configuration.head_dim
    key_dimensions = configuration.num_key_value_heads * configuration.head_dim
    warp_kernel = load_query_key_norm_module(
        configuration.head_dim, torch.bfloat16, half_warp=False
    )
    summaries = []
    for directory in sorted(arguments.capture_directory.iterdir()):
        destination = arguments.output_directory / directory.name
        destination.mkdir()
        rows = []
        finished = torch.load(directory / "finished.pt", weights_only=True)
        capture_files = sorted(directory.glob("*-operators.pt"))
        assert [filename.name for filename in capture_files] == [
            f"{frame:04d}-operators.pt"
            for frame in range(1, len(finished["output_ids"]))
        ]
        for filename in capture_files:
            frame = int(filename.name.split("-")[0])
            snapshots = torch.load(filename, weights_only=True)
            assert len(snapshots) == len(weights)
            for layer, (snapshot, layer_weights) in enumerate(
                zip(snapshots, weights, strict=True)
            ):
                row = {"frame": frame, "layer": layer}
                normalized: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                for variant in ("paired", "warp"):
                    packed = snapshot["qkv"].to(device).unsqueeze(0).contiguous()
                    assert packed.shape == (1, query_dimensions + 2 * key_dimensions)
                    assert packed.dtype == torch.bfloat16
                    query = packed.as_strided(
                        (1, configuration.num_attention_heads, configuration.head_dim),
                        (packed.stride(0), configuration.head_dim, 1),
                    )
                    key = packed.as_strided(
                        (1, configuration.num_key_value_heads, configuration.head_dim),
                        (packed.stride(0), configuration.head_dim, 1),
                        storage_offset=query_dimensions,
                    )
                    alignment = [
                        tensor.data_ptr() % 16
                        for tensor in (
                            query,
                            key,
                            layer_weights["q"],
                            layer_weights["k"],
                        )
                    ]
                    assert not any(alignment)
                    assert (packed.stride(0) * packed.element_size()) % 16 == 0
                    if variant == "paired":
                        fused_inplace_qknorm(
                            q=query,
                            k=key,
                            q_weight=layer_weights["q"],
                            k_weight=layer_weights["k"],
                            eps=configuration.rms_norm_eps,
                            head_dim=configuration.head_dim,
                        )
                    else:
                        warp_kernel.qknorm(
                            query,
                            key,
                            layer_weights["q"],
                            layer_weights["k"],
                            configuration.rms_norm_eps,
                        )
                    normalized[variant] = (query.clone(), key.clone())
                    expected_query = (
                        snapshot["normalized_query"].to(device).reshape_as(query)
                    )
                    expected_key = snapshot["normalized_key"].to(device).reshape_as(key)
                    assert expected_query.dtype == expected_key.dtype == packed.dtype
                    row[variant] = {
                        "packed_token_stride": packed.stride(0),
                        "query_stride": list(query.stride()),
                        "key_stride": list(key.stride()),
                        "pointer_alignment_modulo_16": alignment,
                        "query_exact": torch.equal(query, expected_query),
                        "key_exact": torch.equal(key, expected_key),
                        "query_maximum_error": float(
                            (query.float() - expected_query.float()).abs().max()
                        ),
                        "key_maximum_error": float(
                            (key.float() - expected_key.float()).abs().max()
                        ),
                    }
                row["paired_warp_exact"] = all(
                    torch.equal(paired, warp)
                    for paired, warp in zip(
                        normalized["paired"], normalized["warp"], strict=True
                    )
                )
                rows.append(row)
                with (destination / "layers.jsonl").open("a") as output:
                    output.write(json.dumps(row) + "\n")
        assert rows
        summary = {
            "request_id": directory.name,
            "layer_rows": len(rows),
            "paired_warp_exact_rows": sum(row["paired_warp_exact"] for row in rows),
            "variants": {
                variant: {
                    "query_exact_rows": sum(
                        row[variant]["query_exact"] for row in rows
                    ),
                    "key_exact_rows": sum(row[variant]["key_exact"] for row in rows),
                    "maximum_query_error": max(
                        row[variant]["query_maximum_error"] for row in rows
                    ),
                    "maximum_key_error": max(
                        row[variant]["key_maximum_error"] for row in rows
                    ),
                }
                for variant in ("paired", "warp")
            },
            "scope": "Actual packed projection row replayed through the installed paired QK kernel. Preserved packed token stride and recorded alignment, not an asserted full-batch replay or a new serving backend.",
        }
        summaries.append(summary)
        (arguments.output_directory / "summary.json").write_text(
            json.dumps(summaries, indent=2)
        )
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
