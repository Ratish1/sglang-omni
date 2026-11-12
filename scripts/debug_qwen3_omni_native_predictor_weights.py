"""Compare native-loaded predictor weights against the pinned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from safetensors import safe_open


class WeightFingerprint(BaseModel):
    shape: list[int]
    dtype: str
    sha256: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text())
    checkpoint_directory = Path(
        hf_hub_download(
            arguments.model_path,
            "config.json",
            revision=arguments.model_revision,
            local_files_only=True,
        )
    ).parent
    weight_map = json.loads(
        (checkpoint_directory / "model.safetensors.index.json").read_text()
    )["weight_map"]
    fingerprints = {
        name: WeightFingerprint.model_validate(record)
        for name, record in manifest["weights"].items()
    }
    assert fingerprints
    reports = []
    for name, fingerprint in fingerprints.items():
        if ".qkv_proj." in name:
            source_names = [
                name.replace(".qkv_proj.", f".{projection}.")
                for projection in ("q_proj", "k_proj", "v_proj")
            ]
        elif ".gate_up_proj." in name:
            source_names = [
                name.replace(".gate_up_proj.", f".{projection}.")
                for projection in ("gate_proj", "up_proj")
            ]
        else:
            source_names = [name]
        tensors = []
        for source_name in source_names:
            checkpoint_name = f"talker.{source_name}"
            with safe_open(
                checkpoint_directory / weight_map[checkpoint_name],
                framework="pt",
                device="cpu",
            ) as shard:
                tensors.append(shard.get_tensor(checkpoint_name))
        expected = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
        assert list(expected.shape) == fingerprint.shape, name
        assert str(expected.dtype) == fingerprint.dtype == "torch.bfloat16", name
        expected_hash = hashlib.sha256(
            expected.contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        assert expected_hash == fingerprint.sha256, name
        reports.append({"name": name, "checkpoint_names": source_names, "exact": True})
    configuration = json.loads((checkpoint_directory / "config.json").read_text())[
        "talker_config"
    ]
    groups = configuration["num_code_groups"]
    assert groups == 16
    expected_names = {
        "model.codec_embedding.weight",
        "codec_head.weight",
        "code_predictor.model.norm.weight",
    }
    expected_names.update(
        f"code_predictor.{module}.{group}.weight"
        for group in range(groups - 1)
        for module in ("model.codec_embedding", "lm_head")
    )
    expected_names.update(
        f"code_predictor.model.layers.{layer}.{module}.weight"
        for layer in range(configuration["code_predictor_config"]["num_hidden_layers"])
        for module in (
            "self_attn.qkv_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "input_layernorm",
            "post_attention_layernorm",
            "mlp.gate_up_proj",
            "mlp.down_proj",
        )
    )
    assert fingerprints.keys() == expected_names
    report = {"model_revision": arguments.model_revision, "weights": reports}
    with arguments.output.open("x") as output:
        json.dump(report, output, indent=2)
    print(json.dumps({"exact_checkpoint_weight_tensors": len(reports)}))


if __name__ == "__main__":
    main()
else:
    pass
