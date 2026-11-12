"""Compare unpacked native Thinker identities with the pinned raw checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from safetensors import safe_open

from scripts.qwen3_omni_checkpoint_weights import CheckpointIndex


class NativeParameter(BaseModel):
    shape: list[int]
    dtype: str
    sha256: str


class NativeManifest(BaseModel):
    parameters: dict[str, NativeParameter]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = NativeManifest.model_validate_json(arguments.manifest.read_bytes())
    checkpoint_directory = Path(
        snapshot_download(arguments.model_path, revision=arguments.model_revision)
    )
    index = CheckpointIndex.model_validate_json(
        (checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    expected_names = {
        name
        for name in index.weight_map
        if name.startswith(("thinker.model.", "thinker.lm_head."))
    }
    missing = sorted(expected_names - manifest.parameters.keys())
    unexpected = sorted(manifest.parameters.keys() - expected_names)
    shards: dict[str, list[str]] = {}
    for name in sorted(expected_names & manifest.parameters.keys()):
        shards.setdefault(index.weight_map[name], []).append(name)
    mismatches = []
    checked = 0
    for shard, names in sorted(shards.items()):
        with safe_open(
            checkpoint_directory / shard, framework="pt", device="cpu"
        ) as checkpoint:
            for name in names:
                expected = checkpoint.get_tensor(name)
                actual = manifest.parameters[name]
                digest = hashlib.sha256(
                    expected.contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                if (
                    list(expected.shape) != actual.shape
                    or str(expected.dtype) != actual.dtype
                    or digest != actual.sha256
                ):
                    mismatches.append(name)
                else:
                    pass
                checked += 1
    report = {
        "model_path": arguments.model_path,
        "model_revision": arguments.model_revision,
        "checked_checkpoint_tensors": checked,
        "missing": missing,
        "unexpected": unexpected,
        "mismatches": mismatches,
        "scope": "All native Thinker text-backbone and LM-head learned weights, unpacked after loading. Multimodal encoders and derived rotary buffers are separate checks.",
    }
    arguments.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert not missing and not unexpected and not mismatches, report


if __name__ == "__main__":
    main()
