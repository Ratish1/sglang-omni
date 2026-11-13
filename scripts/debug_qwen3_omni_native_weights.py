"""Compare native learned-weight identities with the pinned raw checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)

from scripts.qwen3_omni_checkpoint_weights import (
    CheckpointIndex,
    audit_checkpoint_weights,
)


class NativeParameter(BaseModel):
    shape: list[int]
    dtype: str
    sha256: str


class NativeManifest(BaseModel):
    parameters: dict[str, NativeParameter]


@torch.inference_mode()
def audit_native_talker_parameters(
    reference: Qwen3OmniMoeTalkerForConditionalGeneration,
    manifest: NativeManifest,
    output_file: Path,
    *,
    model_path: str,
    model_revision: str,
    manifest_sha256: str,
) -> None:
    reference_parameters = dict(reference.named_parameters(remove_duplicate=False))
    visited: set[str] = set()
    mismatches: list[str] = []
    for native_name, actual in manifest.parameters.items():
        reference_names: tuple[str, ...]
        if native_name.endswith(".experts.w13_weight"):
            reference_names = (native_name.removesuffix("w13_weight") + "gate_up_proj",)
            gate, up = reference_parameters[reference_names[0]].chunk(2, dim=1)
            expected = torch.cat((up, gate), dim=1)
        elif native_name.endswith(".experts.w2_weight"):
            reference_names = (native_name.removesuffix("w2_weight") + "down_proj",)
            expected = reference_parameters[reference_names[0]]
        elif ".qkv_proj." in native_name:
            prefix, parameter_kind = native_name.rsplit(".qkv_proj.", 1)
            reference_names = tuple(
                f"{prefix}.{projection}.{parameter_kind}"
                for projection in ("q_proj", "k_proj", "v_proj")
            )
            expected = torch.cat(
                [reference_parameters[name] for name in reference_names], dim=0
            )
        elif ".gate_up_proj." in native_name:
            prefix, parameter_kind = native_name.rsplit(".gate_up_proj.", 1)
            reference_names = tuple(
                f"{prefix}.{projection}.{parameter_kind}"
                for projection in ("gate_proj", "up_proj")
            )
            expected = torch.cat(
                [reference_parameters[name] for name in reference_names], dim=0
            )
        else:
            reference_names = (native_name,)
            expected = reference_parameters[native_name]
        visited.update(reference_names)
        expected = expected.detach().cpu().contiguous()
        identity = NativeParameter(
            shape=list(expected.shape),
            dtype=str(expected.dtype),
            sha256=hashlib.sha256(
                expected.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        )
        if actual != identity:
            mismatches.append(native_name)
        else:
            pass
    unvisited = sorted(reference_parameters.keys() - visited)
    report = {
        "model_path": model_path,
        "model_revision": model_revision,
        "manifest_sha256": manifest_sha256,
        "native_parameters_checked": len(manifest.parameters),
        "reference_parameters_covered": len(visited),
        "unvisited_reference_parameters": unvisited,
        "mismatches": mismatches,
        "scope": "Every learned parameter of this unquantized single-GPU Talker load, including projections, norms, shared/routed experts, embeddings, heads and the residual predictor. Native fused QKV/gate-up packing and FlashInfer up-gate ordering are reconstructed without dtype conversion. Derived buffers are separate checks.",
    }
    output_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert not unvisited and not mismatches, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--component", choices=("thinker", "talker"), default="thinker")
    parser.add_argument("--gpu", type=int)
    arguments = parser.parse_args()
    manifest_bytes = arguments.manifest.read_bytes()
    manifest = NativeManifest.model_validate_json(manifest_bytes)
    checkpoint_directory = Path(
        snapshot_download(arguments.model_path, revision=arguments.model_revision)
    )
    if arguments.component == "talker":
        assert arguments.gpu is not None
        torch.cuda.set_device(arguments.gpu)
        configuration = Qwen3OmniMoeConfig.from_pretrained(
            arguments.model_path, revision=arguments.model_revision
        ).talker_config
        register_checkpoint_conversion_mapping(
            configuration.model_type,
            get_checkpoint_conversion_mapping("qwen3_omni_moe"),
        )
        reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
            arguments.model_path,
            revision=arguments.model_revision,
            config=configuration,
            dtype=torch.bfloat16,
            device_map={"": arguments.gpu},
            attn_implementation="eager",
            experts_implementation="eager",
            use_kernels=False,
            output_loading_info=True,
        )
        assert not loading["missing_keys"] and not loading["mismatched_keys"]
        reference.eval()
        audit_checkpoint_weights(
            reference,
            checkpoint_directory,
            arguments.output.with_name(arguments.output.stem + "-reference.json"),
        )
        audit_native_talker_parameters(
            reference,
            manifest,
            arguments.output,
            model_path=arguments.model_path,
            model_revision=arguments.model_revision,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        )
        return
    else:
        pass
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
