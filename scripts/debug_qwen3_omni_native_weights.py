"""Compare native learned-weight identities with the pinned raw checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field
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


class NativeRotaryModule(BaseModel):
    class_name: Literal["RotaryEmbedding", "MRotaryEmbedding"] = Field(alias="class")
    base: float
    rotary_dim: int
    head_size: int
    position_start: int
    is_neox_style: bool
    use_fallback_kernel: bool


class NativeBufferRuntime(BaseModel):
    predictor_fused_layers: bool
    predictor_triton_attention: bool
    predictor_exact_add_norm: bool
    predictor_rope_stores_kv: bool
    rotary_modules: dict[str, NativeRotaryModule]


@torch.inference_mode()
def audit_native_talker_buffers(
    reference: Qwen3OmniMoeTalkerForConditionalGeneration,
    directory: Path,
    output_file: Path,
) -> None:
    runtime_bytes = (directory / "native-talker-buffer-runtime.json").read_bytes()
    runtime = NativeBufferRuntime.model_validate_json(runtime_bytes)
    buffer_file = directory / "native-talker-buffers.pt"
    buffers = torch.load(buffer_file, weights_only=True)
    visited: set[str] = set()
    reports = []
    for name, module in runtime.rotary_modules.items():
        if name.startswith("code_predictor."):
            rotary = reference.code_predictor.model.rotary_emb
        else:
            assert name.startswith("model.")
            rotary = reference.model.rotary_emb
        configuration = rotary.config
        parameters = configuration.rope_parameters
        assert parameters["rope_type"] == "default"
        assert module.base == parameters["rope_theta"]
        assert module.rotary_dim == module.head_size == configuration.head_dim
        assert module.is_neox_style and module.position_start == 0
        buffer_name = f"{name}.cos_sin_cache"
        actual = buffers[buffer_name].to(reference.device)
        assert actual.isfinite().all()
        assert actual.ndim == 2 and actual.shape[1] == module.rotary_dim
        expected_tables = []
        for device in (torch.device("cpu"), reference.device):
            inverse_frequencies = 1.0 / (
                module.base
                ** (
                    torch.arange(
                        0, module.rotary_dim, 2, dtype=torch.float32, device=device
                    )
                    / module.rotary_dim
                )
            )
            positions = torch.arange(len(actual), dtype=torch.float32, device=device)
            angles = torch.outer(positions, inverse_frequencies)
            expected_tables.append(
                torch.cat((angles.cos(), angles.sin()), dim=-1).to(
                    device=reference.device, dtype=actual.dtype
                )
            )
        exact_cpu = torch.equal(actual, expected_tables[0])
        exact_gpu = torch.equal(actual, expected_tables[1])
        initial_rows = configuration.max_position_embeddings
        assert len(actual) >= initial_rows
        inverse_frequencies, scaling = rotary.compute_default_rope_parameters(
            configuration
        )
        assert scaling == 1.0
        extension_positions = torch.arange(
            initial_rows, len(actual), dtype=torch.float32, device=reference.device
        )
        extension_angles = torch.outer(
            extension_positions, inverse_frequencies.to(reference.device)
        )
        source_table = torch.cat(
            (
                expected_tables[1][:initial_rows],
                torch.cat((extension_angles.cos(), extension_angles.sin()), dim=-1).to(
                    actual.dtype
                ),
            ),
            dim=0,
        )
        source_cache_growth_exact = torch.equal(actual, source_table)
        visited.add(buffer_name)
        if module.class_name == "MRotaryEmbedding":
            section = parameters["mrope_section"]
            axes = torch.zeros(module.rotary_dim // 2, dtype=torch.long)
            if parameters.get("interleaved", False):
                for axis in (1, 2):
                    axes[axis : section[axis] * 3 : 3] = axis
            else:
                axes = torch.repeat_interleave(torch.arange(3), torch.tensor(section))
            axis_name = f"{name}.axis_map"
            assert torch.equal(buffers[axis_name], axes)
            visited.add(axis_name)
        else:
            assert actual.dtype == torch.float32
        reports.append(
            {
                "module": name,
                "class": module.class_name,
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "checkpoint_geometry_exact": True,
                "cpu_table_exact": exact_cpu,
                "gpu_table_exact": exact_gpu,
                "initial_cache_rows": initial_rows,
                "source_cache_growth_exact": source_cache_growth_exact,
                "maximum_gpu_table_error": float(
                    (actual - expected_tables[1]).abs().max()
                ),
            }
        )
    unvisited = sorted(buffers.keys() - visited)
    with buffer_file.open("rb") as source:
        buffers_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    report = {
        "runtime_sha256": hashlib.sha256(runtime_bytes).hexdigest(),
        "buffers_sha256": buffers_sha256,
        "runtime": runtime.model_dump(by_alias=True),
        "rotary_modules": reports,
        "unvisited_buffers": unvisited,
        "scope": "Actual registered buffers after native prefill and residual prediction. Checkpoint theta, dimensions and axes are checked independently. Initial tables are constructed inside the loader's GPU device context. Pre-capture cache growth computes inverse frequencies in the default CPU context then moves them to the cache device, appending rows after the checkpoint-derived original limit. Both phases are reconstructed in the actual consumer dtype. Whole-table CPU/GPU comparisons remain reported, not silently treated as equivalent. This is not a speech-quality or every-kernel oracle.",
    }
    output_file.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    assert reports and not unvisited
    assert all(row["source_cache_growth_exact"] for row in reports)


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
    parser.add_argument("--buffer-directory", type=Path)
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
        if arguments.buffer_directory is not None:
            audit_native_talker_buffers(
                reference,
                arguments.buffer_directory,
                arguments.output.with_name(arguments.output.stem + "-buffers.json"),
            )
        else:
            pass
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
