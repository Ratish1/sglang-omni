"""Audit real native vocoder weights and nonpersistent rotary constants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights
from sglang_omni.models.qwen3_omni.components.code2wav import FusedCode2WavTransformer
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    load_code2wav_model,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    checkpoint_directory = Path(
        snapshot_download(
            arguments.model_path,
            revision=arguments.model_revision,
            local_files_only=True,
        )
    )
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    native = load_code2wav_model(
        str(checkpoint_directory), device=f"cuda:{arguments.gpu}", dtype="bfloat16"
    )
    assert isinstance(native.pre_transformer, FusedCode2WavTransformer)
    reference, loading = Qwen3OmniMoeCode2Wav.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=config.code2wav_config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        key_mapping={r"^code2wav\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"]
    reference.eval()
    native_weights = {
        name.replace("pre_transformer.transformer.", "pre_transformer.", 1): tensor
        for name, tensor in native.state_dict().items()
    }
    reference_weights = reference.state_dict()
    missing_names = sorted(reference_weights.keys() - native_weights.keys())
    unexpected_names = sorted(native_weights.keys() - reference_weights.keys())
    mismatches = [
        name
        for name in reference_weights.keys() & native_weights.keys()
        if not torch.equal(reference_weights[name], native_weights[name])
    ]
    native_rotary = native.pre_transformer.transformer.rotary_emb
    reference_rotary = reference.pre_transformer.rotary_emb
    expected_frequencies, scaling = reference_rotary.compute_default_rope_parameters(
        reference_rotary.config
    )
    expected_frequencies = expected_frequencies.to(reference.device)
    native_frequencies = native_rotary.inv_freq.float()
    cache = native.pre_transformer.cos_sin_cache
    positions = torch.arange(len(cache), device=reference.device, dtype=torch.float32)
    angles = torch.outer(positions, expected_frequencies)
    expected_cache = torch.cat((angles.cos(), angles.sin()), dim=-1) * scaling
    report = {
        "model_revision": arguments.model_revision,
        "weights": {
            "reference_tensors": len(reference_weights),
            "native_tensors": len(native_weights),
            "missing": missing_names,
            "unexpected": unexpected_names,
            "mismatches": mismatches,
        },
        "inverse_frequencies": {
            "native_dtype": str(native_rotary.inv_freq.dtype),
            "reference_dtype": str(reference_rotary.inv_freq.dtype),
            "native_exact_config": torch.equal(
                native_frequencies, expected_frequencies
            ),
            "reference_exact_config": torch.equal(
                reference_rotary.inv_freq, expected_frequencies
            ),
            "different_values": int((native_frequencies != expected_frequencies).sum()),
            "maximum_absolute_error": float(
                (native_frequencies - expected_frequencies).abs().max()
            ),
        },
        "native_cache": {
            "positions": len(cache),
            "dtype": str(cache.dtype),
            "exact_config": torch.equal(cache, expected_cache),
            "maximum_absolute_error": float((cache - expected_cache).abs().max()),
        },
        "scope": "Actual native production loader and checkpoint reference on the same real BF16 model. The full cache comparison includes positions not necessarily reached by a selected request and is not a speech-quality or timing measurement.",
    }
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(report, indent=2)
    )
    torch.save(
        {
            "native_inverse_frequencies": native_rotary.inv_freq.cpu(),
            "reference_inverse_frequencies": reference_rotary.inv_freq.cpu(),
            "expected_inverse_frequencies": expected_frequencies.cpu(),
        },
        arguments.output_directory / "inverse-frequencies.pt",
    )
    audit_checkpoint_weights(
        reference,
        checkpoint_directory,
        arguments.output_directory / "reference-weight-audit.json",
    )
    assert not missing_names and not unexpected_names and not mismatches, report[
        "weights"
    ]
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
