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

from scripts.debug_qwen3_omni_vocoder_replay import read_codec_sequence
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights
from sglang_omni.client.audio import encode_wav
from sglang_omni.models.qwen3_omni.components.code2wav import FusedCode2WavTransformer
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    load_code2wav_model,
)
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--trace-directory", action="append", type=Path)
    parser.add_argument("--before-inverse-frequencies", type=Path)
    parser.add_argument("--sample-rate", type=int)
    arguments = parser.parse_args()
    if arguments.trace_directory and (
        arguments.before_inverse_frequencies is None or arguments.sample_rate is None
    ):
        parser.error(
            "waveform controls require before inverse frequencies and sample rate"
        )
    else:
        pass
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
            "native_dtypes": sorted(
                {str(tensor.dtype) for tensor in native_weights.values()}
            ),
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
    if arguments.trace_directory:
        assert report["inverse_frequencies"]["native_exact_config"]
        before_frequencies = torch.load(
            arguments.before_inverse_frequencies,
            map_location=reference.device,
            weights_only=True,
        )["native_inverse_frequencies"]
        assert before_frequencies.dtype == torch.bfloat16
        assert before_frequencies.shape == expected_frequencies.shape
        before_angles = torch.outer(positions, before_frequencies.float())
        before_cache = (
            torch.cat((before_angles.cos(), before_angles.sin()), dim=-1) * scaling
        )
        corrected_cache = cache.clone()
        fused_activations = fuse_vocoder_decoder(native.decoder)
        waveform_reports = []
        report["waveform_controls"] = {
            "fused_snake_activations": fused_activations,
            "scope": "Same native model, fixed recorded codec sequences and kernel geometry. Only rotary cache values switch between actual retained pre-fix frequencies and corrected frequencies. Native decoding here is fused whole-sequence eager, not a live session-pool or throughput test.",
            "requests": waveform_reports,
        }
        for trace_directory in arguments.trace_directory:
            codes = read_codec_sequence(
                trace_directory, config.talker_config.codec_eos_token_id
            ).to(reference.device)
            expected_waveform = reference.chunked_decode(codes).float()
            native_waveforms = []
            for cache_values in (corrected_cache, before_cache, corrected_cache):
                cache.copy_(cache_values)
                native_waveforms.append(native(codes).float())
            assert torch.equal(native_waveforms[0], native_waveforms[2])
            comparisons = {}
            for mode, waveform in (
                ("corrected", native_waveforms[0]),
                ("before", native_waveforms[1]),
                ("checkpoint", expected_waveform),
            ):
                assert (
                    waveform.shape == expected_waveform.shape
                    and waveform.isfinite().all()
                )
                waveform_file = (
                    arguments.output_directory / f"{trace_directory.name}-{mode}.wav"
                )
                waveform_file.write_bytes(
                    encode_wav(
                        waveform.reshape(-1).cpu().numpy(), arguments.sample_rate
                    )
                )
                comparisons[mode] = {
                    "waveform_file": str(waveform_file),
                    "maximum_absolute_error": float(
                        (waveform - expected_waveform).abs().max()
                    ),
                    "relative_l2_error": float(
                        (waveform - expected_waveform).norm() / expected_waveform.norm()
                    ),
                }
            waveform_reports.append(
                {
                    "request_id": trace_directory.name,
                    "non_eos_frames": codes.shape[-1],
                    "waveform_samples": expected_waveform.numel(),
                    "corrected_repeat_exact": True,
                    "comparisons": comparisons,
                }
            )
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(report, indent=2)
            )
    else:
        pass
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
