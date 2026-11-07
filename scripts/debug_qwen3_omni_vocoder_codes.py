"""Decode saved complete codec sequences with the unmodified checkpoint vocoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile
import torch
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.client.audio import encode_wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--codes-directory", type=Path, required=True)
    parser.add_argument("--native-waveform-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path)
    reference, loading = Qwen3OmniMoeCode2Wav.from_pretrained(
        arguments.model_path,
        config=config.code2wav_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        key_mapping={r"^code2wav\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    reports = []
    with torch.inference_mode():
        for request_id in arguments.request_id:
            codes = torch.load(
                arguments.codes_directory / f"{request_id}.pt",
                map_location="cpu",
                weights_only=True,
            )
            assert codes[-1, 0] == config.talker_config.codec_eos_token_id
            assert (codes[:-1, 0] != config.talker_config.codec_eos_token_id).all()
            assert codes.shape[1] == config.code2wav_config.num_quantizers
            waveform = (
                reference(codes[:-1].T.contiguous().unsqueeze(0).to(reference.device))
                .float()
                .reshape(-1)
                .cpu()
            )
            native_samples, sample_rate = soundfile.read(
                arguments.native_waveform_directory / f"{request_id}.wav",
                dtype="float32",
            )
            native = torch.from_numpy(native_samples)
            assert waveform.shape == native.shape
            assert torch.isfinite(waveform).all() and torch.isfinite(native).all()
            waveform_bytes = encode_wav(waveform.numpy(), sample_rate)
            (arguments.output_directory / f"{request_id}.wav").write_bytes(
                waveform_bytes
            )
            difference = waveform - native
            response = json.loads(
                (
                    arguments.native_waveform_directory / f"{request_id}.response.json"
                ).read_text()
            )
            report = {
                "request_id": request_id,
                "non_eos_frames": codes.shape[0] - 1,
                "sample_rate": sample_rate,
                "waveform_samples": waveform.numel(),
                "native_waveform": str(
                    arguments.native_waveform_directory / f"{request_id}.wav"
                ),
                "reference_waveform": str(
                    arguments.output_directory / f"{request_id}.wav"
                ),
                "target_text": response["choices"][0]["message"]["content"],
                "maximum_absolute_difference": difference.abs().max().item(),
                "relative_l2": (difference.norm() / waveform.norm()).item(),
                "cosine_similarity": torch.nn.functional.cosine_similarity(
                    waveform, native, dim=0
                ).item(),
                "waveform_sha256": hashlib.sha256(waveform_bytes).hexdigest(),
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(reports, indent=2)
    )


if __name__ == "__main__":
    main()
