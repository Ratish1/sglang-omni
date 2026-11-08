"""Decode recorded complete codec sequences with the checkpoint vocoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.client.audio import encode_wav


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--codes-file", type=Path, action="append", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
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
        for filename in arguments.codes_file:
            codes = torch.load(
                filename, map_location=reference.device, weights_only=True
            )
            assert (
                codes.ndim == 2
                and codes.shape[1] == config.talker_config.num_code_groups
            )
            assert int(codes[-1, 0]) == config.talker_config.codec_eos_token_id
            codes = codes[:-1]
            assert (
                (codes >= 0)
                & (codes < config.talker_config.code_predictor_config.vocab_size)
            ).all()
            waveform = reference(codes.T[None]).float().flatten().cpu().numpy()
            assert torch.isfinite(torch.from_numpy(waveform)).all()
            destination = (
                arguments.output_directory
                / f"{filename.parent.name}-{filename.stem}.wav"
            )
            assert not destination.exists()
            waveform_bytes = encode_wav(waveform, arguments.sample_rate)
            destination.write_bytes(waveform_bytes)
            report = {
                "codes_file": str(filename),
                "non_eos_frames": len(codes),
                "audio_duration_seconds": len(waveform) / arguments.sample_rate,
                "waveform_file": str(destination),
                "waveform_sha256": hashlib.sha256(waveform_bytes).hexdigest(),
            }
            reports.append(report)
            print(json.dumps(report), flush=True)
    (arguments.output_directory / "summary.json").write_text(
        json.dumps(reports, indent=2)
    )


if __name__ == "__main__":
    main()
