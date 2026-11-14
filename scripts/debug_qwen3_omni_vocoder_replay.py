# SPDX-License-Identifier: Apache-2.0
"""Re-decode fixed codec sequences with the checkpoint vocoder and score their speech."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    apply_wer,
    load_router_asr,
    transcribe,
)
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights
from sglang_omni.client.audio import encode_wav


def read_codec_sequence(trace_directory: Path, eos_token_id: int) -> torch.Tensor:
    trace_files = sorted(trace_directory.glob("*-post_*.pt"))
    assert trace_files
    frames = []
    for trace_file in trace_files:
        captured = torch.load(trace_file, map_location="cpu", weights_only=True)
        codes = captured["codec_ids"]
        if int(codes[0, 0]) == eos_token_id:
            assert trace_file == trace_files[-1]
        else:
            frames.append(codes)
    return torch.cat(frames, dim=0).T.unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--trace-directory", action="append", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--asr-port", type=int, required=True)
    parser.add_argument("--asr-model", default=QWEN3_ASR_MODEL_PATH)
    target_arguments = parser.add_mutually_exclusive_group(required=True)
    target_arguments.add_argument("--target-text")
    target_arguments.add_argument("--targets-file", type=Path)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    if arguments.dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        pass
    config = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    reference, loading = Qwen3OmniMoeCode2Wav.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=config.code2wav_config,
        dtype={"bfloat16": torch.bfloat16, "float32": torch.float32}[arguments.dtype],
        device_map={"": arguments.gpu},
        key_mapping={r"^code2wav\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    audit_checkpoint_weights(
        reference,
        Path(
            snapshot_download(
                arguments.model_path,
                revision=arguments.model_revision,
                local_files_only=True,
            )
        ),
        arguments.output_directory / "loaded-weight-audit.json",
    )
    target_text_by_request = (
        json.loads(arguments.targets_file.read_text())
        if arguments.targets_file is not None
        else {}
    )
    reports = []
    with torch.inference_mode():
        for trace_directory in arguments.trace_directory:
            codes = read_codec_sequence(
                trace_directory, config.talker_config.codec_eos_token_id
            ).to(reference.device)
            waveform = reference(codes).float().reshape(-1).cpu().numpy()
            assert torch.isfinite(torch.from_numpy(waveform)).all()
            wav_bytes = encode_wav(waveform, arguments.sample_rate)
            trace_digest = hashlib.sha256(
                str(trace_directory.resolve()).encode()
            ).hexdigest()
            waveform_filename = f"{trace_directory.parent.name}-{trace_digest}.wav"
            if arguments.targets_file is None:
                destination = arguments.output_directory / waveform_filename
                target_text = arguments.target_text
            else:
                case_directory = (
                    arguments.output_directory
                    / trace_directory.name
                    / trace_directory.parent.parent.name
                )
                case_directory.mkdir(parents=True, exist_ok=True)
                destination = case_directory / waveform_filename
                target_text = target_text_by_request[trace_directory.name]
            destination.write_bytes(wav_bytes)
            hypothesis = transcribe(
                load_router_asr(arguments.asr_port, arguments.asr_model),
                str(destination),
                "en",
                f"cuda:{arguments.gpu}",
            )
            score = apply_wer(
                SampleOutput(
                    sample_id=trace_directory.parent.name,
                    target_text=target_text,
                    audio_duration_s=len(waveform) / arguments.sample_rate,
                ),
                hypothesis,
                "en",
            )
            report = {
                "trace_directory": str(trace_directory),
                "reference_dtype": arguments.dtype,
                "allow_tf32_convolutions": torch.backends.cudnn.allow_tf32,
                "non_eos_frames": codes.shape[-1],
                "waveform_samples": len(waveform),
                "sample_rate": arguments.sample_rate,
                "waveform_file": str(destination),
                "waveform_sha256": hashlib.sha256(wav_bytes).hexdigest(),
                "score": asdict(score),
            }
            reports.append(report)
            (arguments.output_directory / "summary.json").write_text(
                json.dumps(reports, indent=2)
            )
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
