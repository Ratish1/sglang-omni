"""Decode actual published speech codes independently of the live session state."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from pydantic import BaseModel
from transformers import Qwen3OmniMoeConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import apply_wer, load_router_asr, transcribe
from scripts.qwen3_omni_checkpoint_weights import audit_checkpoint_weights
from sglang_omni.client.audio import encode_wav

CHECKPOINT_CHUNK_FRAMES = 300
SERVING_CHUNK_FRAMES = 10


class PublishedFrame(BaseModel):
    request_id: str
    committed_output_token_count: int
    codes: list[int]


class RecordedSpeech(BaseModel):
    request_id: str
    text: str
    waveform_sha256: str


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-directory", type=Path, required=True)
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--asr-port", type=int, required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    configuration = Qwen3OmniMoeConfig.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    reference, loading = Qwen3OmniMoeCode2Wav.from_pretrained(
        arguments.model_path,
        revision=arguments.model_revision,
        config=configuration.code2wav_config,
        dtype=torch.bfloat16,
        device_map={"": arguments.gpu},
        key_mapping={r"^code2wav\.": ""},
        attn_implementation="sdpa",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["mismatched_keys"]
    reference.eval()
    audit_checkpoint_weights(
        reference,
        Path(
            snapshot_download(arguments.model_path, revision=arguments.model_revision)
        ),
        arguments.output_directory / "weight-audit.json",
    )
    expected_frequencies, _ = (
        reference.pre_transformer.rotary_emb.compute_default_rope_parameters(
            reference.pre_transformer.rotary_emb.config
        )
    )
    frequencies = reference.pre_transformer.rotary_emb.inv_freq
    assert frequencies.dtype == torch.float32
    assert torch.equal(frequencies, expected_frequencies.to(frequencies.device))
    recognizer = load_router_asr(arguments.asr_port, arguments.asr_model)
    reports = []
    for arm in ("before", "after"):
        selected: dict[str, list[PublishedFrame]] = {
            request_id: [] for request_id in arguments.request_id
        }
        with gzip.open(
            arguments.control_directory / f"{arm}-frames.jsonl.gz", "rt"
        ) as source:
            for line in source:
                frame = PublishedFrame.model_validate_json(line)
                if frame.request_id in selected:
                    selected[frame.request_id].append(frame)
                else:
                    pass
        for request_id, frames in selected.items():
            assert frames
            assert [frame.committed_output_token_count for frame in frames] == list(
                range(len(frames))
            )
            all_codes = torch.tensor(
                [frame.codes for frame in frames], dtype=torch.long
            )
            assert all_codes.shape[1] == configuration.talker_config.num_code_groups
            assert all_codes[-1, 0] == configuration.talker_config.codec_eos_token_id
            codes = all_codes[:-1].T.unsqueeze(0).to(reference.device)
            assert bool(
                (
                    (codes >= 0) & (codes < configuration.code2wav_config.codebook_size)
                ).all()
            )
            sources = list(
                arguments.control_directory.glob(f"{arm}-*/{request_id}.speech.json")
            )
            assert len(sources) == 1
            speech = RecordedSpeech.model_validate_json(sources[0].read_bytes())
            assert speech.request_id == request_id
            torch.save(
                all_codes, arguments.output_directory / f"{arm}-{request_id}-codes.pt"
            )
            for chunk_frames in (CHECKPOINT_CHUNK_FRAMES, SERVING_CHUNK_FRAMES):
                waveform = reference.chunked_decode(codes, chunk_size=chunk_frames)
                assert waveform.isfinite().all()
                assert waveform.numel() == codes.shape[-1] * reference.total_upsample
                wav_bytes = encode_wav(
                    waveform.float().reshape(-1).cpu().numpy(), arguments.sample_rate
                )
                destination = (
                    arguments.output_directory
                    / f"{arm}-{request_id}-chunk{chunk_frames}.wav"
                )
                destination.write_bytes(wav_bytes)
                hypothesis = transcribe(
                    recognizer, str(destination), "en", f"cuda:{arguments.gpu}"
                )
                score = apply_wer(
                    SampleOutput(sample_id=request_id, target_text=speech.text),
                    hypothesis,
                    "en",
                )
                reports.append(
                    {
                        "arm": arm,
                        "request_id": request_id,
                        "non_eos_frames": codes.shape[-1],
                        "chunk_frames": chunk_frames,
                        "waveform_samples": waveform.numel(),
                        "sample_rate": arguments.sample_rate,
                        "original_waveform_sha256": speech.waveform_sha256,
                        "waveform_sha256": hashlib.sha256(wav_bytes).hexdigest(),
                        "score": asdict(score),
                    }
                )
                (arguments.output_directory / "summary.json").write_text(
                    json.dumps(reports, indent=2)
                )
                print(json.dumps(reports[-1]), flush=True)


if __name__ == "__main__":
    main()
else:
    pass
