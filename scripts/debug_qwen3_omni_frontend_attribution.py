"""Reconstruct captured mel features and isolate decoding from resampling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel
from transformers import Qwen3OmniMoeProcessor
from transformers.audio_utils import load_audio

from scripts.debug_qwen3_omni_feedback_audit import TensorDifference, compare_tensors
from sglang_omni.preprocessing.audio import decode_audio_bytes, resample_linear
from sglang_omni.serve.protocol import ChatCompletionRequest


class CapturedFrontend(BaseModel):
    request_id: str
    source_batch: str
    source_row: int


class EncoderReplay(BaseModel):
    frontend: list[CapturedFrontend]


@dataclass(kw_only=True)
class FrontendDifference:
    request_id: str
    source_sample_rate: int
    linear_reconstructs_capture: bool
    linear_to_checkpoint: TensorDifference
    band_to_checkpoint: TensorDifference
    band_to_librosa: TensorDifference


def mel_features(
    processor: Qwen3OmniMoeProcessor, waveform: npt.NDArray[np.float32]
) -> torch.Tensor:
    inputs = processor(
        text="<|audio_bos|><|AUDIO|><|audio_eos|>",
        audio=[waveform],
        return_tensors="pt",
    )
    return inputs["input_features"][0][:, inputs["feature_attention_mask"][0].bool()]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--replay-report", type=Path, required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--request-directory", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    replay = EncoderReplay.model_validate_json(arguments.replay_report.read_bytes())
    processor = Qwen3OmniMoeProcessor.from_pretrained(
        arguments.model_path, revision=arguments.model_revision
    )
    target_sample_rate = processor.feature_extractor.sampling_rate
    reports: list[FrontendDifference] = []
    with arguments.output_file.open("x") as output:
        for captured in replay.frontend:
            request = ChatCompletionRequest.model_validate_json(
                (
                    arguments.request_directory / f"{captured.request_id}.request.json"
                ).read_bytes()
            )
            assert request.audios and len(request.audios) == 1
            audio_path = request.audios[0]
            decoded, source_sample_rate = decode_audio_bytes(
                Path(audio_path).read_bytes()
            )
            batch = torch.load(
                arguments.capture_directory / "audio-forwards" / captured.source_batch,
                weights_only=True,
            )
            producer = torch.load(
                arguments.capture_directory
                / captured.request_id
                / "audio-conditioning.pt",
                weights_only=True,
            )
            start = int(batch["audio_output_lengths"][: captured.source_row].sum())
            end = start + int(batch["audio_output_lengths"][captured.source_row])
            assert torch.equal(
                producer["audio_embeds"], batch["audio_embeds"][start:end]
            )
            native_features = batch["input_features"][captured.source_row][
                :, batch["feature_attention_mask"][captured.source_row].bool()
            ]
            linear = mel_features(
                processor,
                resample_linear(decoded, source_sample_rate, target_sample_rate),
            )
            assert torch.equal(native_features, linear), captured.request_id
            band_limited = mel_features(
                processor,
                librosa.resample(
                    decoded, orig_sr=source_sample_rate, target_sr=target_sample_rate
                ),
            )
            checkpoint = mel_features(
                processor, load_audio(audio_path, sampling_rate=target_sample_rate)
            )
            librosa_reference = mel_features(
                processor, librosa.load(audio_path, sr=target_sample_rate)[0]
            )
            difference = FrontendDifference(
                request_id=captured.request_id,
                source_sample_rate=source_sample_rate,
                linear_reconstructs_capture=True,
                linear_to_checkpoint=compare_tensors(linear, checkpoint),
                band_to_checkpoint=compare_tensors(band_limited, checkpoint),
                band_to_librosa=compare_tensors(band_limited, librosa_reference),
            )
            reports.append(difference)
            print(json.dumps(asdict(difference)), flush=True)
        json.dump([asdict(report) for report in reports], output, indent=2)


if __name__ == "__main__":
    main()
