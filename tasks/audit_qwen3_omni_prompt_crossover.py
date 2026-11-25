# SPDX-License-Identifier: Apache-2.0
"""Validate native prompt crossovers and score against their captured text."""

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import validate
from qwen3_omni_corpus_support import SERVER_TIMEOUT_SECONDS, server_command

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    load_qwen3_asr,
    transcribe_and_compute_wer,
)
from sglang_omni.utils import find_available_port

ARMS = (
    "stock-prompt-first",
    "stock-prompt-second",
    "candidate-prompt-second",
    "candidate-prompt-first",
)


@dataclass(kw_only=True)
class PromptResult:
    arm: str
    sample_id: str
    original_text: str
    live_text: str
    conditioning_mode: str
    conditioning_sha256: str
    capture_sha256: str
    waveform_sha256: str
    audio_duration_seconds: float
    prefill_batch_tokens: int
    prefill_batch_requests: int
    changed_prompt_values: int
    opening_matches: dict[str, bool]
    frames: int
    first_changed_codec_frame: int | None
    score: SampleOutput | None = None


def audit(root: Path, native: Path) -> list[PromptResult]:
    torch.set_num_threads(1)
    results: list[PromptResult] = []
    for arm in ARMS:
        directory = root / arm
        references = json.loads(
            (directory / "conditioning-references.json").read_text()
        )
        assert len(references) == 1
        sample_id, reference_path = next(iter(references.items()))
        reference = Path(reference_path)
        fixture = torch.load(
            reference / "conditioning.pt", weights_only=True, map_location="cpu"
        )
        receipt = json.loads(
            (directory / "captures" / f"{sample_id}-conditioning.json").read_text()
        )
        assert all(receipt["checks"].values())
        assert (
            receipt["conditioning_sha256"]
            == sha256((reference / "conditioning.pt").read_bytes()).hexdigest()
        )
        assert (
            receipt["conditioning_receipt_sha256"]
            == sha256((reference / "conditioning.json").read_bytes()).hexdigest()
        )
        frames = validate(directory / "captures" / sample_id)
        expected_mode = (
            "hold_prompt_verify_opening"
            if arm in ("stock-prompt-first", "candidate-prompt-second")
            else "hold_prompt"
        )
        assert receipt["conditioning_mode"] == expected_mode
        for actual_name, expected in (
            ("stored_keys", fixture["prefill_keys"]),
            ("stored_values", fixture["prefill_values"]),
            ("codec_ids", fixture["codec_ids"][0]),
            ("feedback", fixture["first_feedback"]),
        ):
            matches = torch.equal(frames[0][actual_name], expected)
            assert receipt["opening_matches"][actual_name] == matches
            if expected_mode == "hold_prompt_verify_opening":
                assert matches
            else:
                pass
        metadata = json.loads(
            (directory / "captures" / sample_id / "trajectory.json").read_text()
        )
        original_texts = [
            json.loads((native / source / "seedtts/returned-text.json").read_text())[
                sample_id
            ]["text"]
            for source in ("first", "second")
        ]
        assert original_texts[0] == original_texts[1]
        assert torch.equal(
            frames[0]["prefill_embeddings"], fixture["prefill_embeddings"]
        )
        assert torch.equal(frames[0]["input_ids"], fixture["input_ids"])
        for frame_index, frame in enumerate(frames[1:]):
            expected = (
                fixture["text_embeddings"][frame_index]
                if frame_index < len(fixture["text_embeddings"])
                else fixture["padding_embedding"]
            )
            assert torch.equal(
                frame["text_embedding"].reshape(-1), expected.reshape(-1)
            )
        codes = torch.stack([frame["codec_ids"] for frame in frames])
        first_changed = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(codes, fixture["codec_ids"])
                )
                if not torch.equal(actual, expected)
            ),
            None,
        )
        if first_changed is None and len(codes) != len(fixture["codec_ids"]):
            first_changed = min(len(codes), len(fixture["codec_ids"]))
        else:
            pass
        generated = next(
            row
            for row in json.loads((directory / "seedtts/generated.json").read_text())
            if row["sample_id"] == sample_id
        )
        assert generated["is_success"]
        live_text = json.loads((directory / "seedtts/returned-text.json").read_text())[
            sample_id
        ]["text"]
        waveform = directory / "seedtts/audio" / f"{sample_id}.wav"
        results.append(
            PromptResult(
                arm=arm,
                sample_id=sample_id,
                original_text=original_texts[0],
                live_text=live_text,
                conditioning_mode=receipt["conditioning_mode"],
                conditioning_sha256=receipt["conditioning_sha256"],
                capture_sha256=sha256(
                    (directory / "captures" / sample_id / "trajectory.pt").read_bytes()
                ).hexdigest(),
                waveform_sha256=sha256(waveform.read_bytes()).hexdigest(),
                audio_duration_seconds=generated["audio_duration_s"],
                prefill_batch_tokens=metadata["frames"][0]["batch_tokens"],
                prefill_batch_requests=len(metadata["frames"][0]["request_ids"]),
                changed_prompt_values=int(
                    (
                        frames[0]["original_prefill_embeddings"]
                        != frames[0]["prefill_embeddings"]
                    ).sum()
                ),
                opening_matches=receipt["opening_matches"],
                frames=len(frames),
                first_changed_codec_frame=first_changed,
            )
        )
    return results


def main(root: Path, native: Path, asr_checkpoint: Path) -> None:
    results = audit(root, native)
    output = root / "prompt-crossover.json"
    output.write_text(json.dumps([asdict(result) for result in results], indent=2))
    port = find_available_port()
    server = start_server_from_cmd(
        [
            *server_command(),
            "--port",
            str(port),
            "--model-path",
            str(asr_checkpoint),
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.gpu",
            "0",
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        root / "asr.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    try:
        asr = load_qwen3_asr(port)
        for result in results:
            waveform = root / result.arm / "seedtts/audio" / f"{result.sample_id}.wav"
            assert sha256(waveform.read_bytes()).hexdigest() == result.waveform_sha256
            result.score = transcribe_and_compute_wer(
                SampleOutput(
                    sample_id=result.sample_id,
                    target_text=result.original_text,
                    audio_duration_s=result.audio_duration_seconds,
                ),
                str(waveform),
                asr,
                "en",
                "cuda:0",
            )
            assert result.score.is_success and result.score.is_transcribed
            output.write_text(json.dumps([asdict(row) for row in results], indent=2))
            print(json.dumps({"arm": result.arm, "wer": result.score.wer}), flush=True)
    finally:
        stop_server(server)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("asr_checkpoint", type=Path)
    arguments = parser.parse_args()
    main(arguments.root, arguments.native, arguments.asr_checkpoint)
else:
    pass
