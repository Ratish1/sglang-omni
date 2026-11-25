# SPDX-License-Identifier: Apache-2.0
"""Validate carried prefixes and score their native speech continuations."""

import argparse
import json
from dataclasses import asdict
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


def main(root: Path, control: Path, donor: Path, asr_checkpoint: Path) -> None:
    torch.set_num_threads(1)
    original = validate(control)
    sample_id = control.name
    assert donor.name == sample_id
    target_text = json.loads(
        (control.parent.parent / "seedtts/returned-text.json").read_text()
    )[sample_id]["text"]
    results = []
    for arm, prefix_directory in (("first", control), ("second", donor)):
        directory = root / arm
        captures = directory / "captures"
        frames = validate(captures / sample_id)
        prefix = validate(prefix_directory)[0]
        receipt = json.loads((captures / "prefix-trial.json").read_text())
        conditioning = json.loads(
            (captures / f"{sample_id}-conditioning.json").read_text()
        )
        assert all(conditioning["checks"].values())
        assert conditioning["conditioning_mode"] == "hold_prompt_verify_opening"
        assert receipt["capture_sha256"] == [
            sha256((path / "trajectory.pt").read_bytes()).hexdigest()
            for path in (control, prefix_directory)
        ]
        for field in (
            "prefill_embeddings",
            "stored_keys",
            "stored_values",
            "codec_ids",
            "feedback",
        ):
            assert torch.equal(frames[0][field], original[0][field]), field
        for field in ("keys", "values"):
            assert torch.equal(
                frames[0][f"decode_prefix_{field}"], prefix[f"stored_{field}"]
            )
        references = json.loads(
            (directory / "conditioning-references.json").read_text()
        )
        fixture = torch.load(
            Path(references[sample_id]) / "conditioning.pt",
            map_location="cpu",
            weights_only=True,
        )
        for frame_index, frame in enumerate(frames[1:]):
            expected = (
                fixture["text_embeddings"][frame_index]
                if frame_index < len(fixture["text_embeddings"])
                else fixture["padding_embedding"]
            )
            assert torch.equal(
                frame["text_embedding"].reshape(-1), expected.reshape(-1)
            )
        trial = torch.load(
            captures / "prefix-trial.pt", map_location="cpu", weights_only=True
        )
        outputs = trial["outputs"]
        selected_row = receipt["selected_row"]
        assert selected_row == trial["selected_row"]
        peers = torch.arange(receipt["request_rows"]) != selected_row
        for name in outputs[0]:
            assert torch.equal(outputs[0][name], outputs[2][name]), name
            assert torch.equal(outputs[1][name], outputs[3][name]), name
            assert torch.equal(outputs[0][name][peers], outputs[1][name][peers]), name
        for output_field, capture_field in (
            ("logits", "logits"),
            ("query", "predictor_query"),
            ("codes", "codec_ids"),
            ("feedback", "feedback"),
        ):
            assert torch.equal(
                outputs[0][output_field][selected_row], original[1][capture_field]
            )
            assert torch.equal(
                outputs[-1][output_field][selected_row], frames[1][capture_field]
            )
        generated = json.loads((directory / "seedtts/generated.json").read_text())
        assert len(generated) == 1088 and all(row["is_success"] for row in generated)
        selected = next(row for row in generated if row["sample_id"] == sample_id)
        waveform = directory / "seedtts/audio" / f"{sample_id}.wav"
        results.append(
            {
                "arm": arm,
                "sample_id": sample_id,
                "target_text": target_text,
                "waveform_sha256": sha256(waveform.read_bytes()).hexdigest(),
                "capture_sha256": sha256(
                    (captures / sample_id / "trajectory.pt").read_bytes()
                ).hexdigest(),
                "frames": len(frames),
                "duration_seconds": selected["audio_duration_s"],
                "prefix_trial": receipt,
            }
        )
    output = root / "prefix-carry-scored.json"
    with output.open("x") as report:
        json.dump(results, report, indent=2)
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
            waveform = root / result["arm"] / "seedtts/audio" / f"{sample_id}.wav"
            assert (
                sha256(waveform.read_bytes()).hexdigest() == result["waveform_sha256"]
            )
            score = transcribe_and_compute_wer(
                SampleOutput(
                    sample_id=sample_id,
                    target_text=target_text,
                    audio_duration_s=result["duration_seconds"],
                ),
                str(waveform),
                asr,
                "en",
                "cuda:0",
            )
            assert score.is_success and score.is_transcribed
            result["score"] = asdict(score)
            output.write_text(json.dumps(results, indent=2))
            print(json.dumps({"arm": result["arm"], "wer": score.wer}), flush=True)
    finally:
        stop_server(server)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("donor", type=Path)
    parser.add_argument("asr_checkpoint", type=Path)
    arguments = parser.parse_args()
    main(arguments.root, arguments.control, arguments.donor, arguments.asr_checkpoint)
else:
    pass
