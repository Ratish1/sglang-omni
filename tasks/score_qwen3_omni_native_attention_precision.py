# SPDX-License-Identifier: Apache-2.0
"""Score the native precision pair while preserving the held text reference."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from qwen3_omni_corpus_support import SERVER_TIMEOUT_SECONDS, server_command
from score_qwen3_omni_rounded_add_screen import score_seedtts

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import QWEN3_ASR_MODEL_PATH, apply_wer
from sglang_omni.utils import find_available_port


def main(root: Path, original_returned_text: Path, asr_checkpoint: Path) -> None:
    original = json.loads(original_returned_text.read_text())
    arms = (root / "control", root / "scalar")
    references = [
        json.loads((arm / "conditioning-references.json").read_text()) for arm in arms
    ]
    assert references[0] == references[1]
    for sample_id in references[0]:
        receipts = [
            json.loads(
                (arm / "captures" / f"{sample_id}-conditioning.json").read_text()
            )
            for arm in arms
        ]
        assert all(all(receipt["checks"].values()) for receipt in receipts)
        for name in ("conditioning_sha256", "original_capture_sha256"):
            assert receipts[0][name] == receipts[1][name]
    for arm in arms:
        audit = json.loads((arm / "native-attention-precision.json").read_text())
        assert audit["mode"] == arm.name
        assert {row["sample_id"] for row in audit["captures"]} == references[0].keys()
    waveforms = sorted(root.glob("*/seedtts/audio/*.wav"))
    assert len(waveforms) == 2 * 1088
    with (root / "waveform-hashes.json").open("x") as receipt:
        json.dump(
            {
                str(waveform.relative_to(root)): sha256(
                    waveform.read_bytes()
                ).hexdigest()
                for waveform in waveforms
            },
            receipt,
        )
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
    report = {
        "scope": "Full workload supplies native concurrency, but only named requests receive held text and the experimental precision path. This is a causal diagnostic, not full-policy quality or performance qualification.",
        "original_returned_text_sha256": sha256(
            original_returned_text.read_bytes()
        ).hexdigest(),
        "arms": {},
    }
    try:
        for arm in arms:
            score_seedtts(arm, port)
            requested = json.loads((arm / "seedtts/wer_results.json").read_text())
            transcripts = {row["id"]: row for row in requested["per_sample"]}
            held_scores = {}
            for sample_id in references[0]:
                transcript = transcripts[sample_id]
                held = apply_wer(
                    SampleOutput(
                        sample_id=sample_id,
                        target_text=original[sample_id]["text"],
                        audio_duration_s=transcript["audio_duration_s"],
                    ),
                    transcript["whisper_text"],
                    "en",
                )
                assert held.is_success and held.is_transcribed
                held_scores[sample_id] = asdict(held)
            report["arms"][arm.name] = {
                "full_requested_text_wer": requested["summary"],
                "selected_requested_text": {
                    sample_id: transcripts[sample_id] for sample_id in references[0]
                },
                "selected_held_text": held_scores,
            }
            (root / "attention-precision-speech.json").write_text(
                json.dumps(report, indent=2)
            )
            print(json.dumps({"arm": arm.name, "held_scores": held_scores}), flush=True)
    finally:
        stop_server(server)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("original_returned_text", type=Path)
    parser.add_argument("asr_checkpoint", type=Path)
    arguments = parser.parse_args()
    main(arguments.root, arguments.original_returned_text, arguments.asr_checkpoint)
else:
    pass
