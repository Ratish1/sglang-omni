# SPDX-License-Identifier: Apache-2.0
"""Score the full rounded-residual screen, reusing transcripts for both references."""

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from qwen3_omni_corpus_support import (
    MMSU_SAMPLE_COUNT,
    SEEDTTS_SAMPLE_COUNT,
    SERVER_TIMEOUT_SECONDS,
    seedtts_command,
    server_command,
)

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    apply_wer,
    compute_text_audio_consistency_from_records,
)
from sglang_omni.utils import find_available_port


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def score_seedtts(arm: Path, port: int) -> None:
    seedtts = arm / "seedtts"
    subprocess.run(
        [
            *seedtts_command(),
            "--port",
            str(port),
            "--output-dir",
            str(seedtts),
            "--transcribe-only",
        ],
        check=True,
    )
    requested = read_json(seedtts / "wer_results.json")["per_sample"]
    returned = read_json(seedtts / "returned-text.json")
    assert len(requested) == len(returned) == SEEDTTS_SAMPLE_COUNT
    assert {row["id"] for row in requested} == returned.keys()
    assert all(row["is_transcribed"] and row["is_success"] for row in requested)
    scored = [
        apply_wer(
            SampleOutput(
                sample_id=row["id"],
                target_text=returned[row["id"]]["text"],
                audio_duration_s=row["audio_duration_s"],
                latency_s=row["latency_s"],
            ),
            row["whisper_text"],
            "en",
        )
        for row in requested
    ]
    assert all(row.is_success and row.is_transcribed for row in scored)
    write_json(
        seedtts / "returned-text-wer.json",
        {
            "summary": calculate_wer_metrics(scored, "en"),
            "per_sample": [asdict(row) for row in scored],
        },
    )


def score_arm(arm: Path, port: int) -> None:
    score_seedtts(arm, port)
    mmsu_path = arm / "mmsu" / "mmsu_results.json"
    mmsu = read_json(mmsu_path)
    mmsu["wer"] = compute_text_audio_consistency_from_records(
        mmsu["per_sample"],
        "en",
        "cuda:1",
        audio_dir=str(mmsu_path.parent / "audio"),
        asr_router_port=port,
    )
    assert mmsu["wer"]["summary"]["evaluated"] == MMSU_SAMPLE_COUNT
    assert mmsu["wer"]["summary"]["skipped"] == 0
    write_json(mmsu_path, mmsu)


def main(root: Path, checkpoint: Path, candidate: str = "prefill_only") -> None:
    assert candidate in (
        "prefill_only",
        "device_fa3",
        "opening_ties",
        "stream_reuse",
        "bulk_output_lengths",
        "feedback_store",
        "feedback_opening",
    )
    arms = (root / "stock", root / candidate)
    assert read_json(arms[0] / "workload-identity.json") == read_json(
        arms[1] / "workload-identity.json"
    )
    waves = sorted(root.rglob("*.wav"))
    minimum_measured_waves = 2 * (SEEDTTS_SAMPLE_COUNT + MMSU_SAMPLE_COUNT)
    assert len(waves) >= minimum_measured_waves
    write_json(
        root / "waveform-hashes.json",
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in waves
        },
    )
    port = find_available_port()
    server = start_server_from_cmd(
        [
            *server_command(),
            "--port",
            str(port),
            "--model-path",
            str(checkpoint),
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.gpu",
            "1",
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        root / "asr.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    try:
        for arm in arms:
            score_arm(arm, port)
    finally:
        stop_server(server)
    for option, device in (("--similarity-only", "cuda:1"), ("--utmos-only", "cuda:0")):
        for arm in arms:
            subprocess.run(
                [
                    *seedtts_command(),
                    "--output-dir",
                    str(arm / "seedtts"),
                    "--device",
                    device,
                    option,
                ],
                check=True,
            )
    summary = {}
    for arm in arms:
        seedtts = arm / "seedtts"
        mmsu = read_json(arm / "mmsu/mmsu_results.json")
        summary[arm.name] = {
            "seedtts": {
                filename: read_json(seedtts / f"{filename}.json")["summary"]
                for filename in (
                    "speed_results",
                    "wer_results",
                    "returned-text-wer",
                    "similarity_results",
                    "utmos_results",
                )
            },
            "mmsu": {
                "accuracy": mmsu["summary"],
                "speed": mmsu["speed_metrics"],
                "spoken_text_wer": mmsu["wer"]["summary"],
            },
        }
    write_json(root / "qualification-screen-results.json", summary)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), *sys.argv[3:])
