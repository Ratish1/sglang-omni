# SPDX-License-Identifier: Apache-2.0
"""Score the same generated waveforms against requested and returned text."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    evaluate_generated_audio,
)
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    apply_wer,
    compute_text_audio_consistency_from_records,
)
from benchmarks.tasks.tts import run_seedtts_similarity, run_seedtts_utmos
from sglang_omni.utils import find_available_port

SAMPLE_COUNT = 1088


def score_speech(run: Path, port: int) -> dict:
    directory = run / "generation"
    config = OmniSeedttsBenchmarkConfig(
        model="qwen3-omni",
        meta=DATASETS["seedtts"],
        output_dir=str(directory),
        host="127.0.0.1",
        port=port,
        voice_clone=True,
        asr_concurrency=1,
    )
    result = evaluate_generated_audio(config)
    speed = json.loads((directory / "speed_results.json").read_text())
    rows = {row["id"]: row for row in speed["per_request"]}
    assert len(rows) == SAMPLE_COUNT
    assert result["wer_summary"]["evaluated"] == SAMPLE_COUNT
    assert result["wer_summary"]["skipped"] == 0
    spoken_outputs = []
    evidence = []
    for original in result["per_sample"]:
        row = rows[original.sample_id]
        returned = row["returned_text"]
        assert returned and row["server_request_id"] and original.is_success, row["id"]
        spoken = apply_wer(
            SampleOutput(sample_id=original.sample_id, target_text=returned),
            original.whisper_text,
            "en",
        )
        assert spoken.is_success, row["id"]
        spoken_outputs.append(spoken)
        evidence.append(
            {
                "id": original.sample_id,
                "request_id": row["server_request_id"],
                "target_text": original.target_text,
                "returned_text": returned,
                "transcript": original.whisper_text,
                "matches_requested_normalized": original.ref_norm == spoken.ref_norm,
                "requested_wer": original.wer,
                "spoken_wer": spoken.wer,
                "latency_s": row["latency_s"],
                "audio_duration_s": row["audio_duration_s"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "waveform_sha256": hashlib.sha256(
                    Path(row["wav_path"]).read_bytes()
                ).hexdigest(),
            }
        )
    return {
        "speed": speed["summary"],
        "requested_wer": result["wer_summary"],
        "spoken_wer": calculate_wer_metrics(spoken_outputs, "en"),
        "records": evidence,
    }


def score_acoustics(run: Path) -> dict:
    config = OmniSeedttsBenchmarkConfig(
        model="qwen3-omni",
        meta=DATASETS["seedtts"],
        output_dir=str(run / "generation"),
        voice_clone=True,
    )
    return {
        "similarity": run_seedtts_similarity(config),
        "utmos": run_seedtts_utmos(config),
    }


def score_visual(run: Path, port: int) -> dict:
    evidence = json.loads((run / "visual-results.json").read_text())
    for name, report in evidence["reports"].items():
        del report["config"]
        if name == "image-text":
            continue
        else:
            pass
        records = report["per_sample"]
        audio_dir = run / name / "audio"
        assert all((audio_dir / f"{row['sample_id']}.wav").is_file() for row in records)
        report["wer"] = compute_text_audio_consistency_from_records(
            records,
            "en",
            "cuda:0",
            audio_dir=str(audio_dir),
            asr_router_port=port,
            asr_concurrency=1,
        )
        assert report["wer"]["summary"]["evaluated"] == len(records), name
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--visual-runs", nargs="*", type=Path, default=())
    parser.add_argument("runs", nargs="*", type=Path)
    arguments = parser.parse_args()
    if not arguments.runs and not arguments.visual_runs:
        parser.error("provide SeedTTS runs, visual runs, or both")
    else:
        pass
    logging.basicConfig(level=logging.INFO)

    voice_manifests = {}
    for run in arguments.runs:
        manifest = json.loads((run / "inputs.json").read_text())
        assert manifest["seed"] is None and manifest["concurrency"] == 16, run
        assert len(manifest["samples"]) == SAMPLE_COUNT, run
        voice_manifests[run.name] = manifest
    if voice_manifests:
        first = next(iter(voice_manifests.values()))
        for manifest in voice_manifests.values():
            assert manifest["samples"] == first["samples"]
            assert manifest["checkpoint"] == first["checkpoint"]
    else:
        pass

    port = find_available_port()
    server = start_server_from_cmd(
        [
            sys.executable,
            "-P",
            "-m",
            "sglang_omni.cli",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            str(arguments.asr_checkpoint.resolve(strict=True)),
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        arguments.output.with_suffix(".asr.log"),
        port,
        timeout=600,
        strip_proxy=True,
    )
    try:
        visual_reports = {}
        for run in arguments.visual_runs:
            visual_reports[run.name] = score_visual(run, port)
            logging.info(f"Finished visual speech scoring: {run.name}")
        reports = {}
        for run in arguments.runs:
            reports[run.name] = score_speech(run, port)
            logging.info(f"Finished primary ASR scoring: {run.name}")
    finally:
        stop_server(server)

    for run in arguments.runs:
        reports[run.name].update(score_acoustics(run))
        logging.info(f"Finished acoustic scoring: {run.name}")
    arguments.output.write_text(
        json.dumps(
            {
                "manifest": voice_manifests,
                "asr_checkpoint": arguments.asr_checkpoint.name,
                "reports": reports,
                "visual_reports": visual_reports,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
