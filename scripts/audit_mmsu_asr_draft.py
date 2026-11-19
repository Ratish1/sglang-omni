# SPDX-License-Identifier: Apache-2.0
"""Rescore unchanged MMSU waveforms with one Qwen3-ASR checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    compute_text_audio_consistency_from_records,
)
from sglang_omni.utils import find_available_port
from tests.test_model.test_qwen3_omni_mmsu_talker_ci import MAX_SAMPLES
from tests.utils import QWEN3_ASR_WER_CONCURRENCY


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, nargs="+", type=Path)
    parser.add_argument("--model-snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=MAX_SAMPLES)
    parser.add_argument("--plain-only", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir()

    records_by_report = {
        str(path): json.loads(path.read_text())["per_sample"]
        for path in arguments.results
    }
    waveform_hashes = {}
    for path, records in records_by_report.items():
        assert len(records) == arguments.sample_count
        assert len({record["sample_id"] for record in records}) == len(records)
        assert all(record["is_success"] and record["has_audio"] for record in records)
        audio_dir = Path(path).parent / "audio"
        for record in records:
            waveform_path = audio_dir / f"{record['sample_id']}.wav"
            waveform_hashes[waveform_path] = hashlib.sha256(
                waveform_path.read_bytes()
            ).hexdigest()
    (arguments.output / "audio-sha256.json").write_text(
        json.dumps({str(path): digest for path, digest in waveform_hashes.items()})
    )

    modes = ("plain",) if arguments.plain_only else ("draft-a", "plain", "draft-b")
    concurrencies = (1,) if arguments.plain_only else (1, QWEN3_ASR_WER_CONCURRENCY)
    for mode in modes:
        port = find_available_port()
        options = (
            ["--asr.engine.disable_speculative_decoding", "true"]
            if mode == "plain"
            else []
        )
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
                str(arguments.model_snapshot),
                "--model-name",
                QWEN3_ASR_MODEL_PATH,
                *options,
            ],
            arguments.output / f"{mode}.log",
            port,
            timeout=600,
            strip_proxy=True,
        )
        try:
            for concurrency in concurrencies:
                reports = {}
                for path, records in records_by_report.items():
                    audio_dir = Path(path).parent / "audio"
                    report = compute_text_audio_consistency_from_records(
                        records,
                        "en",
                        "cuda:0",
                        audio_dir=str(audio_dir),
                        asr_router_port=port,
                        asr_concurrency=concurrency,
                    )
                    assert report["summary"]["evaluated"] == len(records)
                    assert report["summary"]["skipped"] == 0
                    reports[path] = report
                (arguments.output / f"{mode}-c{concurrency}.json").write_text(
                    json.dumps(reports, indent=2)
                )
        finally:
            stop_server(server)

    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in waveform_hashes.items()
    )


if __name__ == "__main__":
    main()
