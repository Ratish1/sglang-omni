# SPDX-License-Identifier: Apache-2.0
"""Replay the pinned full SeedTTS corpus and score identical WAVs with both SIM paths."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from huggingface_hub import snapshot_download

from benchmarks.benchmarker.utils import (
    no_proxy_env,
    start_server_from_cmd,
    stop_server,
)
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    evaluate_generated_audio,
)
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import QWEN3_ASR_MODEL_PATH, apply_wer
from sglang_omni.utils import find_available_port

SAMPLE_COUNT = 1088
CONCURRENCY = 16
OMNI_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
OMNI_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
ASR_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
ARMS = ("main-a", "fixed-a", "fixed-b", "main-b")
SERVER_TIMEOUT_SECONDS = 600
PHASE_TIMEOUT_SECONDS = 1200


def prepare(run: Path) -> None:
    source = run / "source"
    packet = run / "inputs"
    audio = packet / "audio"
    audio.mkdir(parents=True)
    records = json.loads(gzip.decompress((source / "catalog.json.gz").read_bytes()))[
        "records"
    ]
    samples = {
        sample.sample_id: sample
        for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")
    }
    reference_hashes = {
        Path(path).name: digest
        for digest, path in (
            line.split(maxsplit=1)
            for line in (source / "reference-sha256.txt").read_text().splitlines()
        )
    }
    archived = json.loads((source / "requests/speed_results.json").read_text())[
        "per_request"
    ]
    order = [row["id"] for row in archived]
    requested = {row["sample_id"]: row for row in records if row["workload"] == "tts"}
    assert (
        len(order)
        == len(samples)
        == len(requested)
        == len(reference_hashes)
        == SAMPLE_COUNT
    )
    assert set(order) == set(requested) == samples.keys()
    metadata = []
    payload_hash = hashlib.sha256()
    for sample_id in order:
        row = requested[sample_id]
        request_id = row["request_id"]
        original = (source / "requests" / f"{request_id}.request.json").read_bytes()
        assert hashlib.sha256(original).hexdigest() == row["raw_request_sha256"]
        payload = json.loads(original)
        assert (
            hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            == row["request_sha256"]
        )
        assert payload["request_id"] == request_id and payload.get("seed") is None
        (reference,) = payload["audios"]
        filename = Path(reference).name
        assert (
            filename == f"{sample_id}.wav"
            and samples[sample_id].target_text == row["target_text"]
        )
        reference_bytes = Path(samples[sample_id].ref_audio).read_bytes()
        assert (
            hashlib.sha256(reference_bytes).hexdigest() == reference_hashes[filename]
        ), sample_id
        reference_path = audio / filename
        reference_path.write_bytes(reference_bytes)
        payload["audios"] = [str(reference_path)]
        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        payload_hash.update(payload_bytes)
        (packet / f"{request_id}.request.json").write_bytes(payload_bytes)
        (packet / f"{request_id}.response.json").write_text(
            json.dumps(
                {**row, "request_sha256": hashlib.sha256(payload_bytes).hexdigest()}
            )
        )
        metadata.append(f"{sample_id}||{reference_path}|\n")
    (packet / "meta.lst").write_text("".join(metadata))
    (packet / "order.json").write_text(
        json.dumps({"per_request": [{"id": sample_id} for sample_id in order]})
    )
    print(
        f"Verified {SAMPLE_COUNT} original audio files and ordered request SHA-256 {payload_hash.hexdigest()}",
        flush=True,
    )


def exercise_server(
    *,
    model: str,
    model_name: str,
    tree: Path,
    output_directory: Path,
    command_tail: list[str],
    run_phase: argparse.Namespace,
) -> None:
    port = find_available_port()
    server_env = {"PYTHONPATH": str(tree)}
    package_path = Path(
        subprocess.check_output(
            [
                sys.executable,
                "-P",
                "-c",
                "import sglang_omni; print(sglang_omni.__file__)",
            ],
            env={**no_proxy_env(), **server_env},
            text=True,
        ).strip()
    ).resolve()
    assert package_path.is_relative_to(tree), package_path
    server = start_server_from_cmd(
        [
            sys.executable,
            "-P",
            "-m",
            "sglang_omni.cli",
            "serve",
            "--model-path",
            model,
            "--model-name",
            model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            *command_tail,
        ],
        output_directory / "server.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        env=server_env,
        strip_proxy=True,
    )
    try:
        if run_phase.phase == "generate":
            replay = run_phase.replay_driver.resolve(strict=True)
            packet = run_phase.run.resolve() / "inputs"
            command = [
                str(replay),
                "--mode",
                "replay",
                "--workload",
                "tts",
                "--reference-results",
                str(packet / "order.json"),
                "--input-directory",
                str(packet),
                "--output-directory",
                str(output_directory / "generation"),
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--concurrency",
                str(CONCURRENCY),
            ]
            with (output_directory / "run.log").open("w") as log:
                subprocess.run(
                    [sys.executable, *command],
                    cwd=tree,
                    env={**no_proxy_env(), "PYTHONPATH": str(tree)},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=PHASE_TIMEOUT_SECONDS,
                )
            check_generation(output_directory / "generation")
        else:
            for arm in ARMS:
                generation = run_phase.run.resolve(strict=True) / arm / "generation"
                check_generation(generation)
                config = OmniSeedttsBenchmarkConfig(
                    model="qwen3-omni",
                    meta=str(run_phase.run / "inputs/meta.lst"),
                    output_dir=str(generation),
                    port=port,
                    voice_clone=True,
                )
                started = perf_counter()
                results = evaluate_generated_audio(config)
                assert results["wer_summary"]["evaluated"] == SAMPLE_COUNT
                assert results["wer_summary"]["skipped"] == 0
                speech = {
                    row["request_id"]: row["text"]
                    for row in json.loads((generation / "results.json").read_text())[
                        "per_request"
                    ]
                }
                spoken = [
                    apply_wer(
                        SampleOutput(
                            sample_id=row.sample_id, target_text=speech[row.sample_id]
                        ),
                        row.whisper_text,
                        "en",
                    )
                    for row in results["per_sample"]
                ]
                assert all(row.is_success for row in spoken)
                (generation / "spoken_wer_summary.json").write_text(
                    json.dumps(calculate_wer_metrics(spoken, "en"), indent=2)
                )
                print(
                    f"{arm} WER complete in {perf_counter() - started:.2f} s",
                    flush=True,
                )
    finally:
        stop_server(server)
    processes = subprocess.check_output(["ps", "-eo", "pgid=,stat="], text=True)
    assert not any(
        int(group) == server.pid and not status.startswith("Z")
        for group, status in (line.split() for line in processes.splitlines())
    ), server.pid


def check_generation(generation: Path) -> None:
    results = json.loads((generation / "results.json").read_text())
    summary = results["summary"]
    assert summary["total_requests"] == summary["completed_requests"] == SAMPLE_COUNT
    assert summary["failed_requests"] == 0, summary
    assert summary["finish_reason_observed"] == SAMPLE_COUNT, summary
    speeches = {
        row["sample_id"]: row
        for path in generation.glob("*.speech.json")
        for row in [json.loads(path.read_text())]
    }
    assert len(speeches) == SAMPLE_COUNT
    generated = []
    for row in results["per_request"]:
        speech = speeches[row["request_id"]]
        path = Path(row["wav_path"])
        assert path.is_file()
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == speech["waveform_sha256"]
        )
        generated.append(
            {
                "sample_id": row["request_id"],
                "target_text": speech["target_text"],
                "wav_path": str(path),
                "is_success": row["is_success"],
                "latency_s": row["latency_s"],
                "audio_duration_s": row["audio_duration_s"],
            }
        )
    (generation / "generated.json").write_text(json.dumps(generated))
    speed_report = generation / "speed_results.json"
    if speed_report.exists():
        pass
    else:
        shutil.copyfile(generation / "results.json", speed_report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "generate", "asr"), required=True
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--replay-driver", type=Path)
    arguments = parser.parse_args()
    run = arguments.run.resolve(strict=True)
    if arguments.phase == "prepare":
        prepare(run)
    else:
        tree = arguments.tree.resolve(strict=True)
        model = OMNI_MODEL if arguments.phase == "generate" else QWEN3_ASR_MODEL_PATH
        revision = OMNI_REVISION if arguments.phase == "generate" else ASR_REVISION
        assert Path(snapshot_download(model)).name == revision
        if arguments.phase == "generate":
            destination = run / arguments.arm
            tail = [
                "--config",
                str(tree / "examples/configs/qwen3_omni_colocated_h200.yaml"),
            ]
            model_name = "qwen3-omni"
        else:
            destination = run / "asr"
            tail = []
            model_name = model
        destination.mkdir()
        exercise_server(
            model=model,
            model_name=model_name,
            tree=tree,
            output_directory=destination,
            command_tail=tail,
            run_phase=arguments,
        )


if __name__ == "__main__":
    main()
