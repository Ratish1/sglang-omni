# SPDX-License-Identifier: Apache-2.0
"""Compare TTS and both MMSU workloads with the same pinned servers and data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from benchmarks.benchmarker.utils import (
    no_proxy_env,
    start_server_from_cmd,
    stop_server,
)
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    evaluate_generated_audio,
)
from benchmarks.metrics.wer import SampleOutput, calculate_wer_metrics
from benchmarks.tasks.asr import QWEN3_ASR_MODEL_PATH, apply_wer
from scripts.qualify_qwen3_omni_seedtts import (
    ASR_REVISION,
    CONCURRENCY,
    OMNI_MODEL,
    OMNI_REVISION,
    PHASE_TIMEOUT_SECONDS,
    SAMPLE_COUNT,
    SERVER_TIMEOUT_SECONDS,
    check_generation,
    prepare,
)
from sglang_omni.utils import find_available_port

MMSU_TALKER_COUNT = 40
MMSU_TEXT_COUNT = 2000
MMSU_DATASET = "zhaochenyang20/mmsu-ci-2000"
ARMS = ("main-a", "narrow-a", "full-a", "narrow-b", "main-b")
RESIDUAL_ARM = "residual-a"


def build_mmsu_packet(run: Path) -> None:
    source = run / "source/mmsu"
    archive = source / "archive"
    packet = run / "mmsu-inputs"
    packet.mkdir()
    catalog = json.loads(
        gzip.decompress((run / "source/catalog.json.gz").read_bytes())
    )["records"]
    records = {row["sample_id"]: row for row in catalog if row["workload"] == "mmsu"}
    results = json.loads((archive / "results.json").read_text())
    order = [row["sample_id"] for row in results["per_sample"]]
    samples = json.loads((archive / "samples.json").read_text())
    reference_hashes = {
        Path(path).name: digest
        for digest, path in (
            line.split(maxsplit=1)
            for line in (source / "reference-sha256.txt").read_text().splitlines()
        )
    }
    assert len(order) == len(records) == len(samples) == MMSU_TALKER_COUNT
    assert set(order) == records.keys() == {sample["sample_id"] for sample in samples}
    for sample in samples:
        sample_id = sample["sample_id"]
        row = records[sample_id]
        request_id = row["request_id"]
        body = (archive / row["request_filename"]).read_bytes()
        assert hashlib.sha256(body).hexdigest() == row["raw_request_sha256"]
        payload = json.loads(body)
        assert (
            hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            == row["request_sha256"]
        )
        assert (
            payload["modalities"] == ["text", "audio"]
            and payload["request_id"] == request_id
        )
        (original_audio,) = payload["audios"]
        assert sample["audio_path"] == original_audio
        filename = Path(original_audio).name
        audio_file = source / "audio" / filename
        assert (
            hashlib.sha256(audio_file.read_bytes()).hexdigest()
            == reference_hashes[filename]
        )
        payload["audios"] = [str(audio_file)]
        sample["audio_path"] = str(audio_file)
        request_bytes = json.dumps(payload, sort_keys=True).encode()
        (packet / f"{request_id}.request.json").write_bytes(request_bytes)
        (packet / f"{request_id}.response.json").write_text(
            json.dumps(
                {**row, "request_sha256": hashlib.sha256(request_bytes).hexdigest()}
            )
        )
    (packet / "order.json").write_text(
        json.dumps({"per_sample": [{"sample_id": item} for item in order]})
    )
    (packet / "samples.json").write_text(json.dumps(samples))


def run_logged(command: list[str], log_file: Path, tree: Path) -> None:
    with log_file.open("w") as log:
        subprocess.run(
            [sys.executable, "-P", *command],
            cwd=tree,
            env={**no_proxy_env(), "PYTHONPATH": str(tree)},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=PHASE_TIMEOUT_SECONDS,
        )


def measure_generation(
    run: Path, tree: Path, destination: Path, port: int, replay: Path
) -> None:
    base_url = f"http://127.0.0.1:{port}"
    run_logged(
        [
            "-m",
            "benchmarks.eval.benchmark_omni_mmsu",
            "--base-url",
            base_url,
            "--modalities",
            "text",
            "--repo-id",
            MMSU_DATASET,
            "--model",
            "qwen3-omni",
            "--max-tokens",
            "32",
            "--temperature",
            "0",
            "--warmup",
            "0",
            "--max-concurrency",
            str(CONCURRENCY),
            "--output-dir",
            str(destination / "mmsu-text"),
            "--disable-tqdm",
        ],
        destination / "mmsu-text.log",
        tree,
    )
    text_report = json.loads((destination / "mmsu-text/mmsu_results.json").read_text())
    assert text_report["summary"]["total_samples"] == MMSU_TEXT_COUNT
    assert text_report["summary"]["failed_samples"] == 0
    print(
        f"{destination.name} MMSU text complete: {text_report['summary']['overall_accuracy']}",
        flush=True,
    )
    for workload, packet, name, count in (
        ("mmsu", run / "mmsu-inputs", "mmsu-talker", MMSU_TALKER_COUNT),
        ("tts", run / "inputs", "generation", SAMPLE_COUNT),
    ):
        if destination.name == "full-a" and workload == "tts":
            continue
        else:
            pass
        output = destination / name
        run_logged(
            [
                str(replay),
                "--mode",
                "replay",
                "--workload",
                workload,
                "--reference-results",
                str(packet / "order.json"),
                "--input-directory",
                str(packet),
                "--output-directory",
                str(output),
                "--base-url",
                base_url,
                "--concurrency",
                str(CONCURRENCY),
            ],
            destination / f"{name}.log",
            tree,
        )
        result = json.loads((output / "results.json").read_text())["summary"]
        assert result["completed_requests"] == count and result["failed_requests"] == 0
        if workload == "tts":
            check_generation(output)
        else:
            pass
        print(
            f"{destination.name} {name} complete: {result['throughput_qps']} req/s",
            flush=True,
        )


def score_speech(
    run: Path, tree: Path, port: int, replay: Path, arms: tuple[str, ...]
) -> None:
    for arm in arms:
        destination = run / arm
        run_logged(
            [
                str(replay),
                "--mode",
                "score",
                "--workload",
                "mmsu",
                "--input-directory",
                str(destination / "mmsu-talker"),
                "--output-directory",
                str(destination / "mmsu-asr"),
                "--asr-port",
                str(port),
                "--asr-model",
                QWEN3_ASR_MODEL_PATH,
            ],
            destination / "mmsu-asr.log",
            tree,
        )
        if arm == "full-a":
            continue
        else:
            pass
        generation = destination / "generation"
        config = OmniSeedttsBenchmarkConfig(
            model="qwen3-omni",
            meta=str(run / "inputs/meta.lst"),
            output_dir=str(generation),
            host="127.0.0.1",
            port=port,
            voice_clone=True,
        )
        results = evaluate_generated_audio(config)
        assert (
            results["wer_summary"]["evaluated"] == SAMPLE_COUNT
            and results["wer_summary"]["skipped"] == 0
        )
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", required=True, choices=("prepare", "generate", "asr")
    )
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--arm", choices=(*ARMS, RESIDUAL_ARM))
    parser.add_argument("--replay", type=Path)
    arguments = parser.parse_args()
    run = arguments.run.resolve(strict=True)
    if arguments.phase == "prepare":
        prepare(run)
        build_mmsu_packet(run)
        print("Both request packets verified", flush=True)
        return
    else:
        pass
    tree = arguments.tree.resolve(strict=True)
    replay = arguments.replay.resolve(strict=True)
    port = find_available_port()
    generation = arguments.phase == "generate"
    model, revision = (
        (OMNI_MODEL, OMNI_REVISION)
        if generation
        else (QWEN3_ASR_MODEL_PATH, ASR_REVISION)
    )
    assert Path(snapshot_download(model)).name == revision
    if generation:
        destination = run / arguments.arm
    elif arguments.arm is None:
        destination = run / "asr"
    else:
        destination = run / f"{arguments.arm}-asr"
    destination.mkdir()
    server_options = (
        [
            "--config",
            str(tree / "examples/configs/qwen3_omni_colocated_h200.yaml"),
        ]
        if generation
        else []
    )
    package_path = Path(
        subprocess.check_output(
            [
                sys.executable,
                "-P",
                "-c",
                "import sglang_omni; print(sglang_omni.__file__)",
            ],
            env={**no_proxy_env(), "PYTHONPATH": str(tree)},
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
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            model,
            "--model-name",
            "qwen3-omni" if generation else model,
            *server_options,
        ],
        destination / "server.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        env={"PYTHONPATH": str(tree)},
        strip_proxy=True,
    )
    try:
        if generation:
            measure_generation(run, tree, destination, port, replay)
        else:
            arms = ARMS if arguments.arm is None else (arguments.arm,)
            score_speech(run, tree, port, replay, arms)
    finally:
        stop_server(server)
    processes = subprocess.check_output(["ps", "-eo", "pgid=,stat="], text=True)
    assert not any(
        int(group) == server.pid and not status.startswith("Z")
        for group, status in (line.split() for line in processes.splitlines())
    ), server.pid


if __name__ == "__main__":
    main()
