"""Recover selected receiver histories under the original complete traffic sequence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import time
from pathlib import Path

import psutil
from huggingface_hub import snapshot_download

from benchmarks.benchmarker.utils import wait_for_service
from scripts.debug_qwen3_omni_producer_speech import stop_owned_server

REQUEST_ID = "e97649f2-c5a0-5acd-998c-d32fb75d6714"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
ASR_MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-directory", type=Path, required=True)
    parser.add_argument("--build-process-id", type=int, required=True)
    parser.add_argument("--download-process-id", type=int, required=True)
    arguments = parser.parse_args()
    root = arguments.root_directory
    output = root / "target-recovery"
    output.mkdir(exist_ok=False)
    progress = []
    for process_id in (arguments.build_process_id, arguments.download_process_id):
        try:
            process = psutil.Process(process_id)
        except psutil.NoSuchProcess:
            process = None
        if process is not None:
            print(f"Waiting for preparation process {process_id}", flush=True)
            while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                time.sleep(10)
        else:
            pass
    build = json.loads((root / "raw-scale-sdk/build-result.json").read_bytes())
    assert build["output_arithmetic"] == "expert-order-bfloat16-sum"
    assert build["activation_arithmetic"] == "native"
    assert build["patched_finalizer_sha256"] == (
        "2003f753756c9f479d48bcf4586d1a8512162fdcfa669fa4a8f9ce32086eb3ff"
    )
    with Path(build["library"]).open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest() == build["library_sha256"]
        )
    for model_id, revision in (
        (MODEL_ID, MODEL_REVISION),
        (ASR_MODEL_ID, ASR_MODEL_REVISION),
    ):
        snapshot_download(model_id, revision=revision, local_files_only=True)
    template = output / "server-template.json"
    template.write_text(
        json.dumps(
            {
                "command": [
                    "python3",
                    "-m",
                    "sglang_omni.cli",
                    "serve",
                    "--config",
                    "examples/configs/qwen3_omni_colocated_h200.yaml",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "18330",
                ]
            },
            indent=2,
        )
    )
    for phase in ("receiver-only", "full-operators"):
        command = [
            "python3",
            "-u",
            "-m",
            "scripts.debug_qwen3_omni_activation_corpus",
            "--server-template",
            str(template),
            "--server-working-directory",
            str(root / "capture-source"),
            "--model-path",
            MODEL_ID,
            "--model-revision",
            MODEL_REVISION,
            "--input-directory",
            str(root / "restored-inputs/corpus-inputs"),
            "--output-directory",
            str(output / phase),
            "--base-url",
            "http://127.0.0.1:18330",
            "--concurrency",
            "16",
            "--control",
            "native-weight-contributions",
            "--contribution-source-directory",
            str(root / "raw-scale-sdk/csrc"),
            "--capture-request-id",
            REQUEST_ID,
            "--capture-log-level",
            "info",
        ]
        if phase == "full-operators":
            command.append("--capture-talker-operators")
        else:
            pass
        (output / f"{phase}-command.json").write_text(json.dumps(command, indent=2))
        print(f"Starting {phase}", flush=True)
        with (output / f"{phase}.log").open("wb") as log:
            completed = subprocess.run(
                command,
                cwd=root / "driver-source",
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        progress.append({"phase": phase, "returncode": completed.returncode})
        (output / "progress.json").write_text(json.dumps(progress, indent=2))
        completed.check_returncode()
        population = output / phase / "native-weight-contributions"
        assert (
            json.loads((population / "stop-ownership.json").read_bytes())[
                "live_survivors"
            ]
            == []
        )
        exports = root / "exports"
        exports.mkdir(exist_ok=True)
        archive_path = exports / f"{phase}-reusable-target.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            for directory in (population / "conditioning",):
                archive.add(directory, arcname=str(directory.relative_to(root)))
            for suffix in (".wav", ".speech.json", ".request.json"):
                source = population / "mmsu" / f"{REQUEST_ID}{suffix}"
                archive.add(source, arcname=str(source.relative_to(root)))
            for source in (
                population / "server-command.json",
                population / "stop-ownership.json",
                output / f"{phase}-command.json",
                root / "environment.json",
                root / "restored-inputs/restoration.json",
                root / "raw-scale-sdk/build-result.json",
            ):
                archive.add(source, arcname=str(source.relative_to(root)))
        with archive_path.open("rb") as stream:
            archive_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        archive_path.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "sha256": archive_sha256,
                    "bytes": archive_path.stat().st_size,
                    "scope": "Selected reusable raw receiver/operator history and one diagnostic waveform. No generated audio population.",
                },
                indent=2,
            )
        )
        print(f"Archived {phase}: {archive_path} sha256={archive_sha256}", flush=True)
    scorer = output / "primary-scorer"
    scorer.mkdir()
    server_command = [
        "python3",
        "-u",
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        ASR_MODEL_ID,
        "--host",
        "127.0.0.1",
        "--port",
        "18332",
        "--asr.gpu",
        "0",
    ]
    with (scorer / "server.log").open("wb") as log:
        server = subprocess.Popen(
            server_command,
            cwd="/workspace/sglang-omni",
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (scorer / "command.json").write_text(
        json.dumps(
            {
                "command": server_command,
                "pid": server.pid,
                "create_time": psutil.Process(server.pid).create_time(),
            },
            indent=2,
        )
    )
    try:
        wait_for_service(
            "http://127.0.0.1:18332",
            server_process=server,
            server_log_file=scorer / "server.log",
            trust_env=False,
        )
        for phase in (
            "receiver-only",
            "full-operators",
            "receiver-only-repeat",
            "full-operators-repeat",
        ):
            inputs = (
                output
                / phase.removesuffix("-repeat")
                / "native-weight-contributions/mmsu"
            )
            command = [
                "python3",
                "-u",
                "-m",
                "scripts.debug_qwen3_omni_replay_corpus",
                "--mode",
                "score",
                "--workload",
                "mmsu",
                "--input-directory",
                str(inputs),
                "--output-directory",
                str(output / f"{phase}-scores"),
                "--asr-port",
                "18332",
                "--asr-model",
                ASR_MODEL_ID,
                "--request-id",
                REQUEST_ID,
            ]
            with (output / f"{phase}-score.log").open("wb") as log:
                completed = subprocess.run(
                    command,
                    cwd=root / "driver-source",
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            progress.append(
                {"phase": f"{phase}-score", "returncode": completed.returncode}
            )
            (output / "progress.json").write_text(json.dumps(progress, indent=2))
            completed.check_returncode()
    finally:
        stop_owned_server(server, scorer)
    print(
        "Capture and repeated same-audio scoring complete; inspect all outcomes before attribution",
        flush=True,
    )


if __name__ == "__main__":
    main()
else:
    pass
