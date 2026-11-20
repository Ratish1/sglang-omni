# SPDX-License-Identifier: Apache-2.0
"""Run paired native BF16 SeedTTS and MMSU speech without trace hooks."""

import argparse
import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import psutil

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    compute_text_audio_consistency_from_records,
)
from sglang_omni.utils import find_available_port

MMSU_TTS_PROMPT = (
    "Listen to the audio and answer the multiple-choice question.\n"
    "Briefly explain your reasoning in 2-3 sentences, then on a new final "
    "line output exactly:\n"
    "'Answer: $LETTER' (without quotes) where LETTER is one of the options.\n"
    "Do not exceed 120 words in total."
)
MMSU_SAMPLE_COUNT = 40
CLIENT_CONCURRENCY = 16
REQUEST_SEED = 42
SEEDTTS_SAMPLE_COUNT = 1088
SERVER_TIMEOUT_SECONDS = 600


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--asr-path", required=True)
    parser.add_argument("--server-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms", choices=("baseline", "candidate"), nargs="+", required=True
    )
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--projection-control-build", type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    run_directories = [
        output / f"{index}-{arm}" for index, arm in enumerate(arguments.arms, 1)
    ]
    if not arguments.score_only:
        generate(arguments, run_directories)
    else:
        assert all(directory.is_dir() for directory in run_directories)
    score(arguments.asr_path, output, run_directories)


def server_command() -> list[str]:
    return [
        sys.executable,
        "-P",
        "-m",
        "sglang_omni.cli",
        "serve",
        "--host",
        "127.0.0.1",
    ]


def seedtts_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_seedtts",
        "--voice-clone",
        "--max-concurrency",
        str(CLIENT_CONCURRENCY),
        "--seed",
        str(REQUEST_SEED),
        "--disable-tqdm",
    ]


def generate(arguments: argparse.Namespace, run_directories: list[Path]) -> None:
    output = arguments.output.resolve()
    output.mkdir()
    mmsu_samples = load_mmsu_samples(
        max_samples=MMSU_SAMPLE_COUNT, repo_id=DATASETS["mmsu-ci-2000"]
    )
    assert len(mmsu_samples) == MMSU_SAMPLE_COUNT
    (output / "mmsu-input-order.json").write_text(
        json.dumps([sample.sample_id for sample in mmsu_samples]) + "\n"
    )
    projection_control_build = arguments.projection_control_build
    if projection_control_build is not None:
        projection_control_build = projection_control_build.resolve()
        build_result = json.loads(projection_control_build.read_bytes())
        patched_library = Path(build_result["library"]).resolve()
        with patched_library.open("rb") as library:
            assert (
                hashlib.file_digest(library, "sha256").hexdigest()
                == build_result["library_sha256"]
            )
        assert arguments.arms[0] == "baseline"
    else:
        patched_library = None
    tactic_cache_directory = output / "sglang-cache"
    reference_tactics = None
    for run_directory, arm in zip(run_directories, arguments.arms, strict=True):
        run_directory.mkdir()
        print(f"{run_directory.name}: starting native inference", flush=True)
        server_port = find_available_port()
        if projection_control_build is not None:
            server_environment = {
                "OMNI_WITNESS_FIXED_SHARED_GATE": "0",
                "SGLANG_CACHE_DIR": str(tactic_cache_directory),
            }
            if arm == "candidate":
                server_environment["OMNI_WITNESS_CUTLASS_PROJECTION_SOURCE"] = str(
                    projection_control_build.parent / "csrc"
                )
            else:
                pass
        else:
            server_environment = {
                "OMNI_WITNESS_FIXED_SHARED_GATE": str(int(arm == "candidate"))
            }
        talker_server = start_server_from_cmd(
            [
                *server_command(),
                "--port",
                str(server_port),
                "--config",
                str(arguments.server_config),
                "--model-path",
                arguments.model_path,
                "--model-name",
                "qwen3-omni",
            ],
            run_directory / "talker.log",
            server_port,
            timeout=SERVER_TIMEOUT_SECONDS,
            env=server_environment,
            strip_proxy=True,
        )
        try:
            if patched_library is not None:
                child_processes = psutil.Process(talker_server.pid).children(
                    recursive=True
                )
                mapped_libraries = sorted(
                    {
                        mapping.path
                        for process in child_processes
                        for mapping in process.memory_maps()
                        if Path(mapping.path).name.startswith("fused_moe_90")
                        and mapping.path.endswith(".so")
                    }
                )
                assert mapped_libraries, "No native CUTLASS library mapped"
                assert (str(patched_library) in mapped_libraries) == (
                    arm == "candidate"
                )
                tactic_files = sorted(
                    (tactic_cache_directory / "flashinfer" / "autotune").rglob(
                        "rank_*.json"
                    )
                )
                tactics = {
                    str(tactic_file.relative_to(tactic_cache_directory)): {
                        operation: selection[:2]
                        for operation, selection in json.loads(
                            tactic_file.read_bytes()
                        ).items()
                        if operation.startswith("('trtllm::fused_moe::gemm")
                    }
                    for tactic_file in tactic_files
                }
                assert any(tactics.values()), "No saved CUTLASS tactic selections"
                (run_directory / "cutlass-provider.json").write_text(
                    json.dumps(
                        {
                            "mapped_libraries": mapped_libraries,
                            "tactics": tactics,
                            "fixed_shared_gate": False,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                if reference_tactics is None:
                    reference_tactics = tactics
                else:
                    assert (
                        tactics == reference_tactics
                    ), "Native tactic selections changed"
            else:
                pass
            subprocess.run(
                [
                    *seedtts_command(),
                    "--port",
                    str(server_port),
                    "--output-dir",
                    str(run_directory / "seedtts"),
                    "--generate-only",
                ],
                check=True,
            )
            seedtts_records = json.loads(
                (run_directory / "seedtts" / "generated.json").read_bytes()
            )
            assert len(seedtts_records) == SEEDTTS_SAMPLE_COUNT
            assert all(record["is_success"] for record in seedtts_records)
            mmsu_output = asyncio.run(
                run_mmsu(
                    argparse.Namespace(
                        base_url=f"http://127.0.0.1:{server_port}",
                        model="qwen3-omni",
                        modalities="text+audio",
                        output_dir=str(run_directory / "mmsu"),
                        max_samples=MMSU_SAMPLE_COUNT,
                        max_tokens=256,
                        temperature=0.0,
                        seed=REQUEST_SEED,
                        warmup=0,
                        max_concurrency=CLIENT_CONCURRENCY,
                        request_rate=float("inf"),
                        timeout_s=500,
                        disable_tqdm=True,
                        fingerprint=False,
                        save_audio=True,
                        prompt=MMSU_TTS_PROMPT,
                    ),
                    samples=mmsu_samples,
                    compute_wer=False,
                )
            )
            assert len(mmsu_output["per_sample"]) == MMSU_SAMPLE_COUNT
            assert all(
                row["is_success"] and row["has_audio"]
                for row in mmsu_output["per_sample"]
            )
        finally:
            stop_server(talker_server)
        print(f"{run_directory.name}: completed both generation workloads", flush=True)


def score(asr_path: str, output: Path, run_directories: list[Path]) -> None:
    asr_port = find_available_port()
    asr_server = start_server_from_cmd(
        [
            *server_command(),
            "--port",
            str(asr_port),
            "--model-path",
            asr_path,
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.gpu",
            "0",
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        output / "asr.log",
        asr_port,
        timeout=SERVER_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    try:
        for run_directory in run_directories:
            print(f"{run_directory.name}: scoring WER", flush=True)
            seedtts_directory = run_directory / "seedtts"
            if not (seedtts_directory / "wer_results.json").is_file():
                subprocess.run(
                    [
                        *seedtts_command(),
                        "--port",
                        str(asr_port),
                        "--output-dir",
                        str(seedtts_directory),
                        "--transcribe-only",
                    ],
                    check=True,
                )
            else:
                pass
            mmsu_results_path = run_directory / "mmsu" / "mmsu_results.json"
            mmsu_results = json.loads(mmsu_results_path.read_bytes())
            if not mmsu_results.get("wer"):
                mmsu_results["wer"] = compute_text_audio_consistency_from_records(
                    mmsu_results["per_sample"],
                    "en",
                    "cuda:0",
                    audio_dir=str(run_directory / "mmsu" / "audio"),
                    asr_router_port=asr_port,
                )
                mmsu_results_path.write_text(json.dumps(mmsu_results) + "\n")
            else:
                pass
    finally:
        stop_server(asr_server)

    for run_directory in run_directories:
        print(f"{run_directory.name}: scoring speaker similarity", flush=True)
        seedtts_directory = run_directory / "seedtts"
        if not (seedtts_directory / "similarity_results.json").is_file():
            subprocess.run(
                [
                    *seedtts_command(),
                    "--output-dir",
                    str(seedtts_directory),
                    "--similarity-only",
                ],
                check=True,
            )
        else:
            pass
        for generated_waveform in run_directory.rglob("*.wav"):
            generated_waveform.unlink()
        print(f"{run_directory.name}: finished quality scoring", flush=True)


if __name__ == "__main__":
    main()
