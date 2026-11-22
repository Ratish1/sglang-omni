# SPDX-License-Identifier: Apache-2.0
"""Run the ordinary SeedTTS and MMSU clients with an untimed model observer off."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
from contextlib import AbstractAsyncContextManager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from aiohttp import ClientResponse, ClientSession
from qwen3_omni_corpus_support import (
    CLIENT_CONCURRENCY,
    MMSU_SAMPLE_COUNT,
    MMSU_TTS_PROMPT,
    REQUEST_SEED,
    SEEDTTS_SAMPLE_COUNT,
    SERVER_TIMEOUT_SECONDS,
    run_seedtts_profile,
    seedtts_command,
    server_command,
)

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS, SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval import benchmark_omni_seedtts as seedtts
from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu
from sglang_omni.utils import find_available_port

TracePhase = Literal["native", "opening", "after", "all", "heldstock", "heldopening"]
TRACE_PHASES: tuple[TracePhase, ...] = ("native", "opening", "after", "all")
INPUT_CROSS_PHASES: tuple[TracePhase, ...] = (
    "native",
    "opening",
    "heldstock",
    "heldopening",
)
GPU_STAGES = ("image_encoder", "audio_encoder", "thinker", "talker_ar", "code2wav")


async def run_seedtts(
    output: Path,
    port: int,
    sample_ids: set[str],
    *,
    serial_panel: bool = False,
    trace_phase: TracePhase | None = None,
    trace_sample_ids: set[str] | None = None,
    warmup_requests: int | None = None,
) -> None:
    returned_text = {}
    sample = ContextVar("sample", default=None)
    read_json = ClientResponse.json
    post_request = ClientSession.post

    def label_request(
        session: ClientSession, url: str, *, json: dict[str, object]
    ) -> AbstractAsyncContextManager[ClientResponse]:
        sample_id = sample.get()
        assert sample_id is not None and trace_phase is not None
        if trace_sample_ids is None or sample_id in trace_sample_ids:
            return post_request(
                session,
                url,
                json={
                    **json,
                    "request_id": f"feedback-trace-{trace_phase}-{sample_id}",
                },
            )
        else:
            return post_request(session, url, json=json)

    async def capture_json(response, *args, **kwargs):
        result = await read_json(response, *args, **kwargs)
        sample_id = sample.get()
        if sample_id is not None:
            if trace_phase is not None and (
                trace_sample_ids is None or sample_id in trace_sample_ids
            ):
                assert (
                    result["id"] == f"chatcmpl-feedback-trace-{trace_phase}-{sample_id}"
                )
            else:
                pass
            choice = result["choices"][0]
            returned_text[sample_id] = {
                "text": choice["message"]["content"],
                "finish_reason": choice.get("finish_reason"),
            }
        return result

    class RecordingVoiceClone(seedtts.VoiceCloneOmni):
        async def generate_speech(
            self, session, api_url, model_name, item, *args, **kwargs
        ):
            token = sample.set(item.sample_id if item.sample_id in sample_ids else None)
            try:
                return await super().generate_speech(
                    session, api_url, model_name, item, *args, **kwargs
                )
            finally:
                sample.reset(token)

    def load_panel_samples(
        dataset_id: str, max_samples: int | None, *, split: str
    ) -> list[SampleInput]:
        return [
            sample_row
            for sample_row in load_seedtts_samples(dataset_id, max_samples, split=split)
            if sample_row.sample_id in sample_ids
        ]

    config = seedtts._config_from_args(
        seedtts._build_arg_parser().parse_args(
            [
                "--voice-clone",
                "--max-concurrency",
                str(1 if serial_panel else CLIENT_CONCURRENCY),
                "--seed",
                str(REQUEST_SEED),
                "--disable-tqdm",
                "--port",
                str(port),
                "--output-dir",
                str(output),
                "--generate-only",
                *(
                    ("--warmup", str(warmup_requests))
                    if warmup_requests is not None
                    else (("--warmup", "0") if trace_phase is not None else ())
                ),
            ]
        )
    )
    dataset_loader = load_panel_samples if serial_panel else load_seedtts_samples
    with (
        patch.object(seedtts, "VoiceCloneOmni", RecordingVoiceClone),
        patch.object(ClientResponse, "json", capture_json),
        patch.object(seedtts, "load_seedtts_samples", dataset_loader),
        (
            patch.object(ClientSession, "post", label_request)
            if trace_phase is not None
            else nullcontext()
        ),
    ):
        result = await seedtts.run_omni_seedtts_benchmark(config)
    assert result["summary"]["completed_requests"] == len(sample_ids)
    assert result["summary"]["failed_requests"] == 0
    assert returned_text.keys() == sample_ids
    assert all(
        isinstance(row["text"], str) and row["finish_reason"] != "length"
        for row in returned_text.values()
    )
    (output / "returned-text.json").write_text(json.dumps(returned_text) + "\n")


def main(
    root: Path,
    arm: str,
    checkpoint: Path,
    arithmetic: Literal["rounded_add", "torch_reference"],
    cache_directory: Path,
    prewarm_predictor: bool,
    reuse_predictor_capture_stream: bool,
    profile_seedtts_stages: bool,
    seedtts_only: bool,
    measure_server_memory: bool,
    serial_sample_ids: set[str],
    serial_feedback_frames: bool,
    serial_first_input_cross: bool,
    server_gpu_index: int | None,
) -> None:
    if arm not in (
        "stock",
        "prefill_only",
        "device_fa3",
        "opening_ties",
        "stream_reuse",
        "bulk_output_lengths",
        "feedback_store",
        "feedback_opening",
    ):
        raise ValueError(arm)
    else:
        pass
    if profile_seedtts_stages and not prewarm_predictor:
        raise ValueError("SeedTTS stage profiling requires predictor prewarm")
    else:
        pass
    if serial_sample_ids and not seedtts_only:
        raise ValueError("A serial panel must run with --seedtts-only")
    else:
        pass
    if serial_feedback_frames and not (
        serial_sample_ids and arm in ("stock", "feedback_store")
    ):
        raise ValueError("Feedback traces require a serial stock/candidate panel")
    else:
        pass
    if serial_first_input_cross and not (serial_feedback_frames and arm == "stock"):
        raise ValueError("Input crossovers need stock serial feedback traces")
    else:
        pass
    device_fa3_cap = int(os.environ.get("OMNI_FA3_QUALIFY_SOLO_CAP", "0"))
    if arm == "device_fa3" and device_fa3_cap <= 1:
        raise ValueError("device_fa3 needs the measured native solo-cap value")
    selected_arithmetic = arithmetic if arm == "prefill_only" else arm
    output = root / arm
    output.mkdir(parents=True)
    (output / "arithmetic.json").write_text(
        json.dumps(
            {
                "arithmetic": selected_arithmetic,
                "prewarm_predictor": prewarm_predictor,
                "reuse_predictor_capture_stream": reuse_predictor_capture_stream,
                "profile_seedtts_stages": profile_seedtts_stages,
                "seedtts_only": seedtts_only or profile_seedtts_stages,
            }
        )
        + "\n"
    )
    seedtts_samples = load_seedtts_samples(SEEDTTS_DATASET_ID)
    ids = {item.sample_id for item in seedtts_samples}
    assert len(ids) == len(seedtts_samples) == SEEDTTS_SAMPLE_COUNT
    if serial_sample_ids:
        assert serial_sample_ids <= ids, serial_sample_ids - ids
        ids = serial_sample_ids
        seedtts_samples = [item for item in seedtts_samples if item.sample_id in ids]
    else:
        pass
    mmsu = load_mmsu_samples(
        max_samples=MMSU_SAMPLE_COUNT, repo_id=DATASETS["mmsu-ci-2000"]
    )
    assert len(mmsu) == MMSU_SAMPLE_COUNT
    corpus = {
        "seedtts": [
            [
                item.sample_id,
                item.ref_text,
                item.target_text,
                hashlib.sha256(Path(item.ref_audio).read_bytes()).hexdigest(),
            ]
            for item in seedtts_samples
        ],
        "mmsu": [
            [
                item.sample_id,
                item.question,
                item.choices,
                item.answer_index,
                hashlib.sha256(Path(item.audio_path).read_bytes()).hexdigest(),
            ]
            for item in mmsu
        ],
        "concurrency": 1 if serial_sample_ids else CLIENT_CONCURRENCY,
        "seed": REQUEST_SEED,
        "mmsu_prompt": MMSU_TTS_PROMPT,
        "checkpoint": checkpoint.name,
    }
    digest = hashlib.sha256(json.dumps(corpus).encode()).hexdigest()
    (output / "workload-identity.json").write_text(
        json.dumps({"sha256": digest}) + "\n"
    )
    port = find_available_port()
    server_log_path = output / "server.log"
    memory_timeline_path = output / "server-memory-timeline.jsonl"
    feedback_environment = {}
    if serial_feedback_frames:
        trace_directory = output / "frame-traces"
        trace_directory.mkdir()
        feedback_environment = {
            "OMNI_FEEDBACK_FRAME_TRACE_DIRECTORY": str(trace_directory),
            "OMNI_FEEDBACK_FRAME_NATIVE_REDUCTION": (
                "stock" if arm == "stock" else "checkpoint"
            ),
        }
        if serial_first_input_cross:
            prefix_directory = output / "prefix-traces"
            prefix_directory.mkdir()
            feedback_environment["OMNI_FEEDBACK_PREFIX_DIRECTORY"] = str(
                prefix_directory
            )
        else:
            pass
    else:
        pass
    if measure_server_memory:
        idle_memory_mib = record_server_gpu_memory(memory_timeline_path, "idle")
    else:
        pass
    server = start_server_from_cmd(
        [
            *server_command(),
            "--port",
            str(port),
            "--model-path",
            str(checkpoint),
            "--model-name",
            "qwen3-omni",
            "--config",
            "examples/configs/qwen3_omni_colocated_h200.yaml",
            *(
                argument
                for stage_name in GPU_STAGES
                if server_gpu_index is not None
                for argument in (f"--{stage_name}.gpu", str(server_gpu_index))
            ),
        ],
        server_log_path,
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        env={
            **feedback_environment,
            "OMNI_WITNESS_PREFILL_NORM_ONLY": str(int(arm == "prefill_only")),
            "OMNI_WITNESS_PREFILL_NORM_SCOPE": "feed_forward",
            "OMNI_WITNESS_PREFILL_NORM_ARITHMETIC": arithmetic,
            "OMNI_WITNESS_FIXED_SHARED_GATE": "0",
            "OMNI_WITNESS_RESOLVE_PREFILL_HEAD_TIES": str(int(arm == "opening_ties")),
            "OMNI_WITNESS_REUSE_PREDICTOR_CAPTURE_STREAM": str(
                int(reuse_predictor_capture_stream)
            ),
            "OMNI_WITNESS_PREDICTOR_PREWARM_ROWS": str(
                CLIENT_CONCURRENCY if prewarm_predictor else 0
            ),
            "OMNI_FA3_DEVICE_SCHEDULE_CAP": (
                str(device_fa3_cap) if arm == "device_fa3" else ""
            ),
            "SGLANG_CACHE_DIR": str(cache_directory),
        },
        strip_proxy=True,
    )
    try:
        if measure_server_memory:
            ready_memory_mib = record_server_gpu_memory(memory_timeline_path, "ready")
        else:
            pass
        if prewarm_predictor:
            ready_server_log = server_log_path.read_bytes()
            prewarm_logs = [
                line.decode()
                for line in ready_server_log.splitlines()
                if b"Predictor prewarm complete:" in line
            ]
            assert len(prewarm_logs) == 1, prewarm_logs
            (output / "predictor-prewarm.json").write_text(
                json.dumps({"startup_log": prewarm_logs[0]}) + "\n"
            )
        else:
            pass
        if profile_seedtts_stages:
            run_seedtts_profile(
                [
                    *seedtts_command(),
                    "--port",
                    str(port),
                    "--output-dir",
                    str(output / "seedtts"),
                    "--generate-only",
                ],
                port,
                output,
            )
        elif serial_feedback_frames:
            for phase in (
                INPUT_CROSS_PHASES if serial_first_input_cross else TRACE_PHASES
            ):
                asyncio.run(
                    run_seedtts(
                        output / f"seedtts-{phase}",
                        port,
                        ids,
                        serial_panel=True,
                        trace_phase=phase,
                    )
                )
        elif serial_sample_ids:
            for visit_index in range(2):
                asyncio.run(
                    run_seedtts(
                        output / f"seedtts-serial-{visit_index + 1}",
                        port,
                        ids,
                        serial_panel=True,
                    )
                )
        else:
            asyncio.run(run_seedtts(output / "seedtts", port, ids))
        if measure_server_memory:
            after_seedtts_memory_mib = record_server_gpu_memory(
                memory_timeline_path, "after_seedtts"
            )
        else:
            pass
        if not profile_seedtts_stages and not seedtts_only:
            result = asyncio.run(
                run_mmsu(
                    argparse.Namespace(
                        base_url=f"http://127.0.0.1:{port}",
                        model="qwen3-omni",
                        modalities="text+audio",
                        output_dir=str(output / "mmsu"),
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
                    samples=mmsu,
                    compute_wer=False,
                ),
            )
            assert [row["sample_id"] for row in result["per_sample"]] == [
                item.sample_id for item in mmsu
            ]
            assert all(
                row["is_success"] and row["has_audio"] for row in result["per_sample"]
            )
        else:
            pass
        if prewarm_predictor:
            measured_log = server_log_path.read_bytes()[len(ready_server_log) :]
            capture_lines = [
                line.decode(errors="replace")
                for line in measured_log.splitlines()
                if b"Captured Qwen3-Omni predictor CUDA graph" in line
            ]
            assert not capture_lines, capture_lines
        else:
            pass
        if measure_server_memory:
            after_generation_memory_mib = record_server_gpu_memory(
                memory_timeline_path, "after_generation"
            )
            (output / "server-memory.json").write_text(
                json.dumps(
                    {
                        "idle_mib": idle_memory_mib,
                        "ready_mib": ready_memory_mib,
                        "after_seedtts_mib": after_seedtts_memory_mib,
                        "after_generation_mib": after_generation_memory_mib,
                    }
                )
                + "\n"
            )
        else:
            pass
    finally:
        stop_server(server)


def record_server_gpu_memory(timeline_path: Path, phase: str) -> int:
    device_memory_mib = int(
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
    )
    process_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=0",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    device_process_memory_mib = {
        int(host_pid): int(memory_mib)
        for host_pid, memory_mib in (line.split(",") for line in process_rows)
    }
    with timeline_path.open("a") as timeline:
        timeline.write(
            json.dumps(
                {
                    "phase": phase,
                    "device_memory_mib": device_memory_mib,
                    "host_process_memory_mib": device_process_memory_mib,
                }
            )
            + "\n"
        )
    return device_memory_mib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("arm")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "arithmetic",
        choices=("rounded_add", "torch_reference"),
        nargs="?",
        default="rounded_add",
    )
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--prewarm-predictor", action="store_true")
    parser.add_argument("--reuse-predictor-capture-stream", action="store_true")
    parser.add_argument("--profile-seedtts-stages", action="store_true")
    parser.add_argument("--seedtts-only", action="store_true")
    parser.add_argument("--measure-server-memory", action="store_true")
    parser.add_argument("--serial-sample-id", action="append", default=[])
    parser.add_argument("--serial-feedback-frames", action="store_true")
    parser.add_argument("--serial-first-input-cross", action="store_true")
    parser.add_argument("--server-gpu-index", type=int)
    arguments = parser.parse_args()
    main(
        arguments.root,
        arguments.arm,
        arguments.checkpoint,
        arguments.arithmetic,
        arguments.cache_directory
        or arguments.root.parent / "full-prefill-norm-screen/sglang-cache",
        arguments.prewarm_predictor,
        arguments.reuse_predictor_capture_stream,
        arguments.profile_seedtts_stages,
        arguments.seedtts_only,
        arguments.measure_server_memory,
        set(arguments.serial_sample_id),
        arguments.serial_feedback_frames,
        arguments.serial_first_input_cross,
        arguments.server_gpu_index,
    )
else:
    pass
