# SPDX-License-Identifier: Apache-2.0
"""Run SeedTTS and full MMSU Talker on default BF16 colocated serving."""

import argparse
import asyncio
import json
import logging
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import run_qwen3_omni_rounded_add_screen as prior_benchmark
from qwen3_omni_corpus_support import (
    MMSU_TTS_PROMPT,
    REQUEST_SEED,
    SEEDTTS_SAMPLE_COUNT,
    SERVER_TIMEOUT_SECONDS,
    server_command,
)

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.eval.benchmark_omni_mmsu import run as run_mmsu
from sglang_omni.utils import find_available_port

MMSU_DATASET = "ddwang2000/MMSU"
COLOCATED_STAGES = (
    "image_encoder",
    "audio_encoder",
    "thinker",
    "talker_ar",
    "code2wav",
)
logger = logging.getLogger(__name__)


def main(
    output: Path,
    checkpoint: Path,
    gpu_index: int,
    concurrency: int,
    seedtts_only: bool,
    cache_directory: Path | None,
    audio_single_request: bool,
) -> None:
    output.mkdir()
    (output / "encoder-policy.json").write_text(
        json.dumps({"audio_single_request": audio_single_request}) + "\n"
    )
    seedtts = load_seedtts_samples(SEEDTTS_DATASET_ID)
    sample_ids = {sample.sample_id for sample in seedtts}
    assert len(sample_ids) == len(seedtts) == SEEDTTS_SAMPLE_COUNT
    mmsu = load_mmsu_samples(repo_id=MMSU_DATASET)
    assert mmsu and len({sample.sample_id for sample in mmsu}) == len(mmsu)
    corpus = {
        "seedtts": [
            [
                sample.sample_id,
                sample.ref_text,
                sample.target_text,
                sha256(Path(sample.ref_audio).read_bytes()).hexdigest(),
            ]
            for sample in seedtts
        ],
        "mmsu": [
            [
                sample.sample_id,
                sample.question,
                sample.choices,
                sample.answer_index,
                sha256(Path(sample.audio_path).read_bytes()).hexdigest(),
            ]
            for sample in mmsu
        ],
        "concurrency": concurrency,
        "seed": REQUEST_SEED,
        "mmsu_prompt": MMSU_TTS_PROMPT,
        "mmsu_repo": MMSU_DATASET,
        "checkpoint": checkpoint.name,
    }
    with (output / "workload-identity.json").open("x") as identity:
        json.dump(
            {
                "sha256": sha256(json.dumps(corpus).encode()).hexdigest(),
                "seedtts_count": len(seedtts),
                "mmsu_count": len(mmsu),
                "concurrency": concurrency,
                "checkpoint": checkpoint.name,
            },
            identity,
            indent=2,
        )
    logger.info(
        f"Matched workload: SeedTTS={len(seedtts)}, MMSU={len(mmsu)}, c={concurrency}"
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
            "qwen3-omni",
            "--config",
            "examples/configs/qwen3_omni_colocated_h200.yaml",
            *(
                argument
                for stage_name in COLOCATED_STAGES
                for argument in (f"--{stage_name}.gpu", str(gpu_index))
            ),
        ],
        output / "server.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        env={
            "SGLANG_CACHE_DIR": str(cache_directory or output / "cache"),
            "OMNI_NATIVE_AUDIO_SINGLE_REQUEST": "1" if audio_single_request else "0",
        },
        strip_proxy=True,
    )
    try:
        logger.info("Starting SeedTTS generation")
        with patch.object(prior_benchmark, "CLIENT_CONCURRENCY", concurrency):
            asyncio.run(
                prior_benchmark.run_seedtts(output / "seedtts", port, sample_ids)
            )
        if seedtts_only:
            return
        else:
            pass
        logger.info("Starting full MMSU Talker generation")
        result = asyncio.run(
            run_mmsu(
                argparse.Namespace(
                    base_url=f"http://127.0.0.1:{port}",
                    model="qwen3-omni",
                    modalities="text+audio",
                    output_dir=str(output / "mmsu"),
                    max_samples=None,
                    max_tokens=256,
                    temperature=0.0,
                    seed=REQUEST_SEED,
                    warmup=0,
                    max_concurrency=concurrency,
                    request_rate=float("inf"),
                    timeout_s=500,
                    disable_tqdm=False,
                    fingerprint=False,
                    save_audio=True,
                    prompt=MMSU_TTS_PROMPT,
                ),
                samples=mmsu,
                compute_wer=False,
            )
        )
        assert [row["sample_id"] for row in result["per_sample"]] == [
            sample.sample_id for sample in mmsu
        ]
        failures = sum(not row["is_success"] for row in result["per_sample"])
        missing_audio = sum(not row["has_audio"] for row in result["per_sample"])
        if failures or missing_audio:
            raise RuntimeError(f"MMSU failed={failures}, missing_audio={missing_audio}")
        else:
            pass
    finally:
        stop_server(server)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--seedtts-only", action="store_true")
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--audio-single-request", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.output,
        arguments.checkpoint,
        arguments.gpu_index,
        arguments.concurrency,
        arguments.seedtts_only,
        arguments.cache_directory,
        arguments.audio_single_request,
    )
else:
    pass
