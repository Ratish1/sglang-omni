# SPDX-License-Identifier: Apache-2.0
"""Run full SeedTTS against one uninstrumented BF16 colocated server."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import httpx

from benchmarks.benchmarker.utils import (
    get_wav_duration,
    start_server_from_cmd,
    stop_server,
)
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    run_omni_seedtts_benchmark,
)
from sglang_omni.utils import find_available_port
from tests.test_model.test_qwen3_omni_tts_ci import CONCURRENCY

logger = logging.getLogger(__name__)
FULL_SEEDTTS_SAMPLE_COUNT = 1088
SERVER_TIMEOUT_SECONDS = 600


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voice-probe", action="store_true")
    parser.add_argument("--profile", choices=("h100_bf16", "h200"), default="h200")
    parser.add_argument("--held-request", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--concurrency", type=int, default=CONCURRENCY, choices=(1, CONCURRENCY)
    )
    parser.add_argument("--sample-ids", nargs="+")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tree = arguments.tree.resolve(strict=True)
    checkpoint = arguments.checkpoint.resolve(strict=True)
    output = arguments.output.resolve()
    output.mkdir()

    dataset = DATASETS["seedtts"]
    samples = load_seedtts_samples(dataset)
    assert len(samples) == FULL_SEEDTTS_SAMPLE_COUNT
    if arguments.held_request is None:
        held_request = None
    else:
        held_request = json.loads(arguments.held_request.read_bytes())
        (previous_audio_path,) = held_request["audios"]
        sample_id = Path(previous_audio_path).stem
        (held_sample,) = [sample for sample in samples if sample.sample_id == sample_id]
        held_request["audios"] = [held_sample.ref_audio]
        (output / "held-request.json").write_text(json.dumps(held_request, indent=2))
    if arguments.sample_ids is not None:
        sample_ids = set(arguments.sample_ids)
        samples = [sample for sample in samples if sample.sample_id in sample_ids]
        assert {sample.sample_id for sample in samples} == sample_ids
        rows = [
            (sample.sample_id, sample.ref_text, sample.ref_audio, sample.target_text)
            for sample in samples
        ]
        assert all(
            "|" not in field and "\n" not in field for row in rows for field in row
        )
        metadata = output / "witnesses.lst"
        metadata.write_text("".join("|".join(row) + "\n" for row in rows))
        dataset = str(metadata)
    else:
        pass
    (output / "inputs.json").write_text(
        json.dumps(
            {
                "serving_revision": subprocess.check_output(
                    ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
                ).strip(),
                "checkpoint": checkpoint.name,
                "profile": arguments.profile,
                "seed": arguments.seed,
                "concurrency": arguments.concurrency,
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "text": sample.target_text,
                        "reference_audio_sha256": hashlib.sha256(
                            Path(sample.ref_audio).read_bytes()
                        ).hexdigest(),
                    }
                    for sample in samples
                ],
            },
            indent=2,
        )
    )
    port = find_available_port()
    server = start_server_from_cmd(
        [
            sys.executable,
            "-P",
            "-m",
            "sglang_omni.cli",
            "serve",
            "--config",
            str(
                tree / f"examples/configs/qwen3_omni_colocated_{arguments.profile}.yaml"
            ),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            str(checkpoint),
            "--model-name",
            "qwen3-omni",
        ],
        output / "server.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        env={"PYTHONPATH": str(tree)},
        strip_proxy=True,
    )
    try:
        if held_request is None:
            pass
        else:
            response = httpx.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=held_request,
                timeout=SERVER_TIMEOUT_SECONDS,
                trust_env=False,
            )
            response.raise_for_status()
            completion = response.json()
            assert completion["id"] == f"chatcmpl-{held_request['request_id']}"
            (choice,) = completion["choices"]
            waveform = base64.b64decode(
                choice["message"]["audio"]["data"], validate=True
            )
            (output / "held.wav").write_bytes(waveform)
            (output / "held-response.json").write_text(
                json.dumps(
                    {
                        "sample_id": held_sample.sample_id,
                        "request_id": held_request["request_id"],
                        "reference_audio_sha256": hashlib.sha256(
                            Path(held_sample.ref_audio).read_bytes()
                        ).hexdigest(),
                        "text": choice["message"]["content"],
                        "finish_reason": choice["finish_reason"],
                        "waveform_sha256": hashlib.sha256(waveform).hexdigest(),
                        "audio_duration_s": get_wav_duration(waveform),
                    },
                    indent=2,
                )
            )
        config = OmniSeedttsBenchmarkConfig(
            model="qwen3-omni",
            host="127.0.0.1",
            port=port,
            meta=dataset,
            voice_clone=True,
            max_concurrency=arguments.concurrency,
            output_dir=str(output / "generation"),
            seed=arguments.seed,
            disable_tqdm=True,
        )
        report = asyncio.run(run_omni_seedtts_benchmark(config))
        assert report["summary"]["completed_requests"] == len(samples)
        assert report["summary"]["failed_requests"] == 0
        logger.info(f"{output.name}: {report['summary']}")
        if arguments.voice_probe:
            subprocess.run(
                [
                    sys.executable,
                    "-P",
                    "-m",
                    "scripts.audit_qwen3_omni_voice_stream",
                    "--server-url",
                    f"http://127.0.0.1:{port}",
                    "--output",
                    str(output / "voice-stream.json"),
                ],
                check=True,
            )
        else:
            pass
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
