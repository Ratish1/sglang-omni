#!/usr/bin/env python3
"""Canonical SeedTTS generation: full unprofiled A/B or small named NSYS window.

Run from the Omni checkout with its Python environment. This does not launch a
server. --session controls ONLY the named NSYS session already hosting the server.
No unit tests, model imports or profiler runs are performed by --help.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def wait_healthy(base_url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(base_url.rstrip("/") + "/health", timeout=2) as response:
                if response.status == 200 and b"healthy" in response.read():
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    raise RuntimeError("Server did not become healthy before the readiness deadline")


def nsys(action, session, output, metrics_devices="none"):
    command = ["nsys", action, f"--session={session}"]
    if action == "start":
        command += [
            f"--output={output.resolve() / 'trace'}",
            f"--gpu-metrics-devices={metrics_devices}",
        ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    (output / f"nsys_{action}.log").write_text(
        json.dumps(command) + "\n" + completed.stdout + completed.stderr
    )
    if completed.returncode:
        raise RuntimeError(
            f"NSYS {action} failed; inspect {output / ('nsys_' + action + '.log')}"
        )


async def run(args):
    # Import only after argument validation; use this checkout's native benchmark.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
    from benchmarks.dataset.seedtts import load_seedtts_samples
    from benchmarks.eval.benchmark_tts_seedtts import (
        TtsSeedttsBenchmarkConfig,
        run_tts_seedtts_benchmark,
    )

    options = (
        json.loads(args.generation_json.read_text()) if args.generation_json else {}
    )
    allowed = {
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "seed",
        "voice",
        "task_type",
        "instructions",
        "initial_codec_chunk_frames",
    }
    if not isinstance(options, dict) or set(options) - allowed:
        raise ValueError(
            f"Generation JSON must be an object with fields from {sorted(allowed)}"
        )
    local_meta = Path(args.meta).is_file() or args.meta.endswith(".lst")
    revision = args.dataset_revision
    if not local_meta and revision is None:
        if args.meta != SEEDTTS_DATASET_ID:
            raise ValueError(
                "Pin --dataset-revision when using a different remote corpus"
            )
        revision = SEEDTTS_DATASET_REVISION
    samples = load_seedtts_samples(
        args.meta,
        None if args.samples is None else args.offset + args.samples,
        split=args.lang,
        revision=revision,
    )[args.offset :]
    if not samples:
        raise ValueError("Selected corpus is empty")
    if args.samples is not None and len(samples) != args.samples:
        raise ValueError(f"Requested {args.samples} samples, got {len(samples)}")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("Duplicate sample IDs would overwrite native audio artifacts")
    manifest = []
    for sample in samples:
        ref = Path(sample.ref_audio)
        manifest.append(
            {
                "sample_id": sample.sample_id,
                "target_text_sha256": digest(sample.target_text.encode()),
                "reference_text_sha256": digest(sample.ref_text.encode()),
                "reference_audio_sha256": digest(ref.read_bytes()),
            }
        )
    corpus_hash = digest(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    )
    config = TtsSeedttsBenchmarkConfig(
        model=args.model,
        meta=args.meta,
        base_url=args.base_url,
        output_dir=str(args.output / "measured"),
        lang=args.lang,
        max_samples=args.samples,
        sample_offset=args.offset,
        concurrency=args.concurrency,
        warmup=0 if args.session else args.warmup,
        stream=args.mode == "streaming",
        response_format="pcm" if args.mode == "streaming" else "wav",
        voice_clone=True,
        no_ref_text=args.reference == "audio",
        disable_tqdm=True,
        **options,
    )
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "argv": sys.argv,
        "dataset_revision": None if local_meta else revision,
        "local_meta_sha256": (
            digest(Path(args.meta).read_bytes()) if local_meta else None
        ),
        "ordered_inputs_sha256": corpus_hash,
        "sample_count": len(samples),
        "full_split": args.samples is None and args.offset == 0,
        "client_config": asdict(config),
        "profiled": bool(args.session),
        "nsys_session": args.session,
        "nsys_metrics_devices": args.metrics_devices,
        "note": "Client config max_running_requests/cuda_graph_max_bs are unused existing-server fields; attach actual resolved server config separately.",
        "warmup": {
            "requests": args.warmup,
            "reference": "first selected sample repeated",
            "separate_client_session": bool(args.session),
        },
        "status": "prepared",
    }
    write_json(args.output / "inputs.json", manifest)
    write_json(args.output / "experiment.json", record)
    wait_healthy(args.base_url, args.ready_timeout)
    started = False
    try:
        if args.session:
            if args.warmup:
                warm = replace(
                    config,
                    output_dir=str(args.output / "warmup"),
                    max_samples=args.warmup,
                    sample_offset=0,
                )
                warm_result = await run_tts_seedtts_benchmark(
                    warm, samples=[samples[0]] * args.warmup, save_audio=False
                )
                if any(not r["is_success"] for r in warm_result["per_request"]):
                    raise RuntimeError("Warmup request failed; capture not started")
            nsys("start", args.session, args.output, args.metrics_devices)
            started = True
        result = await run_tts_seedtts_benchmark(config, samples=samples)
        record["status"] = (
            "complete"
            if all(r["is_success"] for r in result["per_request"])
            else "request_failures"
        )
    except BaseException as error:
        record["status"], record["error"] = "failed", str(error)
        raise
    finally:
        try:
            if started:
                nsys("stop", args.session, args.output)
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(args.output / "experiment.json", record)
    if record["status"] != "complete":
        raise RuntimeError("Measured requests failed; inspect native artifacts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument(
        "--dataset-revision",
        help="Defaults to the repository's pinned canonical SeedTTS revision",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=("streaming", "buffered"), required=True)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument(
        "--reference", choices=("audio_text", "audio"), default="audio_text"
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--samples", type=int, help="Omit for the full selected split")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--session", help="Existing named NSYS server session; requires --samples"
    )
    parser.add_argument(
        "--metrics-devices",
        default="none",
        help="NSYS metric GPU IDs (not CUDA ordinals); use 0 only after checking nsys start --gpu-metrics-devices=help",
    )
    parser.add_argument("--generation-json", type=Path)
    parser.add_argument("--ready-timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.warmup < 0
        or args.offset < 0
        or (args.samples is not None and args.samples < 1)
    ):
        parser.error("Concurrency/samples must be positive; warmup/offset nonnegative")
    if args.session and args.samples is None:
        parser.error("Profiling requires an explicit small --samples cohort")
    if args.ready_timeout <= 0:
        parser.error("Readiness timeout must be positive")
    if args.output.exists():
        parser.error("Output already exists; use a fresh run directory")
    args.output.mkdir(parents=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
