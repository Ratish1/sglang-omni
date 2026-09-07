# SPDX-License-Identifier: Apache-2.0
"""Short Higgs traces or full SeedTTS C16 A/B against fresh owned servers."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import gzip
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterator

import numpy as np
import requests

from benchmarks.benchmarker.utils import wait_for_service
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.eval.asr_profiling import (
    build_stage_breakdown,
    collect_environment_fingerprint,
)
from benchmarks.eval.benchmark_tts_seedtts import (
    TtsSeedttsBenchmarkConfig,
    run_tts_seedtts_benchmark,
)

GUMBEL_ENV = "SGLANG_OMNI_HIGGS_USE_GUMBEL_SAMPLE"
PROFILE_ENV = "SGLANG_OMNI_HIGGS_PROFILE_SAMPLING"
MODEL = "bosonai/higgs-audio-v3-tts-4b"


def _write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _dataset_manifest(samples: list[SampleInput]) -> dict:
    entries = []
    for sample in samples:
        entries.append(
            {
                "id": sample.sample_id,
                "text": sample.target_text,
                "ref_text": sample.ref_text,
                "ref_audio_sha256": hashlib.sha256(
                    Path(sample.ref_audio).read_bytes()
                ).hexdigest(),
            }
        )
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return {"count": len(samples), "input_sha256": digest, "samples": entries}


def _resolve_checkpoint(model: str, revision: str | None) -> tuple[str, str | None]:
    local_path = Path(model)
    if local_path.is_dir():
        if revision is not None:
            raise ValueError("--model-revision only applies to a Hub repository")
        return str(local_path.resolve()), None
    from huggingface_hub import snapshot_download

    # Pin the resolved snapshot once for every fresh server in this campaign.
    snapshot = Path(snapshot_download(repo_id=model, revision=revision))
    return str(snapshot.resolve()), snapshot.name


def _server_command(args: argparse.Namespace, samples: list[SampleInput]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tts_engine.engine.max_running_requests",
        str(args.capacity),
        "--tts_engine.engine.cuda_graph_max_bs",
        str(args.capacity),
        "--tts_engine.engine.disable_cuda_graph",
        str(args.graph == "off").lower(),
    ]
    # The same local staged references are reused by every A/B leg.
    media_root = os.path.commonpath(
        [str(Path(sample.ref_audio).resolve().parent) for sample in samples]
    )
    command.extend(["--allowed-local-media-path", media_root])
    return command


def _stop_owned_server(process: subprocess.Popen) -> None:
    # start_new_session fixes the group id to this pid even if the leader exits.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        process.poll()  # Reap the direct child before checking group liveness.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


@contextlib.contextmanager
def _server(
    command: list[str], env: dict[str, str], output_dir: Path, port: int, timeout: int
) -> Iterator[None]:
    # Refuse to attach to an unrelated existing server on this port.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    log_path = output_dir / "server.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            env={**os.environ, **env},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_service(
                f"http://127.0.0.1:{port}",
                timeout,
                server_process=process,
                server_log_file=log_path,
                health_body_contains="healthy",
            )
            expected = f"Higgs sampling: gumbel_requested={env[GUMBEL_ENV] == '1'}"
            if expected not in log_path.read_text():
                raise RuntimeError(
                    f"Server did not confirm {expected!r}; inspect {log_path}"
                )
            yield
        finally:
            # Own only this process group, including startup failure/Ctrl-C.
            _stop_owned_server(process)


def _profile_control(base_url: str, endpoint: str, payload: dict) -> dict:
    with requests.Session() as session:
        session.trust_env = False
        response = session.post(f"{base_url}/{endpoint}", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()


async def _wait_recorders(event_dir: Path) -> list[Path]:
    # Higgs's default topology has two worker processes; the coordinator
    # recorder alone is not proof that either worker received ProfilerStart.
    groups = (("preprocessing", "audio_encoder"), ("tts_engine", "vocoder"))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        files = []
        for group in groups:
            matches = [
                path
                for stage in group
                for path in event_dir.glob(f"events_{stage}_*.jsonl")
            ]
            if matches:
                files.append(matches[0])
        if len(files) == len(groups):
            return files
        await asyncio.sleep(0.1)
    raise TimeoutError("Higgs worker recorders did not start; inspect server.log")


def _recorder_is_open(event_file: Path) -> bool:
    pid = event_file.stem.rsplit("_", 1)[1]
    descriptors = Path("/proc") / pid / "fd"
    if not descriptors.is_dir():
        raise RuntimeError(f"Profiled worker {pid} exited before export completed")
    for descriptor in descriptors.iterdir():
        try:
            target = os.readlink(descriptor)
        except FileNotFoundError:
            continue  # A worker can close an unrelated descriptor during this scan.
        if target == str(event_file):
            return True
    return False


def _check_gzip(path: Path) -> None:
    # Reading through EOF checks the gzip trailer without loading the trace.
    with gzip.open(path, "rb") as trace:
        while trace.read(1024 * 1024):
            pass


async def _wait_exports(output_dir: Path, event_files: list[Path]) -> list[str]:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        traces = []
        for event_file in event_files:
            if _recorder_is_open(event_file):
                break
            pid = event_file.stem.rsplit("_", 1)[1]
            matches = list(output_dir.glob(f"trace_pid{pid}_rank*.trace.json.gz"))
            # gzip removes the source only after successfully writing the trailer.
            if not matches or any(path.with_suffix("").exists() for path in matches):
                break
            traces.extend(matches)
        else:
            for path in traces:
                await asyncio.to_thread(_check_gzip, path)
            return [str(path) for path in traces]
        await asyncio.sleep(0.1)
    raise TimeoutError(
        "Worker trace export did not finish; raw artifacts remain in the output directory"
    )


async def _profile_pass(
    config: TtsSeedttsBenchmarkConfig, samples: list[SampleInput], seconds: float
) -> dict:
    output_dir = Path(config.output_dir)
    run_id = output_dir.name + "_" + output_dir.parent.name
    base_url = f"http://127.0.0.1:{config.port}"
    start = {
        "run_id": run_id,
        "enable_torch": True,
        "trace_path_template": str(output_dir / "trace"),
        "event_dir": str(output_dir / "events"),
    }
    timer = None
    event_files = []
    try:
        # Stop is also attempted if a start response is lost after activation.
        await asyncio.to_thread(_profile_control, base_url, "start_profile", start)
        event_files = await _wait_recorders(output_dir / "events")

        async def stop_after_window() -> None:
            await asyncio.sleep(seconds)
            await asyncio.to_thread(
                _profile_control, base_url, "stop_profile", {"run_id": run_id}
            )

        timer = asyncio.create_task(stop_after_window())
        results = await run_tts_seedtts_benchmark(config, samples=samples)
    finally:
        try:
            if timer is not None:
                if not timer.done():
                    timer.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await timer
        finally:
            await asyncio.to_thread(
                _profile_control, base_url, "stop_profile", {"run_id": run_id}
            )
            if event_files:
                # Retain completed traces even when request execution fails.
                traces = await _wait_exports(output_dir, event_files)
                _write_json(output_dir / "trace_files.json", {"traces": traces})
    return results


def _graph_coverage(event_dir: Path, backend: str) -> dict:
    counts: Counter = Counter()
    for path in event_dir.glob("*.jsonl"):
        with path.open() as events:
            for line in events:
                event = json.loads(line)
                if event["event_name"] != "higgs_sampling_forward":
                    continue
                metadata = event["metadata"]
                if metadata["use_gumbel"] != (backend == "gumbel"):
                    raise RuntimeError(
                        "Trace sampler mode disagrees with the run manifest"
                    )
                counts[
                    (
                        metadata["is_prefill"],
                        metadata["is_graph"],
                        metadata["live_requests"],
                        metadata["forward_batch_size"],
                    )
                ] += 1
    return {
        "has_forward_events": bool(counts),
        "has_decode_events": any(not key[0] for key in counts),
        "batches": [
            dict(
                zip(
                    ("is_prefill", "is_graph", "live_requests", "forward_batch_size"),
                    key,
                ),
                count=count,
            )
            for key, count in sorted(counts.items())
        ],
    }


def _require_success(results: dict, expected_count: int) -> None:
    summary = results["summary"]
    if summary["completed_requests"] != expected_count or summary["failed_requests"]:
        raise RuntimeError(
            f"Incomplete benchmark cohort: {summary}; retain artifacts and investigate"
        )


async def _run_leg(
    args: argparse.Namespace, samples: list[SampleInput], backend: str, output_dir: Path
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    is_profile = args.mode == "profile"
    env = {GUMBEL_ENV: str(int(backend == "gumbel")), PROFILE_ENV: str(int(is_profile))}
    command = _server_command(args, samples)
    config = TtsSeedttsBenchmarkConfig(
        model=args.model,
        meta=args.meta,
        port=args.port,
        host="127.0.0.1",
        ref_format="references",
        output_dir=str(output_dir),
        concurrency=args.concurrency,
        warmup=0,
        lang=args.lang,
        seed=args.seed,
        stream=args.stream,
        response_format="pcm" if args.stream else "wav",
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_running_requests=args.capacity,
        cuda_graph_max_bs=args.capacity,
    )
    manifest = {
        "mode": args.mode,
        "backend": backend,
        "instrumented": is_profile,
        "server_command": command,
        "server_env": env,
        "graph_requested": args.graph,
        "concurrency": args.concurrency,
        "capacity": args.capacity,
        "request_count": len(samples),
        "warmup_requests": args.warmup,
        "stream": args.stream,
        "sampling": {
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "max_new_tokens": args.max_new_tokens,
        "environment": collect_environment_fingerprint(args.model),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(shlex.join(command), flush=True)
    with _server(command, env, output_dir, args.port, args.server_timeout):
        warmup = await run_tts_seedtts_benchmark(
            replace(config, output_dir=str(output_dir / "warmup")),
            samples=[samples[0]] * args.warmup,
            save_audio=False,
        )
        _require_success(warmup, args.warmup)
        if is_profile:
            results = await _profile_pass(config, samples, args.profile_seconds)
            events = output_dir / "events"
            _write_json(
                output_dir / "stage_breakdown.json", build_stage_breakdown(str(events))
            )
            coverage = _graph_coverage(events, backend)
            _write_json(output_dir / "graph_coverage.json", coverage)
            if not coverage["has_decode_events"]:
                raise RuntimeError(
                    "Trace has no Higgs decode events; inspect coverage and increase the window"
                )
        else:
            results = await run_tts_seedtts_benchmark(config, samples=samples)
        _require_success(results, len(samples))
    return {"summary": results["summary"], "path": str(output_dir)}


def _comparison(pairs: list[dict]) -> dict:
    fields = (
        "throughput_qps",
        "latency_mean_s",
        "latency_p95_s",
        "audio_throughput_s_per_s",
        "rtf_mean",
        "audio_ttfp_p95_s",
        "inter_chunk_p95_s",
    )
    report = {}
    for field in fields:
        changes = []
        for pair in pairs:
            baseline = pair["baseline"]["summary"].get(field)
            gumbel = pair["gumbel"]["summary"].get(field)
            if baseline is not None and gumbel is not None and baseline > 0:
                changes.append(100 * (gumbel / baseline - 1))
        if not changes:
            continue
        interval = None
        if len(changes) >= 3:
            rng = np.random.default_rng(20260907)
            means = rng.choice(changes, size=(10_000, len(changes)), replace=True).mean(
                axis=1
            )
            interval = np.percentile(means, [2.5, 97.5]).tolist()
        report[field] = {
            "paired_relative_change_percent": changes,
            "mean_relative_change_percent": float(np.mean(changes)),
            "bootstrap_mean_ci95_percent": interval,
        }
    return {
        "interpretation": "100 * (gumbel / baseline - 1); negative latency/RTF and positive throughput are improvements",
        "qualification": "Measurements only; audio quality and output-length effects require separate evaluation",
        "metrics": report,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "profile"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model", default=MODEL, help="Higgs checkpoint path or repo id"
    )
    parser.add_argument(
        "--model-revision", help="Resolve one Hub snapshot before the campaign"
    )
    parser.add_argument("--meta", default=SEEDTTS_DATASET_ID)
    parser.add_argument("--dataset-revision")
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument(
        "--repetitions", type=int, help="Defaults to 5 full pairs or 1 diagnostic pair"
    )
    parser.add_argument("--graph", choices=("on", "off"), default="on")
    parser.add_argument("--profile-requests", type=int, default=32)
    parser.add_argument("--profile-seconds", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-timeout", type=int, default=1200)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--stream", action="store_true")
    return parser


async def _run(args: argparse.Namespace) -> None:
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    revision = args.dataset_revision
    if revision is None and args.meta == SEEDTTS_DATASET_ID:
        revision = SEEDTTS_DATASET_REVISION
    # Only the profiling mode takes a subset. Performance always loads all rows.
    samples = load_seedtts_samples(
        args.meta,
        args.profile_requests if args.mode == "profile" else None,
        split=args.lang,
        revision=revision,
    )
    if not samples:
        raise ValueError("The dataset contains no samples")
    manifest = _dataset_manifest(samples)
    manifest.update(
        source=args.meta, revision=revision, split=args.lang, mode=args.mode
    )
    _write_json(args.output_dir / "dataset.json", manifest)
    model_source = args.model
    args.model, model_revision = _resolve_checkpoint(model_source, args.model_revision)
    _write_json(
        args.output_dir / "checkpoint.json",
        {
            "source": model_source,
            "requested_revision": args.model_revision,
            "resolved_revision": model_revision,
            "path": args.model,
        },
    )
    pairs = []
    for pair in range(args.repetitions):
        order = ("baseline", "gumbel") if pair % 2 == 0 else ("gumbel", "baseline")
        results = {"pair": pair + 1, "order": list(order)}
        for backend in order:
            print(
                f"{args.mode}: pair={pair + 1} backend={backend} requests={len(samples)}",
                flush=True,
            )
            results[backend] = await _run_leg(
                args,
                samples,
                backend,
                args.output_dir / f"pair_{pair + 1:02d}" / backend,
            )
        pairs.append(results)
        _write_json(args.output_dir / "pairs.json", pairs)
        if args.mode == "benchmark":
            _write_json(args.output_dir / "comparison.json", _comparison(pairs))


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.mode == "profile" and (
        not Path("/proc").is_dir() or shutil.which("gzip") is None
    ):
        parser.error(
            "profile mode requires Linux /proc access and gzip in the local container"
        )
    if args.repetitions is None:
        args.repetitions = 5 if args.mode == "benchmark" else 1
    if (
        min(
            args.concurrency,
            args.capacity,
            args.repetitions,
            args.profile_requests,
            args.warmup,
            args.server_timeout,
            args.max_new_tokens,
        )
        < 1
    ):
        parser.error("counts and timeouts must be positive")
    if args.profile_seconds <= 0 or not np.isfinite(args.profile_seconds):
        parser.error("--profile-seconds must be finite and positive")
    if args.concurrency > args.capacity:
        parser.error(
            "--concurrency must be <= --capacity for this closed-loop comparison"
        )
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
