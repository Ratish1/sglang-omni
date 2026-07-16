# SPDX-License-Identifier: Apache-2.0
"""Replay a deterministic production TTS manifest against one existing worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

from benchmarks.benchmarker.data import RequestResult
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts import (
    _handle_non_streaming_response,
    _handle_raw_pcm_streaming_response,
    _parse_response_headers,
    save_speed_results,
)

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class ManifestRequest:
    request_id: str
    arrival_offset_s: float
    payload: dict[str, Any]


def load_manifest(path: Path, *, server_max_new_tokens: int) -> list[ManifestRequest]:
    requests: list[ManifestRequest] = []
    seen_ids: set[str] = set()
    previous_offset = 0.0
    with path.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            unknown = sorted(set(row) - {"id", "arrival_offset_s", "payload"})
            if unknown:
                raise ValueError(
                    f"{path}:{line_number}: unknown row keys: {', '.join(unknown)}"
                )
            request_id = row.get("id")
            if not isinstance(request_id, str) or SAFE_ID.fullmatch(request_id) is None:
                raise ValueError(
                    f"{path}:{line_number}: id must match {SAFE_ID.pattern!r}"
                )
            if request_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate id {request_id!r}")
            seen_ids.add(request_id)

            offset = row.get("arrival_offset_s", 0)
            if (
                not isinstance(offset, (int, float))
                or isinstance(offset, bool)
                or offset < 0
            ):
                raise ValueError(
                    f"{path}:{line_number}: arrival_offset_s must be non-negative"
                )
            offset = float(offset)
            if offset < previous_offset:
                raise ValueError(
                    f"{path}:{line_number}: arrival offsets must be non-decreasing"
                )
            previous_offset = offset

            payload = row.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: payload must be an object")
            if (
                not isinstance(payload.get("input"), str)
                or not payload["input"].strip()
            ):
                raise ValueError(
                    f"{path}:{line_number}: payload.input must be non-empty"
                )
            stream_response = payload.get("stream", False)
            if not isinstance(stream_response, bool):
                raise ValueError(
                    f"{path}:{line_number}: payload.stream must be boolean"
                )
            response_format = payload.get(
                "response_format", "pcm" if stream_response else "wav"
            )
            expected_format = "pcm" if stream_response else "wav"
            if response_format != expected_format:
                raise ValueError(
                    f"{path}:{line_number}: response_format must be {expected_format!r} "
                    f"when stream={stream_response}"
                )
            max_new_tokens = payload.get("max_new_tokens", server_max_new_tokens)
            if (
                not isinstance(max_new_tokens, int)
                or isinstance(max_new_tokens, bool)
                or not 1 <= max_new_tokens <= server_max_new_tokens
            ):
                raise ValueError(
                    f"{path}:{line_number}: payload.max_new_tokens must be in "
                    f"[1, {server_max_new_tokens}]"
                )
            normalized_payload = dict(payload)
            normalized_payload["stream"] = stream_response
            normalized_payload["response_format"] = response_format
            normalized_payload["max_new_tokens"] = max_new_tokens
            requests.append(ManifestRequest(request_id, offset, normalized_payload))
    if not requests:
        raise ValueError(f"{path}: manifest contains no requests")
    return requests


async def send_request(
    session: aiohttp.ClientSession,
    request: ManifestRequest,
    *,
    api_url: str,
    model: str,
    save_audio_dir: Path,
) -> RequestResult:
    result = RequestResult(
        request_id=request.request_id,
        text=request.payload["input"][:120],
    )
    payload = dict(request.payload)
    payload["model"] = model
    started = time.perf_counter()
    try:
        async with session.post(api_url, json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
            elif payload["stream"]:
                await _handle_raw_pcm_streaming_response(
                    response, result, started, str(save_audio_dir)
                )
                _parse_response_headers(result, response.headers)
            else:
                await _handle_non_streaming_response(
                    response, result, started, str(save_audio_dir)
                )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        result.error = str(exc)
    finally:
        result.latency_s = time.perf_counter() - started
    return result


async def run(args: argparse.Namespace, requests: list[ManifestRequest]) -> None:
    output_dir = args.output_dir.resolve()
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    api_url = f"http://{args.host}:{args.port}/v1/audio/speech"

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for request in requests[: args.warmup]:
            result = await send_request(
                session,
                request,
                api_url=api_url,
                model=args.model,
                save_audio_dir=audio_dir,
            )
            if not result.is_success:
                raise RuntimeError(
                    f"warmup {request.request_id} failed: {result.error}"
                )

        started = time.perf_counter()

        async def scheduled(request: ManifestRequest) -> RequestResult:
            delay = request.arrival_offset_s - (time.perf_counter() - started)
            if delay > 0:
                await asyncio.sleep(delay)
            async with semaphore:
                return await send_request(
                    session,
                    request,
                    api_url=api_url,
                    model=args.model,
                    save_audio_dir=audio_dir,
                )

        outputs = await asyncio.gather(*(scheduled(request) for request in requests))
        wall_clock_s = time.perf_counter() - started

    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    metrics = compute_speed_metrics(outputs, wall_clock_s=wall_clock_s)
    config = {
        "model": args.model,
        "base_url": f"http://{args.host}:{args.port}",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha256,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "server_max_new_tokens": args.server_max_new_tokens,
        "deterministic_arrival_offsets": True,
    }
    save_speed_results(outputs, metrics, config, str(output_dir))
    print(json.dumps({"summary": metrics, "config": config}, indent=2))
    if metrics.get("failed_requests"):
        raise RuntimeError(f"{metrics['failed_requests']} manifest requests failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--server-max-new-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.port not in range(1, 65536):
        parser.error("--port must be in [1, 65535]")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.server_max_new_tokens <= 0 or args.timeout <= 0:
        parser.error("token and timeout limits must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        requests = load_manifest(
            args.manifest, server_max_new_tokens=args.server_max_new_tokens
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    asyncio.run(run(args, requests))


if __name__ == "__main__":
    main()
