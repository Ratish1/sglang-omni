# SPDX-License-Identifier: Apache-2.0
"""Metrics and result types for the TTS serving benchmark."""

from __future__ import annotations

import base64
import json
import struct
import time
from dataclasses import asdict, dataclass, field
from typing import Any

PCM_SAMPLE_RATE = 24000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
SSE_DATA_PREFIX = "data: "
SSE_DONE_MARKER = "data: [DONE]"


@dataclass
class ScenarioResult:
    scenario_id: str
    endpoint: str
    category: str
    stage_id: str | None = None
    load_mode: str | None = None
    load_concurrency: int | None = None
    status: str = "error"
    success: bool = False
    expected_success: bool = True
    http_status: int | None = None
    http_status_class: str | None = None
    latency_s: float = 0.0
    planned_start_s: float | None = None
    actual_start_s: float | None = None
    completed_s: float | None = None
    queue_wait_s: float | None = None
    ttfa_s: float | None = None
    inter_chunk_s: list[float] = field(default_factory=list)
    audio_bytes: int = 0
    request_bytes: int = 0
    response_bytes: int = 0
    batch_size: int | None = None
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    error_type: str | None = None
    error_class: str | None = None
    error: str | None = None
    capability: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    ws_event_counts: dict[str, int] = field(default_factory=dict)
    ws_close_reason: str | None = None
    was_cancelled: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def duration_from_audio_bytes(
    data: bytes,
    *,
    content_type: str | None = None,
    response_format: str | None = None,
    sample_rate: int = PCM_SAMPLE_RATE,
) -> float:
    fmt = (response_format or "").lower()
    ctype = (content_type or "").lower()
    if fmt == "pcm" or "audio/pcm" in ctype or ctype == "application/octet-stream":
        return len(data) / float(sample_rate * PCM_CHANNELS * PCM_SAMPLE_WIDTH)
    if len(data) > 44 and data[:4] == b"RIFF":
        return _wav_duration(data)
    return 0.0


def _wav_duration(data: bytes) -> float:
    try:
        sample_rate = struct.unpack_from("<I", data, 24)[0]
        channels = struct.unpack_from("<H", data, 22)[0]
        bits = struct.unpack_from("<H", data, 34)[0]
    except struct.error:
        return 0.0
    if sample_rate <= 0 or channels <= 0 or bits <= 0:
        return 0.0
    return max(len(data) - 44, 0) / float(sample_rate * channels * bits // 8)


def parse_sse_audio_event(line: str) -> tuple[bytes | None, dict[str, Any] | None]:
    if not line.startswith(SSE_DATA_PREFIX) or line == SSE_DONE_MARKER:
        return None, None
    event = json.loads(line[len(SSE_DATA_PREFIX) :])
    audio = event.get("audio")
    if not isinstance(audio, dict) or not audio.get("data"):
        return None, event
    return base64.b64decode(audio["data"]), event


def finish_timing(result: ScenarioResult, start: float) -> None:
    now = time.perf_counter()
    result.latency_s = now - start
    result.completed_s = now
    if result.audio_duration_s > 0:
        result.rtf = result.latency_s / result.audio_duration_s


def classify_http_status(status: int | None) -> str | None:
    if status is None:
        return None
    return f"{status // 100}xx"
