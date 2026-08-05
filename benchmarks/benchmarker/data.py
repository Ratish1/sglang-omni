# SPDX-License-Identifier: Apache-2.0
"""Shared data structures for the benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestResult:
    request_id: str = ""
    text: str = ""
    is_success: bool = False
    latency_s: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0
    tok_per_s: float = 0.0
    wav_path: str = ""
    error: str = ""
    audio_ttfp_s: float | None = None
    inter_chunk_s: list[float] = field(default_factory=list)
    text_ttft_s: float | None = None
    audio_chunk_count: int = 0
    first_audio_payload_bytes: int = 0
    http_status: int | None = None
    server_request_id: str | None = None
    client_scheduled_arrival_ns: int | None = None
    client_task_created_ns: int | None = None
    client_permit_wait_start_ns: int | None = None
    client_permit_acquired_ns: int | None = None
    client_send_invoked_ns: int | None = None
    client_http_start_ns: int | None = None
    client_http_response_start_ns: int | None = None
    client_response_complete_ns: int | None = None
