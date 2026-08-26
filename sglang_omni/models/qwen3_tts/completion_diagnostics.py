# SPDX-License-Identifier: Apache-2.0
"""Opt-in completion artifacts for Qwen3-TTS repetition diagnostics.

The autoregressive loop does not call this module. Completed requests enqueue
their already-host-resident semantic IDs and codec codes for a background JSONL
writer, keeping filesystem I/O out of generation and scheduler terminalization.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

COMPLETION_DIAGNOSTICS_DIR_ENV = "SGLANG_OMNI_QWEN3_TTS_COMPLETION_DIAGNOSTICS_DIR"
COMPLETION_DIAGNOSTICS_RUN_LABEL_ENV = (
    "SGLANG_OMNI_QWEN3_TTS_COMPLETION_DIAGNOSTICS_RUN_LABEL"
)

_STOP = object()
_WRITER_LOCK = threading.Lock()
_WRITER: _CompletionDiagnosticsWriter | None = None
logger = logging.getLogger(__name__)


def completion_diagnostics_enabled() -> bool:
    """Return whether completion capture is enabled for this process."""

    return bool(os.environ.get(COMPLETION_DIAGNOSTICS_DIR_ENV))


def _safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:96] or "run"


def _as_nested_ints(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_as_nested_ints(item) for item in value]
    return int(value)


def _sequence_sha256(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _materialize_record(item: Mapping[str, Any]) -> dict[str, Any]:
    semantic_token_ids = _as_nested_ints(item["semantic_token_ids"])
    generated_codec_codes = _as_nested_ints(item["generated_codec_codes"])
    metadata = dict(item["metadata"])
    metadata.update(
        {
            "schema_version": 1,
            "record_type": "qwen3_tts_completion",
            "semantic_token_ids": semantic_token_ids,
            "semantic_token_sha256": _sequence_sha256(semantic_token_ids),
            "generated_codec_codes": generated_codec_codes,
            "generated_codec_codes_sha256": _sequence_sha256(generated_codec_codes),
            "sequence_hash_encoding": "canonical_json_v1",
        }
    )
    return metadata


class _CompletionDiagnosticsWriter:
    def __init__(self, output_dir: Path, *, run_label: str) -> None:
        self.output_dir = output_dir
        self.run_label = _safe_label(run_label)
        self.path = output_dir / (
            f"qwen3-tts-completions-{self.run_label}-{os.getpid()}-"
            f"{time.time_ns()}-{uuid.uuid4().hex[:8]}.jsonl"
        )
        self._queue: queue.SimpleQueue[Mapping[str, Any] | object] = queue.SimpleQueue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="qwen3-tts-completion-diagnostics",
            daemon=True,
        )
        self._thread.start()

    def submit(
        self,
        *,
        metadata: Mapping[str, Any],
        semantic_token_ids: Any,
        generated_codec_codes: Any,
    ) -> None:
        if self._closed:
            logger.error("Qwen3-TTS completion diagnostics writer is closed")
            return
        self._queue.put(
            {
                "metadata": dict(metadata),
                "semantic_token_ids": semantic_token_ids,
                "generated_codec_codes": generated_codec_codes,
            }
        )

    def close(self, *, timeout_s: float = 10.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            logger.error(
                "Timed out draining Qwen3-TTS completion diagnostics to %s",
                self.path,
            )

    def _run(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8") as output:
                logger.info("Writing Qwen3-TTS completion diagnostics to %s", self.path)
                while True:
                    item = self._queue.get()
                    if item is _STOP:
                        return
                    record = _materialize_record(item)
                    output.write(
                        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
                    )
                    output.flush()
        except Exception:
            logger.exception(
                "Qwen3-TTS completion diagnostics writer failed for %s", self.path
            )


def _get_writer() -> _CompletionDiagnosticsWriter | None:
    output_dir = os.environ.get(COMPLETION_DIAGNOSTICS_DIR_ENV)
    if not output_dir:
        return None

    global _WRITER
    with _WRITER_LOCK:
        if _WRITER is None:
            run_label = os.environ.get(
                COMPLETION_DIAGNOSTICS_RUN_LABEL_ENV,
                "unlabeled",
            )
            _WRITER = _CompletionDiagnosticsWriter(
                Path(output_dir).expanduser().resolve(),
                run_label=run_label,
            )
        return _WRITER


def record_qwen3_tts_completion(
    *,
    metadata: Mapping[str, Any],
    semantic_token_ids: Any,
    generated_codec_codes: Any,
) -> None:
    """Queue one completed request without blocking on filesystem writes."""

    try:
        writer = _get_writer()
        if writer is not None:
            writer.submit(
                metadata=metadata,
                semantic_token_ids=semantic_token_ids,
                generated_codec_codes=generated_codec_codes,
            )
    except Exception:
        logger.exception("Failed to enqueue Qwen3-TTS completion diagnostics")


def close_completion_diagnostics() -> None:
    """Drain and close the process-local completion writer, if any."""

    global _WRITER
    with _WRITER_LOCK:
        writer = _WRITER
        _WRITER = None
    if writer is not None:
        writer.close()


atexit.register(close_completion_diagnostics)


__all__ = [
    "COMPLETION_DIAGNOSTICS_DIR_ENV",
    "COMPLETION_DIAGNOSTICS_RUN_LABEL_ENV",
    "close_completion_diagnostics",
    "completion_diagnostics_enabled",
    "record_qwen3_tts_completion",
]
