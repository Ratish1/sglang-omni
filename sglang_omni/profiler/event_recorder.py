# SPDX-License-Identifier: Apache-2.0
"""Request-level event recorder.

Each process appends events to ``<dir>/events_<stage>_<pid>.jsonl``; the
views layer merges files by ``request_id``. Kept free of sglang-omni
imports so it can be loaded from any process without circular risk.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Active-stage binding used when ``emit(stage=None)`` is called from code
# that can't plumb the stage name down (preprocessor, encoder callable,
# scheduler internals). Stage._run_scheduler binds the active stage on
# the scheduler thread; the contextvar propagates through
# ``asyncio.to_thread`` / ``loop.run_in_executor``, the thread-local
# covers plain ``threading.Thread`` workers.

_thread_active_stage = threading.local()
_active_stage_cv: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sglang_omni_active_stage", default=None
)


def set_active_stage(stage: str | None) -> contextvars.Token:
    """Bind ``stage`` for this thread / task. Returns a Token for reset."""
    _thread_active_stage.stage = stage
    return _active_stage_cv.set(stage)


def reset_active_stage(token: contextvars.Token | None) -> None:
    """Undo :func:`set_active_stage`. ``token=None`` clears the binding."""
    if token is not None:
        _active_stage_cv.reset(token)
    else:
        _active_stage_cv.set(None)
    _thread_active_stage.stage = None


def get_active_stage() -> str | None:
    """Active stage for this thread / task, contextvar first."""
    stage = _active_stage_cv.get()
    if stage is not None:
        return stage
    return getattr(_thread_active_stage, "stage", None)


@dataclass(frozen=True)
class RequestEvent:
    """A single point-in-time profiling event for one request."""

    request_id: str
    stage: str
    event_name: str
    timestamp_ns: int
    run_id: str | None = None
    pid: int | None = None
    native_tid: int | None = None
    thread_name: str | None = None
    monotonic_ns: int | None = None
    source_sequence: int | None = None
    clock_domain: str = "CLOCK_MONOTONIC"
    host_boot_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _RecorderSession:
    run_id: str
    primary_stage: str
    stages: set[str]
    final_path: Path
    partial_path: Path
    fp: Any
    records: queue.Queue[Any]
    generation: int
    accepting: bool = True
    enqueued_events: int = 0
    written_events: int = 0
    dropped_queue_full: int = 0
    dropped_write_error: int = 0
    max_queue_depth: int = 0
    writer_error: str | None = None
    finalized: bool = False
    bytes: int | None = None
    sha256: str | None = None
    writer: threading.Thread | None = None
    writer_gate: threading.Event = field(default_factory=threading.Event)
    deferred_writes: bool = False
    # Serializes the very small accepting-check + queue publication boundary.
    # JSON serialization and file I/O never run under this lock.
    publish_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def dropped_events(self) -> int:
        return self.dropped_queue_full + self.dropped_write_error


_STOP = object()
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class RequestEventRecorder:
    """Bounded asynchronous process-local JSONL event sink.

    Producers only timestamp, shallow-copy metadata, and publish to a bounded
    queue.  A dedicated writer owns serialization and disk I/O.  The file is
    written as ``.partial`` and atomically renamed only after a successful
    stop/flush, so a visible ``.jsonl`` file is a finalization signal.
    """

    def __init__(
        self,
        *,
        queue_capacity: int | None = None,
        defer_writes: bool | None = None,
        finalize_timeout_s: float | None = None,
    ) -> None:
        self._lock = threading.Lock()
        configured_capacity = os.environ.get(
            "SGLANG_OMNI_EVENT_QUEUE_CAPACITY", "131072"
        )
        self._queue_capacity = (
            int(configured_capacity) if queue_capacity is None else int(queue_capacity)
        )
        if self._queue_capacity < 1:
            raise ValueError("event recorder queue capacity must be positive")
        configured_defer = (
            os.environ.get("SGLANG_OMNI_EVENT_DEFER_WRITES", "").strip() == "1"
        )
        self._defer_writes = (
            configured_defer if defer_writes is None else bool(defer_writes)
        )
        configured_timeout = os.environ.get(
            "SGLANG_OMNI_EVENT_FINALIZE_TIMEOUT_S", "180"
        )
        self._finalize_timeout_s = (
            float(configured_timeout)
            if finalize_timeout_s is None
            else float(finalize_timeout_s)
        )
        if self._finalize_timeout_s <= 0:
            raise ValueError("event recorder finalize timeout must be positive")
        self._session: _RecorderSession | None = None
        self._last_snapshot: dict[str, Any] | None = None
        self._generation = 0
        self._source_state = threading.local()
        self._pid: int = os.getpid()

    # ---- lifecycle -----------------------------------------------------

    def is_active(self) -> bool:
        session = self._session
        return session is not None and session.accepting

    def active_run_id(self) -> str | None:
        session = self._session
        return None if session is None else session.run_id

    def active_path(self) -> str | None:
        session = self._session
        if session is not None:
            return str(session.final_path)
        if self._last_snapshot is not None:
            path = self._last_snapshot.get("path")
            return str(path) if path is not None else None
        return None

    def snapshot(self) -> dict[str, Any]:
        """Return lifecycle and drop state for profiler acknowledgements."""
        with self._lock:
            session = self._session
            if session is None:
                return dict(
                    self._last_snapshot
                    or {
                        "active": False,
                        "schema_version": 2,
                        "run_id": None,
                        "path": None,
                        "partial_path": None,
                        "pid": self._pid,
                        "stages": [],
                        "queue_capacity": self._queue_capacity,
                        "queue_depth": 0,
                        "max_queue_depth": 0,
                        "enqueued_events": 0,
                        "written_events": 0,
                        "dropped_events": 0,
                        "dropped_queue_full": 0,
                        "dropped_write_error": 0,
                        "writer_error": None,
                        "finalized": False,
                        "bytes": None,
                        "sha256": None,
                    }
                )
            return self._snapshot_session(session)

    def _snapshot_session(self, session: _RecorderSession) -> dict[str, Any]:
        return {
            "active": session.accepting,
            "schema_version": 2,
            "run_id": session.run_id,
            "path": str(session.final_path),
            "partial_path": str(session.partial_path),
            "pid": self._pid,
            "stages": sorted(session.stages),
            "queue_capacity": self._queue_capacity,
            "deferred_writes": session.deferred_writes,
            "finalize_timeout_s": self._finalize_timeout_s,
            "queue_depth": session.records.qsize(),
            "max_queue_depth": session.max_queue_depth,
            "enqueued_events": session.enqueued_events,
            "written_events": session.written_events,
            "dropped_events": session.dropped_events,
            "dropped_queue_full": session.dropped_queue_full,
            "dropped_write_error": session.dropped_write_error,
            "writer_error": session.writer_error,
            "finalized": session.finalized,
            "bytes": session.bytes,
            "sha256": session.sha256,
        }

    def start(self, run_id: str, event_dir: str, stage: str) -> str:
        """Open (or join) the per-process JSONL file for ``run_id``.

        Co-located stages share one file per ``(run_id, pid)``; only a
        new ``run_id`` rotates. Returns the absolute path.
        """
        with self._lock:
            session = self._session
            if session is not None:
                if session.run_id == run_id and session.accepting:
                    session.stages.add(stage)
                    return str(session.final_path)
                logger.warning(
                    "RequestEventRecorder already active (run_id=%s); "
                    "rotating to run_id=%s",
                    session.run_id,
                    run_id,
                )
                self._stop_session_unlocked(session)

            directory = Path(event_dir).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            safe_stage = _SAFE_NAME_RE.sub("_", stage)
            final_path = directory / f"events_{safe_stage}_{self._pid}.jsonl"
            partial_path = final_path.with_suffix(final_path.suffix + ".partial")
            if final_path.exists() or partial_path.exists():
                raise FileExistsError(
                    "event artifact already exists; use a unique event directory: "
                    f"{final_path}"
                )
            fp = partial_path.open("x", buffering=1024 * 1024, encoding="utf-8")
            self._generation += 1
            session = _RecorderSession(
                run_id=run_id,
                primary_stage=stage,
                stages={stage},
                final_path=final_path,
                partial_path=partial_path,
                fp=fp,
                records=queue.Queue(maxsize=self._queue_capacity),
                generation=self._generation,
                deferred_writes=self._defer_writes,
            )
            if not session.deferred_writes:
                session.writer_gate.set()
            session.writer = threading.Thread(
                target=self._writer_main,
                args=(session,),
                name=f"omni-profile-writer-{self._pid}",
                daemon=True,
            )
            self._session = session
            self._last_snapshot = None
            session.writer.start()
            logger.info(
                "RequestEventRecorder started run_id=%s stage=%s path=%s",
                run_id,
                stage,
                final_path,
            )
            return str(final_path)

    def stop(self, *, run_id: str | None = None) -> str | None:
        """Close the active file. ``run_id=None`` stops any active session."""
        with self._lock:
            session = self._session
            if session is None:
                return None
            if run_id is not None and run_id != session.run_id:
                logger.warning(
                    "Ignoring RequestEventRecorder stop for run_id=%s; active run_id=%s",
                    run_id,
                    session.run_id,
                )
                return None
            path = str(session.final_path)
            self._stop_session_unlocked(session)
            return path

    def _stop_session_unlocked(self, session: _RecorderSession) -> None:
        with session.publish_lock:
            session.accepting = False
            session.writer_gate.set()
            # Blocking publication is safe here: the writer is concurrently
            # draining and this happens once per bounded capture.
            session.records.put(_STOP)
        writer = session.writer
        if writer is not None:
            writer.join(timeout=self._finalize_timeout_s)
            if writer.is_alive():
                session.writer_error = (
                    f"writer did not stop within {self._finalize_timeout_s:.1f} seconds"
                )
        if session.writer_error is None and not session.finalized:
            session.writer_error = "writer stopped without finalizing the artifact"
        self._last_snapshot = self._snapshot_session(session)
        self._session = None

    def _writer_main(self, session: _RecorderSession) -> None:
        try:
            session.writer_gate.wait()
            while True:
                item = session.records.get()
                try:
                    if item is _STOP:
                        break
                    session.fp.write(
                        json.dumps(item.to_dict(), default=_json_default) + "\n"
                    )
                    session.written_events += 1
                except Exception as exc:
                    session.dropped_write_error += 1
                    if session.writer_error is None:
                        session.writer_error = f"{type(exc).__name__}: {exc}"
                    logger.warning("RequestEventRecorder writer failed", exc_info=True)
                finally:
                    session.records.task_done()
            session.fp.flush()
            os.fsync(session.fp.fileno())
            session.fp.close()
            if session.writer_error is None:
                os.replace(session.partial_path, session.final_path)
                session.bytes = session.final_path.stat().st_size
                digest = hashlib.sha256()
                with session.final_path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                session.sha256 = digest.hexdigest()
                session.finalized = True
        except Exception as exc:
            session.writer_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "RequestEventRecorder failed to finalize cleanly", exc_info=True
            )
            try:
                session.fp.close()
            except Exception:
                logger.debug("Failed to close profiler event file", exc_info=True)

    # ---- emit ----------------------------------------------------------

    def emit(
        self,
        *,
        request_id: str,
        stage: str | None,
        event_name: str,
        metadata: Mapping[str, Any] | None = None,
        timestamp_ns: int | None = None,
    ) -> None:
        """Publish one event. No-op when inactive; queue overflow is counted."""
        session = self._session
        if session is None or not session.accepting:
            return
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        monotonic_ns = time.monotonic_ns()
        native_tid = threading.get_native_id()
        thread_name = threading.current_thread().name
        source_state = getattr(self._source_state, "state", None)
        if source_state is None or source_state[0] != session.generation:
            source_sequence = 1
            self._source_state.state = (session.generation, source_sequence)
        else:
            source_sequence = source_state[1] + 1
            self._source_state.state = (session.generation, source_sequence)
        if stage is None:
            stage = get_active_stage() or session.primary_stage or "unknown"
        event = RequestEvent(
            request_id=request_id,
            stage=stage,
            event_name=event_name,
            timestamp_ns=ts,
            run_id=session.run_id,
            pid=self._pid,
            native_tid=native_tid,
            thread_name=thread_name,
            monotonic_ns=monotonic_ns,
            source_sequence=source_sequence,
            host_boot_id=_read_host_boot_id(),
            metadata=dict(metadata) if metadata else {},
        )
        with session.publish_lock:
            if not session.accepting:
                return
            try:
                session.records.put_nowait(event)
                session.enqueued_events += 1
                queue_depth = session.records.qsize()
                session.max_queue_depth = max(session.max_queue_depth, queue_depth)
            except queue.Full:
                session.dropped_queue_full += 1
                if session.dropped_queue_full == 1:
                    logger.warning(
                        "RequestEventRecorder queue is full; dropping %s for %s",
                        event_name,
                        request_id,
                    )


def _json_default(obj: Any) -> Any:
    """Safe fallback for ``json.dumps``: summarise tensors, never materialise.

    Tensors / arrays return ``{__tensor_summary__, type, shape, dtype,
    device}``; 0-D variants serialise as plain scalars; everything else
    falls back to ``repr``.
    """
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            if len(shape) == 0 and hasattr(obj, "item"):
                return obj.item()
        except TypeError:
            # ``.shape`` without ``__len__`` — skip the 0-D fast path
            # and fall through to the summary serializer below.
            pass
        try:
            shape_list: Any = [int(d) for d in shape]
        except Exception:  # noqa: BLE001 - arbitrary tensor-like metadata
            shape_list = repr(shape)
        device = getattr(obj, "device", None)
        return {
            "__tensor_summary__": True,
            "type": type(obj).__name__,
            "shape": shape_list,
            "dtype": str(dtype),
            "device": str(device) if device is not None else None,
        }
    return repr(obj)


_RECORDER = RequestEventRecorder()


def get_recorder() -> RequestEventRecorder:
    """Return the process-local recorder singleton."""
    return _RECORDER


def emit(
    *,
    request_id: str,
    stage: str | None,
    event_name: str,
    metadata: Mapping[str, Any] | None = None,
    timestamp_ns: int | None = None,
) -> None:
    """Module-level shortcut for ``get_recorder().emit(...)``."""
    _RECORDER.emit(
        request_id=request_id,
        stage=stage,
        event_name=event_name,
        metadata=metadata,
        timestamp_ns=timestamp_ns,
    )


@functools.cache
def _read_host_boot_id() -> str | None:
    # Note: (Jiaxin Deng) constant for the process lifetime, and these events
    # are emitted from the scheduler loop while profiling is active, which is
    # exactly when extra syscalls would contaminate what is being measured.
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip() or None
    except OSError:
        return None


def _emit_model_path(event_name: str, request_id: str, **extra: str) -> None:
    if not _RECORDER.is_active():
        return
    emit(
        request_id=request_id,
        stage=None,
        event_name=event_name,
        metadata={
            "clock": "CLOCK_MONOTONIC",
            "host_boot_id": _read_host_boot_id(),
            "monotonic_ns": time.monotonic_ns(),
            **extra,
        },
    )


def emit_model_path_start(request_id: str) -> None:
    _emit_model_path("model_path_start", request_id)


def emit_model_path_end(request_id: str, *, status: str) -> None:
    _emit_model_path("model_path_end", request_id, status=status)
