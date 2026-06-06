# SPDX-License-Identifier: Apache-2.0
"""Request lifecycle helpers shared by staged schedulers."""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

PreparedT = TypeVar("PreparedT")


class PreparedRequestStore(Generic[PreparedT]):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prepared: dict[str, PreparedT] = {}
        self.inflight: set[str] = set()
        self.aborted: set[str] = set()

    def clear(self) -> None:
        with self.lock:
            self.clear_locked()

    def clear_locked(self) -> None:
        self.prepared.clear()
        self.inflight.clear()
        self.aborted.clear()

    def mark_inflight(self, request_id: str) -> None:
        with self.lock:
            self.inflight.add(str(request_id))

    def discard_inflight_after_error(self, request_id: str) -> None:
        rid = str(request_id)
        with self.lock:
            self.inflight.discard(rid)
            self.aborted.discard(rid)

    def publish(self, request_id: str, prepared: PreparedT) -> bool:
        """Publish prepared state unless this in-flight request was aborted."""
        rid = str(request_id)
        with self.lock:
            self.inflight.discard(rid)
            aborted = rid in self.aborted
            self.aborted.discard(rid)
            if not aborted:
                self.prepared[rid] = prepared
        return not aborted

    def pop(self, request_id: str) -> PreparedT | None:
        with self.lock:
            return self.prepared.pop(str(request_id), None)

    def cleanup(self, request_id: str) -> None:
        """Drop prepared state or tombstone an in-flight request for late publish."""
        rid = str(request_id)
        with self.lock:
            if self.prepared.pop(rid, None) is not None:
                return
            if rid in self.inflight:
                self.aborted.add(rid)


def prepared_marker_from_data(data: object, marker_key: str) -> str | None:
    if not isinstance(data, dict):
        return None
    marker = data.get(marker_key)
    return str(marker) if marker is not None else None
