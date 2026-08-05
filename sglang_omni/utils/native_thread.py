# SPDX-License-Identifier: Apache-2.0
"""Native thread naming for Linux system-level observability."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

_LINUX_COMM_MAX_BYTES = 15
_THREAD_SELF_COMM = Path("/proc/thread-self/comm")


def linux_comm_name(name: str) -> str:
    """Return a valid Linux ``comm`` label without splitting UTF-8."""
    normalized = " ".join(str(name).split()) or "python"
    encoded = normalized.encode("utf-8")[:_LINUX_COMM_MAX_BYTES]
    while encoded:
        try:
            return encoded.decode("utf-8")
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return "python"


def set_native_thread_name(name: str | None = None) -> str | None:
    """Best-effort name the calling Linux thread and return the applied label."""
    if not sys.platform.startswith("linux"):
        return None
    label = linux_comm_name(name or threading.current_thread().name)
    try:
        _THREAD_SELF_COMM.write_text(label, encoding="utf-8")
    except OSError:
        return None
    return label


__all__ = ["linux_comm_name", "set_native_thread_name"]
