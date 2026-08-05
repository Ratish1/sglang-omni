# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

from sglang_omni.utils.native_thread import linux_comm_name, set_native_thread_name


def test_linux_comm_name_is_normalized_and_utf8_safe() -> None:
    assert linux_comm_name("scheduler-asr") == "scheduler-asr"
    assert linux_comm_name("omni-request-build_0") == "omni-request-bu"
    assert linux_comm_name("  audio   worker  ") == "audio worker"
    label = linux_comm_name("é" * 10)
    assert len(label.encode("utf-8")) <= 15
    assert label.encode("utf-8").decode("utf-8") == label


def test_set_native_thread_name_is_visible_in_procfs() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("requires Linux procfs")
    observed: list[tuple[str | None, str]] = []

    def worker() -> None:
        applied = set_native_thread_name("profile-worker")
        comm = Path("/proc/thread-self/comm").read_text(encoding="utf-8").strip()
        observed.append((applied, comm))

    thread = threading.Thread(target=worker, name="python-profile-worker")
    thread.start()
    thread.join()
    assert observed == [("profile-worker", "profile-worker")]
