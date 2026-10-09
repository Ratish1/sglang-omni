# SPDX-License-Identifier: Apache-2.0
"""Spawn target that reports the CPU mask of every thread in its process."""

from __future__ import annotations

import multiprocessing
import os
import threading
from multiprocessing.queues import Queue

if multiprocessing.parent_process() is not None:
    # note (Richard Wang): a thread started while the child imports its target,
    # before the target runs, as native thread pools are.
    threading.Thread(target=threading.Event().wait, daemon=True).start()
else:
    pass


def report_thread_cpus(
    spec: object, ready_event: object, startup_error_channel: Queue
) -> None:
    startup_error_channel.put(
        [
            sorted(os.sched_getaffinity(int(tid)))
            for tid in os.listdir("/proc/self/task")
        ]
    )
