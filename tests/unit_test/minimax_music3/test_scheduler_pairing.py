# SPDX-License-Identifier: Apache-2.0
"""Contract tests for MiniMax Music 3 CFG pair enqueueing.

Every conditioned request enqueues an unconditioned twin directly behind it
in the waiting queue; when the conditioned row was not enqueued (aborted
before admission), the twin is dropped so no unpaired row can reach a batch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from sglang_omni.models.minimax_music3.scheduler import (  # noqa: E402
    MiniMaxMusic3Scheduler,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402


def _twin():
    ids = [1, 2, 3]
    return SimpleNamespace(origin_input_ids=ids, origin_input_ids_unpadded=ids)


class _StubScheduler:
    _normalize_req_token_arrays = staticmethod(
        OmniScheduler._normalize_req_token_arrays
    )

    def __init__(self) -> None:
        self.waiting_queue: list = []

    def enqueue_cfg_uncond(self, cond_req, uncond) -> None:
        MiniMaxMusic3Scheduler._enqueue_cfg_uncond(
            self, SimpleNamespace(req=cond_req), uncond
        )


def test_uncond_twin_enqueues_behind_its_conditioned_row() -> None:
    scheduler = _StubScheduler()
    cond_req = SimpleNamespace()
    scheduler.waiting_queue.append(cond_req)
    twin = _twin()
    uncond = SimpleNamespace(req=twin, is_cfg_uncond=True)

    scheduler.enqueue_cfg_uncond(cond_req, uncond)

    assert scheduler.waiting_queue == [cond_req, twin]
    assert twin._omni_data is uncond
    assert twin._omni_terminal_claimed is False


def test_uncond_twin_dropped_when_conditioned_row_was_not_enqueued() -> None:
    scheduler = _StubScheduler()
    twin = _twin()

    scheduler.enqueue_cfg_uncond(SimpleNamespace(), SimpleNamespace(req=twin))

    assert scheduler.waiting_queue == []
