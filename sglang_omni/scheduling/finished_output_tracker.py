# SPDX-License-Identifier: Apache-2.0
"""The KV reservation estimate from finished outputs.

SGLang admits a request only while the KV pool holds, for every running
request, min(max_new_tokens, CLIP_MAX_NEW_TOKENS) times new_token_ratio, its
guess at how much of the cap an output uses: 0.7 at first, decaying over 600
decode steps, raised after a retract. Omni's stages set caps far above their
typical output (a TTS talker emits tens of frames under a cap of thousands),
so the guess reserves most of the pool for outputs that never come and
admission stops well below max_running_requests. OmniScheduler measures the
fraction instead: it records every finished request here and pushes the
largest recent fraction into the tracker SGLang reads before every admission,
the way SGLang's own RecentPrefillBatchSizeTracker feeds max_prefill_bs.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS


class RecentFinishedOutputTracker:
    """Track the largest fraction of the clipped cap used by recent finished requests.

    The window is one cohort of the rows that share the pool at once, so each
    running row is reserved what the longest member of the previous cohort
    needed and an outlier leaves after one turnover of the batch. Until a
    request has finished the maximum is None and the scheduler leaves SGLang's
    guess in place. An output longer than any recent one takes SGLang's
    retract path like any other overrun.
    """

    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        self._recent_fractions: deque[float] = deque(maxlen=window_size)

    @property
    def max_fraction(self) -> float | None:
        return max(self._recent_fractions, default=None)

    def observe_finished(self, req: Any) -> None:
        cap = req.sampling_params.max_new_tokens
        if not cap:
            return
        clipped_cap = min(int(cap), CLIP_MAX_NEW_TOKENS)
        # An output that ran past the clip counts as the whole clipped cap.
        # SGLang keeps the ratio within (0, 1]: from_config clamps init and
        # min at 1.0, the post retract estimate clamps at 1.0, and the running
        # budget in PrefillAdder multiplies the unclipped cap by the ratio, so
        # a ratio above 1.0 would reserve more than the cap itself.
        used = min(len(req.output_ids), clipped_cap)
        self._recent_fractions.append(used / clipped_cap)
