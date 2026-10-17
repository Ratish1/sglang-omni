# SPDX-License-Identifier: Apache-2.0
"""The talker's decode graphs, one per draw: SGLang's decode graph runner captures every
batch size once per graph variant and picks a variant for each batch at replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)

from sglang_omni.models.qwen3_omni.components.talker import (
    FLASHINFER_DRAW,
    SEEDED_DRAW,
    Qwen3OmniTalker,
)
from sglang_omni.vendor.sglang.core import ForwardBatch


@dataclass(frozen=True)
class TalkerDrawGraphVariants:
    """A batch with a request seed replays the seeded PyTorch draw's graph, any other
    batch the FlashInfer draw's. The talker stages the batch's seeds before its forward,
    so select reads them from the talker rather than from the forward batch."""

    talker: Qwen3OmniTalker
    capture_labels: ClassVar[tuple[str, ...]] = (SEEDED_DRAW, FLASHINFER_DRAW)

    def select(self, forward_batch: ForwardBatch) -> str:
        if self.talker.decode_has_request_seed:
            return SEEDED_DRAW
        else:
            return FLASHINFER_DRAW


class TalkerDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """SGLang's decode graph runner with the talker's draws as its graph variants."""

    def capture(self) -> None:
        self.attention_graph_variants = TalkerDrawGraphVariants(self.model_runner.model)
        super().capture()
