# SPDX-License-Identifier: Apache-2.0
"""The talker's decode graphs hold the FlashInfer draw; a batch with a request seed decodes
eagerly with the seeded PyTorch draw, so a seeded request's tokens do not depend on its row."""

from __future__ import annotations

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)

from sglang_omni.vendor.sglang.core import ForwardBatch


class TalkerDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """SGLang's decode graph runner, refusing a batch with a request seed."""

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        if self.model_runner.model.decode_has_request_seed:
            return False
        else:
            return super().can_run_graph(forward_batch)
