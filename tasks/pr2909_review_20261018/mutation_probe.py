"""Check whether tests detect swapped stream slots in a multi-row next graph."""

from __future__ import annotations

import argparse

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    CapturedGraph,
    Code2WavCudaGraphRunner,
    Code2WavRunResult,
    GraphKey,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutate", action="store_true")
    parser.add_argument("--existing", action="store_true")
    arguments = parser.parse_args()
    original_replay = Code2WavCudaGraphRunner.replay

    def swapped_replay(
        runner: Code2WavCudaGraphRunner,
        captured: CapturedGraph,
        key: GraphKey,
        codes: torch.Tensor,
        slots: list[int] | None,
    ) -> Code2WavRunResult:
        if key.decode == "next" and key.batch_size > 1:
            slots = list(reversed(slots))
        else:
            pass
        return original_replay(runner, captured, key, codes, slots=slots)

    if arguments.mutate:
        Code2WavCudaGraphRunner.replay = swapped_replay
    else:
        pass
    if arguments.existing:
        target = "tests/unit_test/qwen3_omni/test_code2wav.py::test_scheduler_decodes_streams_from_their_slots"
    else:
        target = "tasks/pr2909_review_20261018/test_graph_cohorts.py"
    raise SystemExit(pytest.main(["-q", target]))


if __name__ == "__main__":
    main()
