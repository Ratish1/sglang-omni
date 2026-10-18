"""Check that the author's coalesced graph test catches swapped stream state."""

import argparse

import pytest
import torch

from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    CapturedGraph,
    Code2WavCudaGraphRunner,
    Code2WavRunResult,
    GraphKey,
)

parser = argparse.ArgumentParser()
parser.add_argument("--mutate", action="store_true")
arguments = parser.parse_args()
original_replay = Code2WavCudaGraphRunner.replay


def swapped_replay(
    runner: Code2WavCudaGraphRunner,
    captured: CapturedGraph,
    key: GraphKey,
    codes: torch.Tensor,
    *,
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
raise SystemExit(
    pytest.main(
        [
            "-q",
            "tests/unit_test/qwen3_omni/test_code2wav.py::test_streams_decoded_in_one_replay_keep_their_own_slots[True]",
        ]
    )
)
