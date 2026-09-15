# SPDX-License-Identifier: Apache-2.0
"""Contracts of the Qwen3-TTS sampling buffer restage."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker

MAX_BS = 4
STREAM_CYCLES = 1_000_000_000


def _talker(device: torch.device) -> Qwen3TTSTalker:
    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    talker.config = SimpleNamespace(
        code_predictor_config=SimpleNamespace(vocab_size=2048)
    )
    talker._sub_temperature_tensor = torch.zeros(
        MAX_BS, dtype=torch.float32, device=device
    )
    talker._sub_top_p_tensor = torch.zeros(MAX_BS, dtype=torch.float32, device=device)
    talker._sub_top_k_tensor = torch.zeros(MAX_BS, dtype=torch.long, device=device)
    talker._semantic_sampling_seed_tensor = torch.zeros(
        MAX_BS, dtype=torch.long, device=device
    )
    talker._sub_sampling_seed_tensor = torch.zeros(
        MAX_BS, dtype=torch.long, device=device
    )
    talker._sub_do_sample_tensor = torch.zeros(MAX_BS, dtype=torch.bool, device=device)
    return talker


def _request(
    request_id: str,
    temperature: float,
    *,
    top_k: int = 40,
    top_p: float = 0.9,
    do_sample: bool = True,
    seeds: tuple[int, int] = (5, 7),
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        data=Qwen3TTSSGLangRequestData(
            semantic_sampling_seed=seeds[0],
            subtalker_dosample=do_sample,
            subtalker_temperature=temperature,
            subtalker_top_p=top_p,
            subtalker_top_k=top_k,
            subtalker_sampling_seed=seeds[1],
        ),
    )


def test_restage_lands_the_second_batch_after_two_changes_in_a_row() -> None:
    talker = _talker(torch.device("cpu"))
    first = [_request("a", 0.8), _request("b", 0.6, top_k=20)]

    talker.prepare_decode_buffers(first)
    talker.prepare_decode_buffers(list(reversed(first)))

    assert talker._sub_temperature_tensor[:2].tolist() == pytest.approx([0.6, 0.8])
    assert talker._sub_top_k_tensor[:2].tolist() == [20, 40]
    assert talker._semantic_sampling_seed_tensor[:2].tolist() == [5, 5]
    assert talker._sub_do_sample_tensor[:2].tolist() == [True, True]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_each_restage_lands_its_own_six_columns_behind_a_busy_stream() -> None:
    device = torch.device("cuda")
    talker = _talker(device)
    batches = [
        [_request("a", 0.8, seeds=(11, 12)), _request("b", 0.5, do_sample=False)],
        [
            _request("b", 0.5, do_sample=False),
            _request("c", 0.0, top_k=0, top_p=0.7, seeds=(31, 32)),
            _request("a", 0.8, seeds=(11, 12)),
        ],
        [_request("c", 0.0, top_k=0, top_p=0.7, seeds=(31, 32))],
    ]
    expected = [
        [[11, 5], [0.8, 1.0], [0.9, 1.0], [40, 1], [12, 7], [True, False]],
        [
            [5, 31, 11],
            [1.0, 1e-5, 0.8],
            [1.0, 0.7, 0.9],
            [1, 0, 40],
            [7, 32, 12],
            [False, True, True],
        ],
        [[31], [1e-5], [0.7], [0], [32], [True]],
    ]

    torch.cuda._sleep(STREAM_CYCLES)
    snapshots = []
    for batch in batches:
        talker.prepare_decode_buffers(batch)
        snapshots.append(
            [
                buffer[: len(batch)].clone()
                for buffer in (
                    talker._semantic_sampling_seed_tensor,
                    talker._sub_temperature_tensor,
                    talker._sub_top_p_tensor,
                    talker._sub_top_k_tensor,
                    talker._sub_sampling_seed_tensor,
                    talker._sub_do_sample_tensor,
                )
            ]
        )
    torch.cuda.synchronize()

    for snapshot, columns in zip(snapshots, expected):
        landed = [column.tolist() for column in snapshot]
        assert landed[0] == columns[0]
        assert landed[1] == pytest.approx(columns[1])
        assert landed[2] == pytest.approx(columns[2])
        assert landed[3:] == columns[3:]
