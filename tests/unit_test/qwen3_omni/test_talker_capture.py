# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_omni.talker_capture import CAPTURE_DIR_ENV, TalkerCapture


def request_data() -> SimpleNamespace:
    return SimpleNamespace(
        req=SimpleNamespace(
            origin_input_ids=[5, 6],
            sampling_params=SimpleNamespace(
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.05,
                sampling_seed=7,
            ),
        ),
        prefill_input_embeds=torch.zeros(2, 4),
        decode_input_embeds=[torch.full((4,), 2.0)],
        stage_payload=SimpleNamespace(
            request=SimpleNamespace(metadata={"sample_id": "first"})
        ),
    )


def test_finish_writes_each_request_steps_and_inputs(tmp_path: Path) -> None:
    capture = TalkerCapture(tmp_path)
    capture.record_step(
        ["first", "second"],
        torch.arange(8.0).view(2, 4),
        torch.arange(32).view(2, 16),
    )
    capture.record_step(
        ["first"], torch.ones(1, 4), torch.zeros(1, 16, dtype=torch.long)
    )

    capture.finish("first", request_data())

    record = torch.load(tmp_path / "first.pt")
    assert torch.equal(
        record["hidden"], torch.stack([torch.arange(4.0), torch.ones(4)])
    )
    assert torch.equal(
        record["codes"],
        torch.stack([torch.arange(16), torch.zeros(16, dtype=torch.long)]),
    )
    assert torch.equal(record["prefill_input_ids"], torch.tensor([5, 6]))
    assert torch.equal(record["decode_input_embeds"], torch.full((1, 4), 2.0))
    assert record["metadata"] == {"sample_id": "first"}
    assert record["sampling"]["seed"] == 7
    assert "second" in capture.steps and "first" not in capture.steps


def test_record_step_keeps_rows_the_next_step_overwrites(tmp_path: Path) -> None:
    capture = TalkerCapture(tmp_path)
    hidden = torch.zeros(1, 4)
    codes = torch.zeros(1, 16, dtype=torch.long)

    capture.record_step(["first"], hidden, codes)
    hidden.fill_(1.0)
    codes.fill_(3)

    assert torch.equal(capture.steps["first"].hidden[0], torch.zeros(4))
    assert torch.equal(
        capture.steps["first"].codes[0], torch.zeros(16, dtype=torch.long)
    )


def test_finish_without_steps_writes_nothing(tmp_path: Path) -> None:
    TalkerCapture(tmp_path).finish("first", request_data())

    assert not list(tmp_path.iterdir())


def test_capture_is_off_without_a_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CAPTURE_DIR_ENV, raising=False)

    assert TalkerCapture.from_environment() is None
