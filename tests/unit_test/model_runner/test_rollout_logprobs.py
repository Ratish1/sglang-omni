# SPDX-License-Identifier: Apache-2.0
"""Rollout logprob recording uses sampler-computed logprobs."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.base import ModelRunner


def _req(inflight_middle_chunks: int = 0):
    return SimpleNamespace(
        inflight_middle_chunks=inflight_middle_chunks,
        sampling_params=SimpleNamespace(sampling_seed=None),
    )


def _data(top_logprobs_num: int = 0, **overrides):
    fields = dict(
        return_logprob=True,
        output_token_logprobs=[],
        top_logprobs_num=top_logprobs_num,
        output_top_logprobs=[],
        req=_req(),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_rollout_logprobs_record_sampler_values_not_raw_logits() -> None:
    runner = object.__new__(ModelRunner)
    logits = torch.tensor([[2.0, 1.0]])
    token_ids = torch.tensor([0])
    raw_logprob = torch.log_softmax(logits, dim=-1)[0, 0].item()
    sampler_logprob = torch.log_softmax(logits / 0.5, dim=-1)[0, 0].item()
    data = _data()
    request = SimpleNamespace(data=data)

    runner._record_rollout_logprobs(
        torch.tensor([sampler_logprob]), token_ids, [request]
    )

    assert len(data.output_token_logprobs) == 1
    assert data.output_token_logprobs[0][1] == 0
    assert math.isclose(data.output_token_logprobs[0][0], sampler_logprob, abs_tol=1e-4)
    assert not math.isclose(data.output_token_logprobs[0][0], raw_logprob, abs_tol=1e-4)


def test_rollout_logprobs_align_per_request_at_batch_2() -> None:
    runner = object.__new__(ModelRunner)
    data0 = _data()
    data1 = _data()
    requests = [SimpleNamespace(data=data0), SimpleNamespace(data=data1)]

    runner._record_rollout_logprobs(
        torch.tensor([-0.5, -1.5]), torch.tensor([11, 22]), requests
    )

    # each request must get ITS OWN row, not the other request's token/logprob
    assert data0.output_token_logprobs[0][1] == 11
    assert data1.output_token_logprobs[0][1] == 22
    assert math.isclose(data0.output_token_logprobs[0][0], -0.5, abs_tol=1e-4)
    assert math.isclose(data1.output_token_logprobs[0][0], -1.5, abs_tol=1e-4)


def test_rollout_logprobs_raises_on_batch_size_mismatch() -> None:
    runner = object.__new__(ModelRunner)
    data = _data()

    # more logprobs than requests => batching assumption broke => fail loud
    with pytest.raises(RuntimeError, match="batch-size mismatch"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.5, -1.5]),
            torch.tensor([11, 22]),
            [SimpleNamespace(data=data)],
        )

    assert data.output_token_logprobs == []


def test_rollout_logprobs_raises_on_malformed_sampler_shape() -> None:
    runner = object.__new__(ModelRunner)
    data = _data()

    with pytest.raises(RuntimeError, match="Failed to convert"):
        runner._record_rollout_logprobs(
            [[-0.5, -0.7]],
            torch.tensor([11]),
            [SimpleNamespace(data=data)],
        )

    assert data.output_token_logprobs == []


def test_rollout_logprobs_raises_when_sampler_omits_token_ids() -> None:
    runner = object.__new__(ModelRunner)
    data = _data()

    with pytest.raises(RuntimeError, match="next_token_ids"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.5]),
            None,
            [SimpleNamespace(data=data)],
        )

    assert data.output_token_logprobs == []


def test_enable_sampler_logprobs_sets_per_row_top_k() -> None:
    forward_batch = SimpleNamespace(top_logprobs_nums=None, token_ids_logprobs=None)
    requests = [
        SimpleNamespace(data=_data(top_logprobs_num=3)),
        SimpleNamespace(data=SimpleNamespace(return_logprob=False)),
        SimpleNamespace(data=_data()),
    ]

    ModelRunner._enable_sampler_logprobs(forward_batch, requests)

    assert forward_batch.return_logprob is True
    assert forward_batch.top_logprobs_nums == [3, 0, 0]
    assert forward_batch.token_ids_logprobs == [None, None, None]


def test_enable_sampler_logprobs_requests_no_top_k_for_middle_chunk_rows() -> None:
    forward_batch = SimpleNamespace(top_logprobs_nums=None, token_ids_logprobs=None)
    requests = [
        SimpleNamespace(data=_data(top_logprobs_num=3, req=_req(1))),
        SimpleNamespace(data=_data(top_logprobs_num=3)),
    ]

    ModelRunner._enable_sampler_logprobs(forward_batch, requests)

    assert forward_batch.top_logprobs_nums == [0, 3]


def test_rollout_logprobs_skip_middle_chunk_rows() -> None:
    runner = object.__new__(ModelRunner)
    middle = _data(top_logprobs_num=2, req=_req(1))
    final = _data(top_logprobs_num=2)
    requests = [SimpleNamespace(data=middle), SimpleNamespace(data=final)]

    runner._record_rollout_logprobs(
        torch.tensor([-0.3, -0.5]),
        torch.tensor([7, 11]),
        requests,
        top_logprobs_val=[torch.tensor([]), torch.tensor([-0.5, -1.2])],
        top_logprobs_idx=[
            torch.tensor([], dtype=torch.long),
            torch.tensor([11, 13]),
        ],
    )

    assert middle.output_token_logprobs == []
    assert middle.output_top_logprobs == []
    assert final.output_token_logprobs == [[-0.5, 11]]
    assert [entry[1] for entry in final.output_top_logprobs[0]] == [11, 13]


def test_rollout_logprobs_record_the_final_chunk_row() -> None:
    runner = object.__new__(ModelRunner)
    data = _data(req=_req(0))

    runner._record_rollout_logprobs(
        torch.tensor([-0.25]), torch.tensor([7]), [SimpleNamespace(data=data)]
    )

    assert data.output_token_logprobs == [[-0.25, 7]]


def test_rollout_top_logprobs_record_only_rows_that_asked() -> None:
    runner = object.__new__(ModelRunner)
    data0 = _data(top_logprobs_num=2)
    data1 = _data()
    requests = [SimpleNamespace(data=data0), SimpleNamespace(data=data1)]

    runner._record_rollout_logprobs(
        torch.tensor([-0.5, -1.5]),
        torch.tensor([11, 22]),
        requests,
        top_logprobs_val=[torch.tensor([-0.5, -1.2]), torch.tensor([])],
        top_logprobs_idx=[torch.tensor([11, 13]), torch.tensor([], dtype=torch.long)],
    )

    assert data0.output_token_logprobs[0][1] == 11
    (row,) = data0.output_top_logprobs
    assert [entry[1] for entry in row] == [11, 13]
    assert math.isclose(row[0][0], -0.5, abs_tol=1e-4)
    assert math.isclose(row[1][0], -1.2, abs_tol=1e-4)
    assert data1.output_token_logprobs[0][1] == 22
    assert data1.output_top_logprobs == []


def test_rollout_top_logprobs_raise_when_sampler_omits_them() -> None:
    runner = object.__new__(ModelRunner)
    data = _data(top_logprobs_num=2)

    with pytest.raises(RuntimeError, match="next_token_top_logprobs"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.5]), torch.tensor([11]), [SimpleNamespace(data=data)]
        )

    assert data.output_token_logprobs == []
    assert data.output_top_logprobs == []


def test_rollout_top_logprobs_raise_on_row_size_mismatch() -> None:
    runner = object.__new__(ModelRunner)
    data = _data(top_logprobs_num=3)

    with pytest.raises(RuntimeError, match="top-k logprob size mismatch"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.5]),
            torch.tensor([11]),
            [SimpleNamespace(data=data)],
            top_logprobs_val=[torch.tensor([-0.5, -1.2])],
            top_logprobs_idx=[torch.tensor([11, 13])],
        )

    assert data.output_token_logprobs == []
    assert data.output_top_logprobs == []


def test_record_rollout_logprobs_skips_without_return_flag() -> None:
    runner = object.__new__(ModelRunner)
    data = _data(return_logprob=False)

    runner._record_rollout_logprobs(
        torch.tensor([-0.25]), torch.tensor([33]), [SimpleNamespace(data=data)]
    )

    assert data.output_token_logprobs == []


def test_record_rollout_logprobs_requires_output_list() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(return_logprob=True, top_logprobs_num=0, req=_req())

    with pytest.raises(AttributeError, match="output_token_logprobs"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.25]), torch.tensor([33]), [SimpleNamespace(data=data)]
        )

    assert "output_token_logprobs" not in vars(data)


def test_record_rollout_logprobs_requires_return_flag() -> None:
    runner = object.__new__(ModelRunner)
    data = SimpleNamespace(output_token_logprobs=[])

    with pytest.raises(AttributeError, match="return_logprob"):
        runner._record_rollout_logprobs(
            torch.tensor([-0.25]), torch.tensor([33]), [SimpleNamespace(data=data)]
        )


def test_sample_next_token_ids_requires_sampler_logprobs_when_requested() -> None:
    runner = object.__new__(ModelRunner)
    runner._apply_repetition_penalty = lambda *args: None
    runner._apply_codec_suppress_tokens = lambda *args: None
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            sample=lambda _logits_output, _forward_batch: torch.tensor([44])
        )
    )
    data = _data()
    request = SimpleNamespace(data=data)
    forward_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(device="cpu", sampling_seed=None),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )
    logits_output = SimpleNamespace()

    with pytest.raises(RuntimeError, match="next_token_logprobs"):
        runner._sample_next_token_ids(
            logits_output,
            forward_batch,
            SimpleNamespace(),
            [request],
        )

    assert forward_batch.return_logprob is True
    assert data.output_token_logprobs == []


def test_sample_next_token_ids_records_sampler_top_logprobs() -> None:
    runner = object.__new__(ModelRunner)
    runner._apply_repetition_penalty = lambda *args: None
    runner._apply_codec_suppress_tokens = lambda *args: None
    logits_output = SimpleNamespace(
        next_token_logprobs=None,
        next_token_top_logprobs_val=None,
        next_token_top_logprobs_idx=None,
    )

    def sample(_logits_output, forward_batch):
        assert forward_batch.return_logprob is True
        assert forward_batch.top_logprobs_nums == [2]
        logits_output.next_token_logprobs = torch.tensor([-0.25])
        logits_output.next_token_top_logprobs_val = [torch.tensor([-0.25, -1.75])]
        logits_output.next_token_top_logprobs_idx = [torch.tensor([44, 45])]
        return torch.tensor([44])

    runner.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(sample=sample))
    data = _data(top_logprobs_num=2)
    forward_batch = SimpleNamespace(
        sampling_info=SimpleNamespace(device="cpu", sampling_seed=None),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )

    runner._sample_next_token_ids(
        logits_output,
        forward_batch,
        SimpleNamespace(),
        [SimpleNamespace(data=data)],
    )

    assert data.output_token_logprobs == [[-0.25, 44]]
    assert data.output_top_logprobs == [[[-0.25, 44], [-1.75, 45]]]
