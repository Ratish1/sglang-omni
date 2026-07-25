# SPDX-License-Identifier: Apache-2.0
"""Regression: the resolve collect must not mutate output for a row whose
request finished or retracted in a PRIOR step (the lookahead overrun).

Before the guard, every ZONOS2 request that finished under async decode got
one extra frame appended by the next (already-launched) step's resolve — the
generic skip in ``_finalize`` runs after this hook, so it could not help.
These drive the REAL ``_collect_resolve`` body with real upstream ``Req``
lifecycle state; only the CUDA copy-stream plumbing is stubbed (CPU host).
"""

from __future__ import annotations

import contextlib
import types
from unittest import mock

import pytest
import torch

from sglang_omni.models.zonos2.model_runner import Zonos2ModelRunner

N_CODEBOOKS = 4


@pytest.fixture()
def server_args():
    from sglang.srt import server_args as sa_mod

    previous = getattr(sa_mod, "_global_server_args", None)
    args = sa_mod.ServerArgs(model_path="dummy", tokenizer_path="dummy")
    sa_mod.set_global_server_args_for_scheduler(args)
    try:
        yield args
    finally:
        sa_mod._global_server_args = previous


def _real_req(rid: str, *, finished: bool = False, retracted: bool = False):
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    params = SamplingParams(max_new_tokens=8)
    params.normalize(tokenizer=None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=[1, 2, 3],
        sampling_params=params,
        vocab_size=128,
    )
    if finished:
        req.finished_reason = FINISH_LENGTH(length=1)
    req.is_retracted = retracted
    return req


class _RecordingPool:
    def __init__(self):
        self.released: list[str] = []

    def release_row(self, rid: str) -> None:
        self.released.append(rid)


class _FakeCopyStream:
    def wait_event(self, event) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _run_collect(rows):
    """Drive the real _collect_resolve over ``rows`` =
    [(req, sampler_finished)]. Returns (datas, pool, result)."""
    runner = Zonos2ModelRunner.__new__(Zonos2ModelRunner)
    runner._copy_stream = _FakeCopyStream()
    pool = _RecordingPool()
    runner.model = types.SimpleNamespace(_decode_state_pool=pool)

    requests, datas = [], []
    codes, meta = [], []
    for i, (req, sampler_finished) in enumerate(rows):
        data = types.SimpleNamespace(req=req, output_codes=[], eos_frame=None)
        datas.append(data)
        requests.append(types.SimpleNamespace(request_id=req.rid, data=data))
        codes.append([100 + i] * N_CODEBOOKS)
        # packed layout produced by the launch: codes | eos_set | eos_val | finished
        meta.append([0, 0, int(sampler_finished)])
    packed = torch.tensor([c + m for c, m in zip(codes, meta)], dtype=torch.int64)
    next_ids = torch.arange(len(rows), dtype=torch.int64)
    launch_buf = (requests, packed, N_CODEBOOKS, next_ids, object())
    result = types.SimpleNamespace(next_token_ids=None)

    with mock.patch("torch.cuda.stream", lambda s: contextlib.nullcontext()):
        runner._collect_resolve(launch_buf, result)
    return datas, pool, result


def test_prior_finished_and_retracted_rows_do_not_append(server_args):
    live = _real_req("live")
    done = _real_req("done", finished=True)
    retracted = _real_req("retracted", retracted=True)
    datas, pool, _ = _run_collect([(live, False), (done, False), (retracted, False)])

    assert len(datas[0].output_codes) == 1, "live row must keep its frame"
    assert datas[1].output_codes == [], "prior-finished overrun must not append"
    assert datas[2].output_codes == [], "retracted overrun must not append"
    assert pool.released == [], "overrun rows were released at their own step"


def test_sampler_finished_live_row_appends_terminal_frame_and_releases(server_args):
    """The guard must not eat the LEGITIMATE terminal path: a row whose EOS is
    detected by THIS step's sampler (finished_cpu) is not yet finished() on the
    host — its terminal frame must land and its pool row must be released."""
    live_terminal = _real_req("t")
    datas, pool, result = _run_collect([(live_terminal, True)])

    assert len(datas[0].output_codes) == 1
    assert datas[0].output_codes[0].tolist() == [100] * N_CODEBOOKS
    assert pool.released == ["t"]
    assert result.next_token_ids is not None, "resolve must restore the snapshot"
