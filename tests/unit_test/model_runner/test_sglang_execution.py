# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge


class _FutureMap:
    def __init__(self) -> None:
        self.stashed = None
        self.published = None

    def stash(self, indices, payload) -> None:
        self.stashed = (indices, payload)

    def publish(self, indices, seq_lens) -> None:
        self.published = (indices, seq_lens)


class _SpecAlgorithm:
    def __init__(self, future_map: _FutureMap) -> None:
        self.future_map = future_map

    def create_future_map(self, device, req_to_token_pool, needs_cpu_seq_lens):
        assert device == torch.device("cpu")
        assert req_to_token_pool == "pool"
        assert needs_cpu_seq_lens is True
        return self.future_map

    def is_none(self) -> bool:
        return True


def _make_bridge() -> tuple[SGLangExecutionBridge, _FutureMap]:
    future_map = _FutureMap()
    worker = SimpleNamespace(model_runner=SimpleNamespace())
    bridge = SGLangExecutionBridge(
        device=torch.device("cpu"),
        worker=worker,
        req_to_token_pool="pool",
        spec_algorithm=_SpecAlgorithm(future_map),
        enable_stream_overlap=False,
    )
    return bridge, future_map


def test_publish_next_tokens_uses_future_map_and_retires_input_ids() -> None:
    bridge, future_map = _make_bridge()
    batch = SimpleNamespace(
        req_pool_indices=torch.tensor([4, 7]),
        seq_lens=torch.tensor([12, 19]),
        input_ids=torch.tensor([1, 2]),
    )
    next_token_ids = torch.tensor([31, 32])

    bridge.publish_next_tokens(batch, next_token_ids)

    stash_indices, payload = future_map.stashed
    assert torch.equal(stash_indices, batch.req_pool_indices)
    assert torch.equal(payload.bonus_tokens, next_token_ids)
    publish_indices, publish_seq_lens = future_map.published
    assert torch.equal(publish_indices, batch.req_pool_indices)
    assert torch.equal(publish_seq_lens, torch.tensor([13, 20]))
    assert batch.input_ids is None


def test_forward_context_resolves_inputs_and_isolates_sampling_info(
    monkeypatch,
) -> None:
    bridge, _ = _make_bridge()
    original_sampling_info = SimpleNamespace(
        copy_for_forward=lambda: "forward-only-sampling-info"
    )
    batch = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(_FutureMap()),
        sampling_info=original_sampling_info,
    )
    seen = []

    def resolve_forward_inputs(actual_batch, future_map) -> None:
        seen.append((actual_batch, future_map))

    monkeypatch.setattr(
        "sglang.srt.managers.overlap_utils.resolve_forward_inputs",
        resolve_forward_inputs,
    )

    with bridge.forward_context(batch):
        assert batch.sampling_info == "forward-only-sampling-info"

    assert seen == [(batch, bridge.future_map)]
    assert batch.sampling_info is original_sampling_info


def test_overlap_fences_scheduler_on_graph_read_done_event(monkeypatch) -> None:
    class _ScheduleStream:
        def __init__(self) -> None:
            self.waited_event = None
            self.waited_stream = None

        def wait_event(self, event) -> None:
            self.waited_event = event

        def wait_stream(self, stream) -> None:
            self.waited_stream = stream

    class _DeviceModule:
        @staticmethod
        def stream(stream):
            del stream
            return nullcontext()

    future_map = _FutureMap()
    read_done = object()
    forward_stream = object()
    runner = SimpleNamespace(
        forward_stream=forward_stream,
        war_fastpath_read_done_event=read_done,
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: _DeviceModule())
    bridge = SGLangExecutionBridge(
        device=torch.device("cuda"),
        worker=SimpleNamespace(model_runner=runner),
        req_to_token_pool="pool",
        spec_algorithm=SimpleNamespace(
            create_future_map=lambda *args, **kwargs: future_map
        ),
        enable_stream_overlap=True,
    )
    schedule_stream = _ScheduleStream()
    bridge.schedule_stream = schedule_stream

    bridge.before_schedule()

    assert schedule_stream.waited_event is read_done
    assert schedule_stream.waited_stream is None
    assert runner.war_fastpath_read_done_event is None

    bridge.before_schedule()
    assert schedule_stream.waited_stream is forward_stream
