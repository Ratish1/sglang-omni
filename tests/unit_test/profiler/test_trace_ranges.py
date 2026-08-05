# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.profiler import trace_ranges


def test_async_trace_range_is_noop_when_nvtx_is_disabled(monkeypatch) -> None:
    monkeypatch.setattr(trace_ranges, "_NVTX_ENABLED", False)

    assert trace_ranges.start_async_trace_range("request_build.total") is None


def test_async_trace_range_uses_matching_nvtx_handle(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(trace_ranges, "_NVTX_ENABLED", True)
    monkeypatch.setattr(trace_ranges.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        trace_ranges.torch.cuda.nvtx,
        "range_start",
        lambda name: calls.append(("start", name)) or 17,
    )
    monkeypatch.setattr(
        trace_ranges.torch.cuda.nvtx,
        "range_end",
        lambda handle: calls.append(("end", handle)),
    )

    handle = trace_ranges.start_async_trace_range("request_build.head_of_line")
    trace_ranges.end_async_trace_range(handle)

    assert handle == 17
    assert calls == [
        ("start", "request_build.head_of_line"),
        ("end", 17),
    ]


def test_async_trace_range_does_not_end_missing_handle(monkeypatch) -> None:
    monkeypatch.setattr(
        trace_ranges.torch.cuda.nvtx,
        "range_end",
        lambda handle: (_ for _ in ()).throw(AssertionError(handle)),
    )

    trace_ranges.end_async_trace_range(None)
