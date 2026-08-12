# SPDX-License-Identifier: Apache-2.0
"""Process-global profiler lifecycle for colocated logical stages."""

from __future__ import annotations

from pathlib import Path

import pytest

from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.profiler.process_session import ProcessProfilerSession
from sglang_omni.profiler.torch_profiler import TorchProfiler


@pytest.fixture(autouse=True)
def _clean_process_session():
    ProcessProfilerSession.force_stop(reason="test setup")
    recorder = get_recorder()
    if recorder.is_active():
        recorder.stop()
    yield
    ProcessProfilerSession.force_stop(reason="test teardown")
    if recorder.is_active():
        recorder.stop()


def _install_fake_torch_profiler(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    state: dict[str, object] = {"active": False, "starts": [], "stops": []}

    def start(cls, trace_path_template: str, run_id: str | None = None) -> str:
        state["active"] = True
        starts = state["starts"]
        assert isinstance(starts, list)
        starts.append((trace_path_template, run_id))
        return f"{trace_path_template}_rank0.trace.json.gz"

    def stop(cls, *, run_id: str | None = None) -> dict:
        state["active"] = False
        stops = state["stops"]
        assert isinstance(stops, list)
        stops.append(run_id)
        return {"trace": "trace.json.gz", "table": None}

    monkeypatch.setattr(TorchProfiler, "start", classmethod(start))
    monkeypatch.setattr(TorchProfiler, "stop", classmethod(stop))
    monkeypatch.setattr(
        TorchProfiler,
        "is_active",
        classmethod(lambda cls: bool(state["active"])),
    )
    return state


def test_colocated_stages_share_one_process_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = _install_fake_torch_profiler(monkeypatch)
    sync_modes: list[str] = []
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.set_sync_debug_mode",
        sync_modes.append,
    )

    common = {
        "run_id": "qwen3-tts-run",
        "event_dir": str(tmp_path),
        "enable_torch": True,
        "cuda_capable_process": True,
        "cuda_sync_debug_mode": "warn",
    }
    ProcessProfilerSession.start(
        participant="preprocessing",
        trace_path_template=str(tmp_path / "trace_preprocessing"),
        **common,
    )
    ProcessProfilerSession.start(
        participant="tts_engine",
        trace_path_template=str(tmp_path / "trace_tts_engine"),
        **common,
    )
    ProcessProfilerSession.start(
        participant="vocoder",
        trace_path_template=str(tmp_path / "trace_vocoder"),
        **common,
    )

    assert ProcessProfilerSession.participants() == frozenset(
        {"preprocessing", "tts_engine", "vocoder"}
    )
    assert profiler["starts"] == [
        (str(tmp_path / "trace_preprocessing"), "qwen3-tts-run")
    ]
    assert sync_modes == ["warn"]
    assert get_recorder().is_active()

    ProcessProfilerSession.stop(participant="preprocessing", run_id="qwen3-tts-run")
    ProcessProfilerSession.stop(participant="tts_engine", run_id="qwen3-tts-run")
    assert get_recorder().is_active()
    assert profiler["stops"] == []
    assert sync_modes == ["warn"]

    ProcessProfilerSession.stop(participant="vocoder", run_id="qwen3-tts-run")
    assert ProcessProfilerSession.active_run_id() is None
    assert not get_recorder().is_active()
    assert profiler["stops"] == ["qwen3-tts-run"]
    assert sync_modes == ["warn", "default"]


def test_conflicting_join_does_not_mutate_active_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch_profiler(monkeypatch)
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.is_available",
        lambda: False,
    )
    ProcessProfilerSession.start(
        participant="tts_engine",
        run_id="same-run",
        trace_path_template=str(tmp_path / "trace"),
        event_dir=None,
        enable_torch=True,
        cuda_capable_process=True,
        cuda_sync_debug_mode="warn",
    )

    with pytest.raises(RuntimeError, match="Conflicting profiler configuration"):
        ProcessProfilerSession.start(
            participant="vocoder",
            run_id="same-run",
            trace_path_template=str(tmp_path / "other"),
            event_dir=None,
            enable_torch=True,
            cuda_capable_process=True,
            cuda_sync_debug_mode="error",
        )
    assert ProcessProfilerSession.participants() == frozenset({"tts_engine"})


def test_mismatched_stop_is_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler = _install_fake_torch_profiler(monkeypatch)
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.is_available",
        lambda: False,
    )
    ProcessProfilerSession.start(
        participant="tts_engine",
        run_id="active-run",
        trace_path_template=str(tmp_path / "trace"),
        event_dir=None,
        enable_torch=True,
        cuda_capable_process=True,
        cuda_sync_debug_mode="default",
    )
    ProcessProfilerSession.stop(participant="tts_engine", run_id="other-run")
    assert ProcessProfilerSession.active_run_id() == "active-run"
    assert profiler["stops"] == []


def test_non_cuda_process_does_not_arm_visible_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch_profiler(monkeypatch)
    sync_modes: list[str] = []
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.set_sync_debug_mode",
        sync_modes.append,
    )

    ProcessProfilerSession.start(
        participant="cpu_preprocessing",
        run_id="cpu-run",
        trace_path_template=str(tmp_path / "trace"),
        event_dir=None,
        enable_torch=False,
        cuda_capable_process=False,
        cuda_sync_debug_mode="warn",
    )
    ProcessProfilerSession.stop(participant="cpu_preprocessing", run_id="cpu-run")

    assert sync_modes == []


def test_force_stop_resets_debug_before_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    state = _install_fake_torch_profiler(monkeypatch)
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.set_sync_debug_mode",
        lambda mode: order.append(f"debug:{mode}"),
    )

    original_start = TorchProfiler.start
    original_stop = TorchProfiler.stop

    def ordered_start(cls, trace_path_template: str, run_id: str | None = None):
        order.append("profiler:start")
        return original_start(trace_path_template, run_id=run_id)

    def ordered_stop(cls, *, run_id: str | None = None):
        order.append("profiler:stop")
        return original_stop(run_id=run_id)

    monkeypatch.setattr(TorchProfiler, "start", classmethod(ordered_start))
    monkeypatch.setattr(TorchProfiler, "stop", classmethod(ordered_stop))
    ProcessProfilerSession.start(
        participant="vocoder",
        run_id="force-run",
        trace_path_template=str(tmp_path / "trace"),
        event_dir=None,
        enable_torch=True,
        cuda_capable_process=True,
        cuda_sync_debug_mode="error",
    )
    ProcessProfilerSession.force_stop(reason="stage teardown")

    assert state["stops"] == ["force-run"]
    assert order == [
        "profiler:start",
        "debug:error",
        "debug:default",
        "profiler:stop",
    ]
    assert ProcessProfilerSession.active_run_id() is None


def test_force_stop_clears_ownership_when_profiler_export_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_torch_profiler(monkeypatch)
    sync_modes: list[str] = []
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "sglang_omni.profiler.process_session.torch.cuda.set_sync_debug_mode",
        sync_modes.append,
    )

    def failed_stop(cls, *, run_id: str | None = None) -> None:
        del cls, run_id
        raise RuntimeError("export failed")

    monkeypatch.setattr(
        TorchProfiler,
        "stop",
        classmethod(failed_stop),
    )
    ProcessProfilerSession.start(
        participant="tts_engine",
        run_id="failed-export",
        trace_path_template=str(tmp_path / "trace"),
        event_dir=None,
        enable_torch=True,
        cuda_capable_process=True,
        cuda_sync_debug_mode="warn",
    )

    ProcessProfilerSession.force_stop(reason="test exporter failure")

    assert ProcessProfilerSession.active_run_id() is None
    assert ProcessProfilerSession.participants() == frozenset()
    assert sync_modes == ["warn", "default"]


def test_torch_profiler_same_run_start_is_idempotent() -> None:
    old_profiler = TorchProfiler._profiler
    old_template = TorchProfiler._trace_template
    old_run_id = TorchProfiler._active_run_id
    try:
        TorchProfiler._profiler = object()  # type: ignore[assignment]
        TorchProfiler._trace_template = "/tmp/existing"
        TorchProfiler._active_run_id = "same-run"
        result = TorchProfiler.start("/tmp/ignored", run_id="same-run")
        assert result == "/tmp/existing_rank0.trace.json.gz"
    finally:
        TorchProfiler._profiler = old_profiler
        TorchProfiler._trace_template = old_template
        TorchProfiler._active_run_id = old_run_id


def test_semantic_range_is_created_only_while_profiler_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    names: list[str] = []
    old_profiler = TorchProfiler._profiler
    monkeypatch.setattr(
        "sglang_omni.profiler.torch_profiler.record_function",
        lambda name: names.append(name) or _TestContext(),
    )
    try:
        TorchProfiler._profiler = None
        with TorchProfiler.record_function("inactive"):
            pass
        assert names == []

        TorchProfiler._profiler = object()  # type: ignore[assignment]
        with TorchProfiler.record_function("active"):
            pass
        assert names == ["active"]
    finally:
        TorchProfiler._profiler = old_profiler


class _TestContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False
