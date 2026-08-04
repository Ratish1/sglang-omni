# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

import sglang_omni.profiler.torch_profiler as torch_profiler
from sglang_omni.profiler.torch_profiler import TorchProfiler, TorchProfilerConfig


@pytest.fixture(autouse=True)
def _reset_profiler_state():
    TorchProfiler._clear_state()
    yield
    TorchProfiler._clear_state()


class _FakeProfile:
    def __init__(self, *, on_trace_ready, **kwargs):
        self.on_trace_ready = on_trace_ready
        self.kwargs = kwargs
        self.steps = 0
        self.started = False

    def start(self) -> None:
        self.started = True

    def step(self) -> None:
        self.steps += 1

    def stop(self) -> None:
        pass

    def export_chrome_trace(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "name": "sglang_omni.profiler.scheduler_owner.asr",
                            "cat": "cpu_op",
                            "pid": 1,
                            "tid": 2,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )


def test_config_bounds_and_unknown_fields() -> None:
    assert TorchProfilerConfig(active_steps=3).total_steps == 5
    with pytest.raises(ValueError, match="positive"):
        TorchProfilerConfig(active_steps=0)
    with pytest.raises(ValueError, match="repeat must be 1"):
        TorchProfilerConfig(repeat=2)
    with pytest.raises(ValueError, match="unknown"):
        TorchProfilerConfig.from_dict({"mystery": 1})


def test_lifecycle_is_owner_thread_and_finalizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        torch_profiler,
        "profile",
        lambda **kwargs: _FakeProfile(**kwargs),
    )
    monkeypatch.setattr(
        torch_profiler,
        "schedule",
        lambda **kwargs: kwargs,
    )
    config = TorchProfilerConfig(
        wait_steps=0,
        warmup_steps=0,
        active_steps=2,
        include_cuda=False,
    )
    expected = TorchProfiler.start(
        str(tmp_path / "trace"),
        run_id="run",
        config=config,
    )
    assert expected.endswith(".trace.json.gz")
    TorchProfiler.step()
    TorchProfiler.step()

    errors: list[BaseException] = []

    def wrong_owner() -> None:
        try:
            TorchProfiler.step()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=wrong_owner)
    thread.start()
    thread.join()
    assert errors
    assert "owner thread" in str(errors[0])

    result = TorchProfiler.stop(run_id="run")
    assert result is not None
    assert result["trace_finalized"]
    assert result["schedule_complete"]
    assert Path(result["trace"]).is_file()
    assert not TorchProfiler.is_active()


def test_conflicting_owner_start_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        torch_profiler,
        "profile",
        lambda **kwargs: _FakeProfile(**kwargs),
    )
    monkeypatch.setattr(torch_profiler, "schedule", lambda **kwargs: kwargs)
    config = TorchProfilerConfig(include_cuda=False)
    TorchProfiler.start(str(tmp_path / "one"), run_id="run", config=config)
    with pytest.raises(RuntimeError, match="already active"):
        TorchProfiler.start(str(tmp_path / "two"), run_id="other", config=config)


def test_trace_compression_is_deferred_until_acknowledged_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        torch_profiler,
        "profile",
        lambda **kwargs: _FakeProfile(**kwargs),
    )
    monkeypatch.setattr(torch_profiler, "schedule", lambda **kwargs: kwargs)
    TorchProfiler.start(
        str(tmp_path / "trace"),
        run_id="run",
        config=TorchProfilerConfig(include_cuda=False, compress=True),
    )
    profiler = TorchProfiler._profiler
    assert isinstance(profiler, _FakeProfile)
    profiler.on_trace_ready(profiler)

    snapshot = TorchProfiler.snapshot()
    assert snapshot["trace_exported"]
    assert not snapshot["trace_finalized"]
    assert (tmp_path / "trace_rank0.trace.json").is_file()
    assert not (tmp_path / "trace_rank0.trace.json.gz").exists()

    stopped = TorchProfiler.stop(run_id="run")
    assert stopped is not None
    assert stopped["trace_finalized"]
    assert Path(stopped["trace"]).is_file()
