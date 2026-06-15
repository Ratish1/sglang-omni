# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sglang_omni.pipeline.stage.runtime import _format_trace_path_template
from sglang_omni.profiler.torch_profiler import TorchProfiler


def test_trace_template_preserves_legacy_pid_rank_suffixes() -> None:
    template, template_has_rank = _format_trace_path_template(
        "/tmp/{run_id}/{stage}",
        run_id="run-a",
        stage="vocoder",
        pid=123,
        rank=4,
    )

    assert template == "/tmp/run-a/vocoder_pid123"
    assert template_has_rank is False
    assert (
        TorchProfiler._trace_json_path(
            template,
            rank=4,
            template_has_rank=template_has_rank,
        )
        == "/tmp/run-a/vocoder_pid123_rank4.trace.json"
    )


def test_trace_template_accepts_explicit_pid_and_rank_placeholders() -> None:
    template, template_has_rank = _format_trace_path_template(
        "/tmp/{run_id}/{stage}_pid{pid}_rank{rank}",
        run_id="run-a",
        stage="vocoder",
        pid=123,
        rank=4,
    )

    assert template == "/tmp/run-a/vocoder_pid123_rank4"
    assert template_has_rank is True
    assert (
        TorchProfiler._trace_json_path(
            template,
            rank=4,
            template_has_rank=template_has_rank,
        )
        == "/tmp/run-a/vocoder_pid123_rank4.trace.json"
    )


def test_trace_template_rejects_unknown_placeholders() -> None:
    with pytest.raises(ValueError, match="trace_path_template only supports"):
        _format_trace_path_template(
            "/tmp/{run_id}/{stage}_{worker}",
            run_id="run-a",
            stage="vocoder",
            pid=123,
            rank=4,
        )
