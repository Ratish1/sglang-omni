# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_omni.pipeline.control_plane import deserialize_message, serialize_message
from sglang_omni.proto import (
    ProfilerResultMessage,
    ProfilerStartMessage,
    ProfilerStopMessage,
)


def test_profiler_start_round_trip_includes_owner_and_reply() -> None:
    message = ProfilerStartMessage(
        op_id="op",
        run_id="run",
        trace_path_template="/tmp/{stage}",
        reply_endpoint="ipc:///tmp/reply.sock",
        event_dir="/tmp/events",
        enable_torch=True,
        enable_nvtx=False,
        torch_owner="scheduler",
        torch_config={"active_steps": 4},
        timeout_s=33.0,
    )
    decoded = deserialize_message(serialize_message(message))
    assert isinstance(decoded, ProfilerStartMessage)
    assert decoded.op_id == "op"
    assert decoded.reply_endpoint.endswith("reply.sock")
    assert decoded.torch_config == {"active_steps": 4}
    assert decoded.timeout_s == 33.0


def test_profiler_result_round_trip_preserves_rank_targets() -> None:
    message = ProfilerResultMessage(
        op_id="op",
        run_id="run",
        stage="asr",
        action="stop",
        success=True,
        targets=[{"rank": 0, "pid": 123, "success": True}],
    )
    decoded = deserialize_message(serialize_message(message))
    assert isinstance(decoded, ProfilerResultMessage)
    assert decoded.targets == [{"rank": 0, "pid": 123, "success": True}]


def test_legacy_profiler_stop_wire_shape_is_preserved() -> None:
    assert ProfilerStopMessage(run_id="run").to_dict() == {
        "type": "profiler_stop",
        "run_id": "run",
    }
