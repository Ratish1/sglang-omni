# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import uuid

import pytest

from sglang_omni.pipeline.control_plane import PullSocket, PushSocket
from sglang_omni.profiler.profiler_control import ProfilerControlClient
from sglang_omni.proto import ProfilerResultMessage, ProfilerStartMessage


@pytest.mark.asyncio
async def test_client_waits_for_acknowledged_stage_result() -> None:
    endpoint = f"inproc://profiler-stage-{uuid.uuid4().hex}"
    stage_socket = PullSocket(endpoint, bind=True)
    await stage_socket.start()
    client = ProfilerControlClient({"asr": endpoint})
    await client.start()
    try:
        operation = asyncio.create_task(
            client.broadcast_start(
                run_id="run",
                trace_path_template="/tmp/{stage}/trace",
                stages=["asr"],
                event_dir="/tmp/events",
                enable_torch=False,
            )
        )
        message = await stage_socket.recv()
        assert isinstance(message, ProfilerStartMessage)

        reply = PushSocket(message.reply_endpoint)
        await reply.connect()
        await reply.send(
            ProfilerResultMessage(
                op_id=message.op_id,
                run_id=message.run_id,
                stage="asr",
                action="start",
                success=True,
                targets=[{"rank": 0, "pid": 10, "success": True}],
            )
        )
        reply.close()
        manifest = await operation
        assert manifest["success"]
        assert manifest["missing_stages"] == []
    finally:
        await client.close()
        stage_socket.close()


@pytest.mark.asyncio
async def test_client_rejects_unknown_stage() -> None:
    client = ProfilerControlClient({"asr": "inproc://unused"})
    await client.start()
    try:
        with pytest.raises(ValueError, match="unknown"):
            await client.broadcast_stop(stages=["router"])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_client_rejects_invalid_timeout() -> None:
    client = ProfilerControlClient({"asr": "inproc://unused"})
    await client.start()
    try:
        with pytest.raises(ValueError, match="finite and positive"):
            await client.broadcast_stop(stages=["asr"], timeout_s=0)
    finally:
        await client.close()
