"""Exercise API, replica and TP-follower heap freezing in separate processes."""

import asyncio
import gc
import json
import multiprocessing
import os
import tempfile
import weakref

import uvicorn
from fastapi import FastAPI

from sglang_omni.pipeline.control_plane import ControlPlaneContext, StageControlPlane
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.pipeline.tp_control import TPFollowerControlPlane, TPLeaderFanout
from sglang_omni.proto.messages import FreezeGCMessage
from sglang_omni.serve.launcher import PipelineUvicornServer, warm_up_server
from tests.unit_test.pipeline.helpers import make_stage


class Cycle:
    def __init__(self) -> None:
        self.reference = self


def run_stage(
    name, role, endpoints, work_queue, abort_queue, ready_queue, reports_queue
):
    warm = Cycle()
    warm_reference = weakref.ref(warm)
    if role == "follower":
        control = TPFollowerControlPlane(
            stage_name=name, work_queue=work_queue, abort_queue=abort_queue
        )
        fanout = None
    else:
        control = StageControlPlane(
            name, endpoints[name], endpoints["complete"], endpoints["abort"]
        )
        if role == "leader":
            fanout = TPLeaderFanout(
                name,
                follower_work_queues=[work_queue],
                follower_abort_queues=[abort_queue],
            )
        else:
            fanout = None
    stage = make_stage(name=name, role=role, control_plane=control, tp_fanout=fanout)
    original_start = stage.start
    original_handle = stage.handle_message

    async def start():
        await original_start()
        ready_queue.put(name)

    async def handle(message):
        nonlocal warm
        await original_handle(message)
        if isinstance(message, FreezeGCMessage):
            del warm
            fresh = Cycle()
            fresh_reference = weakref.ref(fresh)
            del fresh
            gc.collect()
            reports_queue.put(
                {
                    "name": name,
                    "pid": os.getpid(),
                    "warm_retained": warm_reference() is not None,
                    "fresh_collected": fresh_reference() is None,
                }
            )
        else:
            pass

    stage.start = start
    stage.handle_message = handle
    try:
        asyncio.run(stage.run())
    finally:
        gc.unfreeze()
        ControlPlaneContext.close()


async def main() -> None:
    context = multiprocessing.get_context("spawn")
    work_queue = context.Queue()
    abort_queue = context.Queue()
    ready_queue = context.Queue()
    reports_queue = context.Queue()
    with tempfile.TemporaryDirectory() as directory:
        endpoints = {
            name: f"ipc://{directory}/{name}.sock"
            for name in ("stage@r0", "stage@r1", "complete", "abort")
        }
        coordinator = Coordinator(
            endpoints["complete"], endpoints["abort"], entry_stage="stage@r0"
        )
        await coordinator.start()
        processes = []
        try:
            for name, role in (
                ("stage@r0", "leader"),
                ("stage@r1", "single"),
                ("stage-follower", "follower"),
            ):
                process = context.Process(
                    target=run_stage,
                    args=(
                        name,
                        role,
                        endpoints,
                        work_queue,
                        abort_queue,
                        ready_queue,
                        reports_queue,
                    ),
                )
                process.start()
                processes.append(process)
            names = {ready_queue.get(timeout=90) for _ in processes}
            assert names == {"stage@r0", "stage@r1", "stage-follower"}
            for name in ("stage@r0", "stage@r1"):
                coordinator.register_stage(name, endpoints[name])
            server = PipelineUvicornServer(uvicorn.Config("unused:app"))
            server.started = True
            await warm_up_server(
                server,
                FastAPI(),
                coordinator,
                base_url="http://127.0.0.1:8000",
                requests=[],
                timeout_s=5,
            )
            reports = [reports_queue.get(timeout=30) for _ in processes]
            assert server.has_frozen_api_gc
            assert {report["name"] for report in reports} == names
            assert len({report["pid"] for report in reports}) == len(processes)
            assert all(
                report["warm_retained"] and report["fresh_collected"]
                for report in reports
            )
            await coordinator.shutdown_stages()
            for process in processes:
                await asyncio.to_thread(process.join, 30)
                assert process.exitcode == 0, (process.pid, process.exitcode)
            print(json.dumps({"api_frozen": True, "processes": reports}), flush=True)
        finally:
            gc.unfreeze()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(10)
                else:
                    pass
            await coordinator.stop()
            ControlPlaneContext.close()
            for channel in (work_queue, abort_queue, ready_queue, reports_queue):
                channel.close()
                channel.join_thread()


if __name__ == "__main__":
    asyncio.run(main())
