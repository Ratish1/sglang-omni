"""Check whether real cold stage-freeze sends yield to other API tasks."""

import asyncio
import json
import tempfile

from sglang_omni.pipeline.control_plane import ControlPlaneContext, StageControlPlane
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.messages import FreezeGCMessage


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        completion_endpoint = f"ipc://{directory}/complete.sock"
        abort_endpoint = f"ipc://{directory}/abort.sock"
        coordinator = Coordinator(
            completion_endpoint, abort_endpoint, entry_stage="stage0"
        )
        await coordinator.start()
        stages: list[StageControlPlane] = []
        try:
            for index in range(3):
                stage_name = f"stage{index}"
                stage_endpoint = f"ipc://{directory}/{stage_name}.sock"
                stage = StageControlPlane(
                    stage_name, stage_endpoint, completion_endpoint, abort_endpoint
                )
                await stage.start()
                stages.append(stage)
                coordinator.register_stage(stage_name, stage_endpoint)
            scheduling_points: list[str] = []
            asyncio.get_running_loop().call_soon(
                scheduling_points.append, "other API task ran"
            )
            await coordinator.freeze_gc()
            did_yield = bool(scheduling_points)
            messages = await asyncio.wait_for(
                asyncio.gather(*(stage.recv() for stage in stages)), timeout=5
            )
            assert all(isinstance(message, FreezeGCMessage) for message in messages)
            print(json.dumps({"freeze_send_yielded": did_yield}), flush=True)
        finally:
            for stage in stages:
                stage.close()
            await coordinator.stop()
            ControlPlaneContext.close()


if __name__ == "__main__":
    asyncio.run(main())
