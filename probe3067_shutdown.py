"""Force server completion while the post-warmup freeze send is pending."""

import asyncio
import gc
import json

import uvicorn
from fastapi import FastAPI

import sglang_omni.serve.launcher as launcher
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.messages import FreezeGCMessage


async def main():
    end_serving = asyncio.Event()
    sending = asyncio.Event()
    send_accepted = asyncio.Event()

    async def serve(self, sockets=None):
        self.started = True
        await end_serving.wait()

    class ControlPlane:
        async def submit_to_stage(self, stage_name, stage_endpoint, message):
            assert isinstance(message, FreezeGCMessage)
            sending.set()
            await send_accepted.wait()

    coordinator = Coordinator(
        "inproc://complete", "inproc://abort", entry_stage="stage"
    )
    coordinator.register_stage("stage", "inproc://stage")
    coordinator.control_plane = ControlPlane()

    original = uvicorn.Server.serve
    uvicorn.Server.serve = serve
    server = launcher.PipelineUvicornServer(uvicorn.Config("unused:app"))
    app = FastAPI()
    watcher = asyncio.create_task(launcher.serve_with_failure_watch(server, []))
    warmup = asyncio.create_task(
        launcher.warm_up_server(
            server,
            app,
            coordinator,
            base_url="http://127.0.0.1:8000",
            requests=[],
            timeout_s=5,
        )
    )
    try:
        await asyncio.wait_for(sending.wait(), timeout=5)
        server.should_exit = True
        end_serving.set()
        send_accepted.set()
        await asyncio.wait_for(watcher, timeout=5)
        warmup.cancel()
        try:
            await warmup
        except asyncio.CancelledError:
            pass
        print(
            json.dumps(
                dict(
                    has_frozen_api_gc_after_serve=server.has_frozen_api_gc,
                    frozen_objects_after_serve=gc.get_freeze_count(),
                )
            ),
            flush=True,
        )
    finally:
        uvicorn.Server.serve = original
        gc.unfreeze()


if __name__ == "__main__":
    asyncio.run(main())
