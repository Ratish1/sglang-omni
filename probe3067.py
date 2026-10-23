"""Check the lifetime of real HTTP warmup responses across API heap freezing."""

import asyncio
import gc
import inspect
import json
import sys
import weakref

import httpx
import uvicorn
from fastapi import FastAPI

import sglang_omni.serve.launcher as launcher
from sglang_omni.serve.openai_api import ServerStatus
from sglang_omni.serve.protocol import ChatCompletionRequest, ChatMessage


class Coordinator:
    async def freeze_gc(self):
        return None


async def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "current"
    gc.collect()
    automatic_gc = len(sys.argv) > 2 and sys.argv[2] == "automatic"
    if automatic_gc:
        gc.enable()
    else:
        gc.disable()
    references = []

    def respond(request):
        response = httpx.Response(
            200,
            json={
                "id": "warmup",
                "object": "chat.completion",
                "created": 0,
                "model": "warmup",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "x" * (4 * 1024 * 1024),
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )
        references.append(weakref.ref(response))
        return response

    original_client = httpx.AsyncClient

    def client(*args, **kwargs):
        return original_client(*args, transport=httpx.MockTransport(respond), **kwargs)

    launcher.httpx.AsyncClient = client
    server = launcher.PipelineUvicornServer(uvicorn.Config("unused:app"))
    server.started = True
    app = FastAPI()
    app.state.server_status = ServerStatus.STARTING
    request = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
    has_freeze = hasattr(server, "freeze_api_gc")
    if has_freeze:
        original_freeze = server.freeze_api_gc
        if mode == "collect":

            def collect_then_freeze():
                gc.collect()
                original_freeze()

            server.freeze_api_gc = collect_then_freeze
        elif mode == "deferred":
            server.freeze_api_gc = lambda: None
        else:
            pass
    else:
        pass
    kwargs = dict(base_url="http://127.0.0.1:8000", requests=[request] * 3, timeout_s=5)
    if "coordinator" in inspect.signature(launcher.warm_up_server).parameters:
        kwargs["coordinator"] = Coordinator()
    else:
        pass
    try:
        await launcher.warm_up_server(server, app, **kwargs)
        if mode == "deferred" and has_freeze:
            gc.collect()
            original_freeze()
        else:
            pass
        gc.collect()
        retained = [ref() is not None for ref in references]
        print(
            json.dumps(
                dict(
                    mode=mode,
                    automatic_gc=automatic_gc,
                    response_count=len(references),
                    retained_after_warmup=retained,
                    retained_body_bytes=sum(
                        len(ref().content) for ref in references if ref() is not None
                    ),
                    freeze_count=gc.get_freeze_count(),
                )
            ),
            flush=True,
        )
    finally:
        launcher.httpx.AsyncClient = original_client
        gc.unfreeze()
        gc.collect()
        print(
            json.dumps(
                dict(retained_after_unfreeze=[ref() is not None for ref in references])
            ),
            flush=True,
        )
        gc.enable()


if __name__ == "__main__":
    asyncio.run(main())
