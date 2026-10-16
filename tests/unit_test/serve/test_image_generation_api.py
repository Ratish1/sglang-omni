# SPDX-License-Identifier: Apache-2.0
"""Image API validation, metadata forwarding, and response format."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Literal, NoReturn, TypedDict

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client
from sglang_omni.client.types import ClientError, GenerateRequest
from sglang_omni.models.llada2_uni.components.image_decoder import LLaDA2ImageDecoder
from sglang_omni.models.llada2_uni.config import (
    IMAGE_DECODE_STAGE,
    LLaDA2ImageDecoderStageConfig,
    LLaDA2UniOmniPipelineConfig,
)
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto import CompleteMessage, StreamMessage
from sglang_omni.proto.messages import AbortMessage, SubmitMessage
from sglang_omni.serve import create_app
from sglang_omni.serve.protocol import ImageGenerationParams
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane


class DecodePayload(TypedDict, total=False):
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int


class ImageDecodePayload(TypedDict):
    image: str


class PipelineResult(TypedDict, total=False):
    decode: DecodePayload
    image_decode: ImageDecodePayload
    image: str


class RecordingCoordinator:
    def __init__(self) -> None:
        self.requests: list[GenerateRequest] = []

    async def submit(self, request_id: str, request: GenerateRequest) -> PipelineResult:
        self.requests.append(request)
        return {
            "decode": {
                "text": "A red square",
                "finish_reason": "length",
                "prompt_tokens": 3,
                "completion_tokens": 2,
            },
            "image_decode": {"image": "cG5n"},
        }

    async def stream(
        self, request_id: str, request: GenerateRequest
    ) -> AsyncIterator[CompleteMessage | StreamMessage]:
        raise AssertionError("Image requests must be rejected before streaming")
        yield


@pytest.fixture
def api() -> Iterator[tuple[TestClient, RecordingCoordinator]]:
    coordinator = RecordingCoordinator()
    app = create_app(
        Client(coordinator),
        model_name="served-model",
        supports_image_api=True,
    )
    with TestClient(app) as client:
        yield client, coordinator


def test_non_image_app_does_not_import_diffusion() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType

class NoDiffusion(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: list[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        if fullname == "sglang.multimodal_gen" or fullname.startswith("sglang.multimodal_gen."):
            raise ModuleNotFoundError("Diffusion is not installed", name=fullname)
        else:
            return None

sys.meta_path.insert(0, NoDiffusion())
from fastapi.testclient import TestClient
from sglang_omni.client.client import Client
from sglang_omni.serve.openai_api import create_app

app = create_app(Client(None), model_name="test", supports_image_api=False)
with TestClient(app) as client:
    assert client.get("/v1/models").status_code == 200
    assert client.post("/v1/images/generations", json={"prompt": "Draw"}).status_code == 404
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "params",
    [
        {"cfg_scale": 0.9},
        {"cfg_text_scale": -0.1},
        {"cfg_image_scale": -0.1},
        {"cfg_rescale": -0.1},
        {"cfg_rescale": 1.1},
        {"decoder_steps": 0},
        {"dllm_steps": 0},
        {"image_h": 31},
        {"image_w": 33},
        {"mode": "interleaved"},
        {"mode": "thinking"},
        {"decode_mode": "unknown"},
        {"decode_mode": "normal", "decoder_steps": 1},
    ],
)
def test_invalid_image_parameters(params: dict[str, int | float | str]) -> None:
    with pytest.raises(ValidationError):
        ImageGenerationParams(**params)


def test_image_input_without_generation_config_remains_chat(
    api: tuple[TestClient, RecordingCoordinator],
) -> None:
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": ""}], "images": ["input.png"]},
    )
    assert response.status_code == 200
    assert "image_generation" not in coordinator.requests[0].metadata


@pytest.mark.parametrize("editing", [False, True])
def test_native_image_routes_preserve_generation_controls(
    api: tuple[TestClient, RecordingCoordinator], editing: bool
) -> None:
    client, coordinator = api
    fields = {
        "prompt": "Make it red",
        "seed": 0,
        "guidance_scale": 4.0,
        "num_inference_steps": 8,
        "dllm_steps": 32,
        "decode_mode": "decoder-turbo",
        "response_format": "b64_json",
    }
    if editing:
        response = client.post(
            "/v1/images/edits",
            data=fields,
            files={"image[]": ("source.png", b"png", "image/png")},
        )
    else:
        response = client.post(
            "/v1/images/generations", json={**fields, "size": "1024x768"}
        )
    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["b64_json"] == "cG5n"
    request = coordinator.requests[0]
    assert request.metadata["output_modalities"] == ["image"]
    assert request.params["stream"] is False
    assert "choices" not in response.json()
    options = request.metadata["image_generation"]
    assert options["seed"] == 0
    assert options["cfg_scale"] == 4.0
    assert options["decoder_steps"] == 8
    assert options["dllm_steps"] == 32
    if editing:
        assert request.inputs["images"] == ["data:image/png;base64,cG5n"]
        assert "image_w" not in options
    else:
        assert (options["image_w"], options["image_h"]) == (1024, 768)


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize(
    "factory_mode,request_mode,steps,status",
    [
        ("normal", None, 1, 400),
        ("decoder-turbo", None, 1, 200),
        ("normal", "decoder-turbo", 1, 200),
        ("decoder-turbo", "normal", 1, 400),
        ("decoder-turbo", None, None, 200),
        ("normal", "decoder-turbo", None, 200),
    ],
)
def test_image_steps_follow_factory_mode(
    editing: bool,
    factory_mode: Literal["normal", "decoder-turbo"],
    request_mode: Literal["normal", "decoder-turbo"] | None,
    steps: int | None,
    status: int,
) -> None:
    config = LLaDA2UniOmniPipelineConfig(model_path="test")
    stage = config.stage_named(IMAGE_DECODE_STAGE)
    assert isinstance(stage, LLaDA2ImageDecoderStageConfig)
    stage.factory.decode_mode = factory_mode
    coordinator = RecordingCoordinator()
    app = create_app(
        Client(coordinator),
        supports_image_api=config.supports_image_api,
        image_decode_mode=config.image_decode_mode(),
    )
    fields: dict[str, str | int] = {"prompt": "Make it red"}
    if request_mode is not None:
        fields["decode_mode"] = request_mode
    if steps is not None:
        fields["num_inference_steps"] = steps
    with TestClient(app) as client:
        if editing:
            response = client.post(
                "/v1/images/edits",
                data=fields,
                files={"image": ("source.png", b"png", "image/png")},
            )
        else:
            response = client.post("/v1/images/generations", json=fields)
    assert response.status_code == status, response.text
    if status == 400:
        assert not coordinator.requests
    else:
        options = coordinator.requests[0].metadata["image_generation"]
        assert options.get("decode_mode") == request_mode
        assert ("decode_mode" in options) == (request_mode is not None)
        assert options.get("decoder_steps") == steps
        assert ("decoder_steps" in options) == (steps is not None)


def test_native_image_rejects_unsupported_controls_and_multiple_sources(
    api: tuple[TestClient, RecordingCoordinator],
) -> None:
    client, coordinator = api
    for options in ({"n": 2}, {"width": 0, "height": 1024}, {"mask": "mask.png"}):
        response = client.post(
            "/v1/images/generations", json={"prompt": "Draw", **options}
        )
        assert response.status_code == 400
    response = client.post(
        "/v1/images/edits",
        data={"prompt": "Edit"},
        files=[
            ("image", ("a.png", b"png", "image/png")),
            ("image[]", ("b.png", b"png", "image/png")),
        ],
    )
    assert response.status_code == 400
    assert coordinator.requests == []


@pytest.mark.parametrize(
    ("editing", "client_error", "message", "status"),
    [
        (
            False,
            True,
            "Requested token count exceeds the model's maximum context length",
            400,
        ),
        (
            True,
            False,
            "Request requires more tokens than the thinker KV cache can hold",
            400,
        ),
        (False, False, "Image decoder failed", 500),
        (True, True, "Image decoder failed", 500),
        (False, True, QueueFullError.MESSAGE, 503),
        (True, False, QueueFullError.MESSAGE, 503),
    ],
)
def test_native_image_pipeline_errors(
    api: tuple[TestClient, RecordingCoordinator],
    monkeypatch: pytest.MonkeyPatch,
    editing: bool,
    client_error: bool,
    message: str,
    status: int,
) -> None:
    client, coordinator = api

    async def submit(request_id: str, request: GenerateRequest) -> NoReturn:
        if client_error:
            raise ClientError(message)
        else:
            raise QueueFullError.from_message(message)

    monkeypatch.setattr(coordinator, "submit", submit)
    if editing:
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "Make it red"},
            files={"image": ("source.png", b"png", "image/png")},
        )
    else:
        response = client.post("/v1/images/generations", json={"prompt": "Draw"})
    assert response.status_code == status
    assert response.json()["detail"] == message


class ReviewControlPlane(RecordingCoordinatorControlPlane):
    def __init__(self) -> None:
        super().__init__()
        self.admitted = asyncio.Event()
        self.aborted = asyncio.Event()

    async def submit_to_stage(
        self, stage: str, endpoint: str, message: SubmitMessage
    ) -> None:
        await super().submit_to_stage(stage, endpoint, message)
        self.admitted.set()

    async def broadcast_abort(self, message: AbortMessage) -> None:
        await super().broadcast_abort(message)
        self.aborted.set()


@pytest.mark.parametrize("route", ["generations", "edits", "chat"])
def test_review_disconnect_releases_admitted_request(route: str) -> None:
    """A disconnected buffered request releases its real coordinator admission."""

    async def exercise() -> None:
        coordinator = Coordinator(
            "inproc://review-complete",
            "inproc://review-abort",
            entry_stage="preprocessing",
            max_in_flight=1,
        )
        coordinator.control_plane.close()
        transport = ReviewControlPlane()
        coordinator.control_plane = transport
        coordinator.register_stage("preprocessing", "inproc://review-stage")
        app = create_app(Client(coordinator), supports_image_api=True)
        disconnect = asyncio.Event()
        receive_count = 0
        if route == "edits":
            path = "/v1/images/edits"
            content_type = b"multipart/form-data; boundary=review"
            body = (
                b'--review\r\nContent-Disposition: form-data; name="prompt"\r\n\r\n'
                b"Make it red\r\n--review\r\nContent-Disposition: form-data; "
                b'name="url"\r\n\r\ndata:image/png;base64,cG5n\r\n--review--\r\n'
            )
        else:
            path = (
                "/v1/chat/completions" if route == "chat" else "/v1/images/generations"
            )
            content_type = b"application/json"
            payload = (
                {"messages": [{"role": "user", "content": "Draw"}]}
                if route == "chat"
                else {"prompt": "Draw"}
            )
            body = json.dumps(payload).encode()

        async def receive():
            nonlocal receive_count
            receive_count += 1
            if receive_count == 1:
                return {"type": "http.request", "body": body, "more_body": False}
            else:
                await disconnect.wait()
                return {"type": "http.disconnect"}

        async def send(message):
            pass

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", content_type)],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8000),
        }
        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(transport.admitted.wait(), timeout=10)
            disconnect.set()
            await asyncio.wait_for(transport.aborted.wait(), timeout=10)
            await asyncio.wait_for(task, timeout=10)
            print(
                f"REVIEW disconnect route={route} retained={len(coordinator.requests)} aborts={len(transport.aborts)}"
            )
            assert not coordinator.requests
            assert len(transport.aborts) == 1
            await coordinator.submit_request("next", {})
            assert "next" in coordinator.requests
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            await coordinator.stop()

    asyncio.run(exercise())


@pytest.mark.parametrize("editing", [False, True])
def test_review_normal_override_validates_inherited_step_count(
    tmp_path: Path, editing: bool
) -> None:
    """A one-step turbo factory cannot serve normal mode with inherited steps."""
    decoder = LLaDA2ImageDecoder(
        str(tmp_path), device="cuda:0", decode_mode="decoder-turbo", num_steps=1
    )
    coordinator = RecordingCoordinator()

    async def submit(request_id, request):
        coordinator.requests.append(request)
        options = request.metadata["image_generation"]
        decoder.decode(
            [1],
            1,
            1,
            decode_mode=options.get("decode_mode"),
            num_steps=options.get("decoder_steps"),
        )
        raise AssertionError("An invalid normal step count must never load weights")

    coordinator.submit = submit
    config = LLaDA2UniOmniPipelineConfig(model_path=str(tmp_path))
    stage = config.stage_named(IMAGE_DECODE_STAGE)
    stage.factory.decode_mode = "decoder-turbo"
    stage.factory.num_steps = 1
    app = create_app(
        Client(coordinator),
        supports_image_api=True,
        image_decode_mode=config.image_decode_mode(),
    )
    with TestClient(app) as client:
        fields = {"prompt": "Draw", "decode_mode": "normal"}
        if editing:
            response = client.post(
                "/v1/images/edits",
                data=fields,
                files={"image": ("source.png", b"png", "image/png")},
            )
        else:
            response = client.post("/v1/images/generations", json=fields)
    print(
        f"REVIEW inherited-steps editing={editing} dispatched={len(coordinator.requests)} status={response.status_code} body={response.text}"
    )
    assert response.status_code == 400
    assert not coordinator.requests
