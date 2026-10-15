# SPDX-License-Identifier: Apache-2.0
"""Image API validation, metadata forwarding, and response format."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.types import Message as ASGIMessage

from sglang_omni.admission import QueueFullError
from sglang_omni.client import Client
from sglang_omni.client.types import ClientError
from sglang_omni.models.llada2_uni.components.image_decoder import LLaDA2ImageDecoder
from sglang_omni.models.llada2_uni.components.preprocessor import SOI_TOKEN
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.messages import AbortMessage, SubmitMessage
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest, ImageGenerationParams
from tests.unit_test.fixtures.pipeline_fakes import RecordingCoordinatorControlPlane
from tests.unit_test.llada2_uni.test_preprocessor_understanding import make_preprocessor


class RecordingCoordinator:
    def __init__(self) -> None:
        self.requests = []

    async def submit(self, request_id, request):
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

    async def stream(self, request_id, request):
        raise AssertionError("Image requests must be rejected before streaming")
        yield


@pytest.fixture
def api():
    coordinator = RecordingCoordinator()
    app = create_app(
        Client(coordinator), model_name="llada2-uni", supports_image_api=True
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
import sys

class NoDiffusion(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
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
    "field,value",
    [
        ("cfg_scale", "NaN"),
        ("cfg_text_scale", "Infinity"),
        ("cfg_image_scale", "-Infinity"),
        ("cfg_rescale", "NaN"),
    ],
)
def test_http_rejects_nonfinite_cfg_without_dispatch(api, field, value):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw"}],
            "image_generation": {field: value},
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "image_generation", field]
    assert coordinator.requests == []


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
    ],
)
def test_invalid_image_parameters(params):
    with pytest.raises(ValidationError):
        ImageGenerationParams(**params)


def test_image_config_and_modalities_reach_omni_request() -> None:
    modalities = ["text", "image"]
    image_config = {"cfg_scale": 1.0, "cfg_text_scale": 0.0}
    req = ChatCompletionRequest(
        model="llada2-uni",
        messages=[{"role": "user", "content": "Make it red"}],
        images=["input.png"],
        image_generation=image_config,
        modalities=modalities,
        temperature=1.0,
        max_completion_tokens=64,
        stage_sampling={"thinker": {"temperature": 0.5, "max_new_tokens": 32}},
        stage_params={"thinker": {"return_logprob": True}},
    )
    generate = build_chat_generate_request(req)
    omni = Client.build_omni_request(generate)
    assert generate.output_modalities == modalities
    assert omni.metadata["output_modalities"] == modalities
    assert omni.metadata["image_generation"] == image_config
    assert omni.metadata["model"] == "llada2-uni"
    assert omni.inputs["images"] == ["input.png"]
    assert omni.inputs["messages"] == [{"role": "user", "content": "Make it red"}]
    assert omni.params["stream"] is False
    assert omni.params["max_new_tokens"] == 64
    assert omni.params["stage_sampling"]["thinker"]["temperature"] == 0.5
    assert omni.params["stage_sampling"]["thinker"]["max_new_tokens"] == 32
    assert omni.params["stage_params"] == req.stage_params
    assert omni.params["temperature"] == 1.0


def test_all_image_controls_preserve_explicit_values() -> None:
    messages = [{"role": "user", "content": "Draw a square"}]
    config = {
        "mode": "normal",
        "decode_mode": "decoder-turbo",
        "decoder_steps": 1,
        "seed": 0,
        "cfg_scale": 1.0,
        "cfg_text_scale": 0.0,
        "cfg_image_scale": 0.0,
        "cfg_rescale": 0.0,
        "image_h": 32,
        "image_w": 1024,
        "dllm_steps": 1,
    }
    req = ChatCompletionRequest(messages=messages, image_generation=config)
    assert build_chat_generate_request(req).metadata["image_generation"] == config
    req = ChatCompletionRequest(
        messages=messages, image_generation={"cfg_text_scale": None}
    )
    assert build_chat_generate_request(req).metadata["image_generation"] == {}
    assert (
        "image_generation"
        not in build_chat_generate_request(
            ChatCompletionRequest(messages=messages)
        ).metadata
    )


@pytest.mark.parametrize("modalities", [None, ["text"], ["image"], ["text", "image"]])
def test_chat_image_response_keeps_pr3_format_and_modality_choices(api, modalities):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw a square"}],
            "image_generation": {},
            "modalities": modalities,
        },
    )
    assert response.status_code == 200
    data = response.json()
    message = data["choices"][0]["message"]
    if modalities and "image" in modalities:
        assert message["image"] == {"data": "cG5n", "format": "png"}
    else:
        assert "image" not in message
    if modalities == ["image"]:
        assert "content" not in message
    else:
        assert message["content"] == "A red square"
    assert "audio" not in message
    assert data["choices"][0]["finish_reason"] == "length"
    assert data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert len(coordinator.requests) == 1
    assert coordinator.requests[0].metadata["image_generation"] == {}


@pytest.mark.parametrize(
    "extra",
    [
        {"image_generation": {}},
        {"modalities": ["image"]},
        {"modalities": ["text", "image"]},
        {"image_generation": {}, "modalities": ["text"]},
    ],
)
def test_image_streaming_is_rejected_before_dispatch(api, extra):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw"}],
            "stream": True,
            **extra,
        },
    )
    assert response.status_code == 400
    assert "stream=false" in response.json()["detail"]
    assert response.headers["content-type"] == "application/json"
    assert coordinator.requests == []


@pytest.mark.parametrize(
    "image_part",
    [
        {"type": "image_url", "image_url": {"url": "input.png"}},
        {"type": "image_url", "image_url": "input.png"},
        {"type": "image", "image": "input.png"},
        None,
    ],
)
@pytest.mark.parametrize("instruction", ["", "Make it red"])
def test_edit_requires_latest_user_instruction(api, image_part, instruction):
    client, coordinator = api
    body = {
        "messages": [
            {"role": "user", "content": "Earlier instruction must not count"},
            {"role": "assistant", "content": "Assistant text must not count"},
            {
                "role": "user",
                "content": [
                    *([image_part] if image_part else []),
                    {"type": "text", "text": instruction},
                ],
            },
        ],
        "image_generation": {},
        "modalities": ["image"],
    }
    if image_part is None:
        body["images"] = ["input.png"]
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == (200 if instruction else 400)
    assert len(coordinator.requests) == bool(instruction)
    if not instruction:
        assert (
            response.json()["detail"]
            == "Image editing requires a non-empty instruction"
        )


@pytest.mark.parametrize("content", ["  ", None, [{"text": None}], [{"text": 123}]])
def test_empty_or_malformed_edit_text_is_not_a_server_error(api, content):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": content}],
            "images": ["input.png"],
            "image_generation": {},
        },
    )
    assert response.status_code == 400
    assert coordinator.requests == []


def test_image_input_without_generation_config_remains_chat(api):
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": ""}], "images": ["input.png"]},
    )
    assert response.status_code == 200
    assert "image_generation" not in coordinator.requests[0].metadata


@pytest.mark.parametrize("editing", [False, True])
def test_native_image_routes_preserve_generation_controls(api, editing):
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


def test_native_image_rejects_unsupported_controls_and_multiple_sources(api):
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
    api, monkeypatch, editing, client_error, message, status
):
    client, coordinator = api

    async def submit(request_id, request):
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
def test_review_disconnect_releases_real_coordinator(route: str) -> None:
    """A closed HTTP client must release the generation slot before work completes."""

    async def exercise() -> None:
        coordinator = Coordinator(
            "inproc://review-complete",
            "inproc://review-abort",
            entry_stage="preprocessing",
            max_in_flight=1,
        )
        coordinator.control_plane.close()
        control_plane = ReviewControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocessing", "inproc://review-stage")
        app = create_app(
            Client(coordinator), model_name="llada2-uni", supports_image_api=True
        )
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

        async def receive() -> ASGIMessage:
            nonlocal receive_count
            receive_count += 1
            if receive_count == 1:
                return {"type": "http.request", "body": body, "more_body": False}
            else:
                await disconnect.wait()
                return {"type": "http.disconnect"}

        async def send(message: ASGIMessage) -> None:
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
        request_task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(control_plane.admitted.wait(), timeout=10)
            disconnect.set()
            if route == "chat":
                await asyncio.wait_for(control_plane.aborted.wait(), timeout=10)
                await asyncio.wait_for(request_task, timeout=10)
            else:
                # The admitted handler is suspended on its real completion future.
                await asyncio.sleep(0)
            retained = len(coordinator.requests)
            abort_count = len(control_plane.aborts)
            print(
                f"REVIEW disconnect route={route} retained={retained} "
                f"aborts={abort_count} receive_count={receive_count}"
            )
            if retained:
                with pytest.raises(QueueFullError):
                    await coordinator.submit_request("next-request", {})
            else:
                pass
        finally:
            request_task.cancel()
            with suppress(asyncio.CancelledError):
                await request_task
            await coordinator.stop()
        assert retained == 0
        assert abort_count == 1

    asyncio.run(exercise())


def test_review_image_config_returns_generated_image(api, monkeypatch) -> None:
    """Selecting generation without modalities must not discard the PNG."""
    client, coordinator = api
    preprocessor = make_preprocessor()
    preprocessor.soi_id = preprocessor.tokenizer.convert_tokens_to_ids(SOI_TOKEN)
    original_submit = coordinator.submit

    async def preprocess(request_id: str, request: OmniRequest):
        payload = StagePayload(request_id=request_id, request=request, data={})
        prepared = await preprocessor(payload)
        state = LLaDA2UniPipelineState.from_dict(prepared.data)
        print(f"REVIEW config-only real-preprocessor task_kind={state.task_kind}")
        assert state.task_kind == "t2i"
        return await original_submit(request_id, request)

    monkeypatch.setattr(coordinator, "submit", preprocess)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw"}],
            "image_generation": {},
        },
    )
    print(f"REVIEW config-only status={response.status_code} body={response.text}")
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["image"]["data"] == "cG5n"


def test_review_missing_requested_image_is_not_success(api, monkeypatch) -> None:
    """A missing image cannot become a successful image completion with text."""
    client, coordinator = api

    async def text_only(request_id: str, request) -> dict[str, str]:
        return {"text": "Text without an image"}

    monkeypatch.setattr(coordinator, "submit", text_only)
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw"}],
            "modalities": ["image"],
        },
    )
    print(f"REVIEW no-image status={response.status_code} body={response.text}")
    assert response.status_code == 500


def test_review_normal_one_step_is_rejected_before_dispatch(
    api, monkeypatch, tmp_path: Path
) -> None:
    """The HTTP boundary must apply the decoder's actual normal-mode step floor."""
    client, coordinator = api
    decoder = LLaDA2ImageDecoder(str(tmp_path), device="cpu")

    async def decode(request_id: str, request) -> dict[str, str]:
        coordinator.requests.append(request)
        decoder.decode(
            [1], 1, 1, num_steps=request.metadata["image_generation"]["decoder_steps"]
        )
        raise AssertionError("The invalid step count must never load weights")

    monkeypatch.setattr(coordinator, "submit", decode)
    response = client.post(
        "/v1/images/generations",
        json={"prompt": "Draw", "decode_mode": "normal", "num_inference_steps": 1},
    )
    print(
        f"REVIEW normal-one-step dispatched={len(coordinator.requests)} "
        f"status={response.status_code} body={response.text}"
    )
    assert response.status_code == 400
    assert coordinator.requests == []


def test_review_unsupported_chat_image_control_is_rejected(api) -> None:
    """A server-only decoder setting must not be silently accepted per request."""
    client, coordinator = api
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Draw"}],
            "modalities": ["image"],
            "image_generation": {"resolution_multiplier": 4},
        },
    )
    print(
        f"REVIEW unsupported-chat-control status={response.status_code} "
        f"forwarded={coordinator.requests[0].metadata}"
    )
    assert response.status_code == 422
    assert coordinator.requests == []
