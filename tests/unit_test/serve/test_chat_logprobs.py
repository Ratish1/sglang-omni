# SPDX-License-Identifier: Apache-2.0
"""POST /v1/chat/completions logprobs and top_logprobs."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from sglang_omni.client.types import CompletionResult, GenerateRequest, UsageInfo
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import ChatCompletionRequest

TOKEN_LOGPROBS = [
    {
        "token": "Answer",
        "token_id": 16141,
        "logprob": -0.05,
        "top_logprobs": [
            {"token": "Answer", "token_id": 16141, "logprob": -0.05},
            {"token": "The", "token_id": 785, "logprob": -3.1},
        ],
    },
    {
        "token": " D",
        "token_id": 422,
        "logprob": -0.4,
        "top_logprobs": [
            {"token": " D", "token_id": 422, "logprob": -0.4},
            {"token": " A", "token_id": 362, "logprob": -1.1},
        ],
    },
]


class _ChatClient:
    """Captures the converted request and returns a canned CompletionResult."""

    def __init__(self, result: CompletionResult) -> None:
        self._result = result
        self.requests: list[GenerateRequest] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        del request_id, audio_format
        self.requests.append(request)
        return self._result


def _result(token_logprobs: list[dict[str, Any]] | None) -> CompletionResult:
    return CompletionResult(
        request_id="r1",
        text="Answer D",
        finish_reason="stop",
        usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        output_token_logprobs=[[-0.05, 16141], [-0.4, 422]],
        token_logprobs=token_logprobs,
    )


def _post(client: _ChatClient, body: dict[str, Any]):
    tc = TestClient(create_app(client, model_name="qwen3-omni"))
    return tc.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], **body},
    )


def test_chat_returns_openai_logprobs_block() -> None:
    client = _ChatClient(_result(TOKEN_LOGPROBS))

    resp = _post(client, {"logprobs": True, "top_logprobs": 2, "temperature": 0.0})

    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["message"]["content"] == "Answer D"
    content = choice["logprobs"]["content"]
    assert [entry["token"] for entry in content] == ["Answer", " D"]
    assert content[1] == {
        "token": " D",
        "token_id": 422,
        "logprob": -0.4,
        "bytes": [32, 68],
        "top_logprobs": [
            {"token": " D", "token_id": 422, "logprob": -0.4, "bytes": [32, 68]},
            {"token": " A", "token_id": 362, "logprob": -1.1, "bytes": [32, 65]},
        ],
    }
    extra = client.requests[0].extra_params
    assert extra["return_logprob"] is True
    assert extra["top_logprobs_num"] == 2
    assert extra["return_token_logprobs"] is True


def test_chat_without_logprobs_leaves_choice_logprobs_null() -> None:
    client = _ChatClient(_result(None))

    resp = _post(client, {})

    assert resp.status_code == 200
    assert resp.json()["choices"][0]["logprobs"] is None
    assert "return_logprob" not in client.requests[0].extra_params
    assert "return_token_logprobs" not in client.requests[0].extra_params


def test_chat_logprobs_alone_requests_zero_top_k() -> None:
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}], logprobs=True
    )

    gen = _build_chat_generate_request(req)

    assert gen.extra_params["return_logprob"] is True
    assert gen.extra_params["top_logprobs_num"] == 0


def test_chat_rejects_missing_backend_logprobs() -> None:
    client = _ChatClient(_result(None))

    resp = _post(client, {"logprobs": True})

    assert resp.status_code == 501
    assert "token_logprobs" in resp.text


def test_chat_rejects_token_logprobs_length_mismatch() -> None:
    client = _ChatClient(_result(TOKEN_LOGPROBS[:1]))

    resp = _post(client, {"logprobs": True})

    assert resp.status_code == 500
    assert "completion_tokens=2" in resp.text


def test_chat_rejects_top_logprobs_without_logprobs() -> None:
    client = _ChatClient(_result(TOKEN_LOGPROBS))

    resp = _post(client, {"top_logprobs": 2})

    assert resp.status_code == 422
    assert "logprobs" in resp.text
    assert client.requests == []


def test_chat_rejects_top_logprobs_above_limit() -> None:
    client = _ChatClient(_result(TOKEN_LOGPROBS))

    resp = _post(client, {"logprobs": True, "top_logprobs": 21})

    assert resp.status_code == 422
    assert client.requests == []


def test_chat_rejects_logprobs_with_stream() -> None:
    client = _ChatClient(_result(TOKEN_LOGPROBS))

    resp = _post(client, {"logprobs": True, "stream": True})

    assert resp.status_code == 400
    assert "stream" in resp.text
    assert client.requests == []
