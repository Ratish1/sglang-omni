from __future__ import annotations

import asyncio

from benchmarks.dataset import mmmu
from benchmarks.eval.benchmark_omni_rollout_stress import (
    _duplicate_prompt,
    _make_rollout_send_fn,
)
from benchmarks.tasks.visual_understand import make_mmmu_send_fn


class _Image:
    def convert(self, mode: str):
        return self


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> dict:
        return {
            "choices": [{"message": {"content": "Answer: A"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }


class _Session:
    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def post(self, url: str, *, json: dict) -> _Response:
        self.payloads.append(json)
        return _Response()


def _sample() -> mmmu.MMMUSample:
    return mmmu.MMMUSample(
        sample_id="sample-0",
        question="Question",
        options=["Option"],
        answer="A",
        image_data_uris=("data:image/png;base64,cG5n",),
        subject="Math",
        prompt="Question\nA. Option",
    )


def test_mmmu_dataset_prepares_image_data_uris(monkeypatch) -> None:
    image = _Image()
    encoded_images: list[_Image] = []

    def encode_image(value: _Image) -> str:
        encoded_images.append(value)
        return "data:image/png;base64,cG5n"

    monkeypatch.setattr(mmmu, "image_to_data_uri", encode_image)
    dataset = [
        {
            "id": "sample-0",
            "__subject__": "Math",
            "question": "Question",
            "options": ["Option"],
            "answer": "A",
            "image_1": image,
        }
    ]

    samples = mmmu._dataset_to_samples(dataset, max_samples=None)

    assert encoded_images == [image]
    assert samples[0].image_data_uris == ("data:image/png;base64,cG5n",)


def test_mmmu_sender_uses_prepared_image_payload() -> None:
    session = _Session()
    send_fn = make_mmmu_send_fn(
        "qwen3-omni",
        "http://127.0.0.1:8000/v1/chat/completions",
    )

    result = asyncio.run(send_fn(session, _sample()))

    assert result.is_success
    assert session.payloads[0]["images"] == ["data:image/png;base64,cG5n"]


def test_rollout_duplicates_reuse_prepared_image_payload() -> None:
    sample = _sample()
    duplicated = _duplicate_prompt(sample, 2)
    session = _Session()
    send_fn = _make_rollout_send_fn(
        model_name="qwen3-omni",
        api_url="http://127.0.0.1:8000/v1/chat/completions",
        rollout_group_id="rollout:sample-0",
        max_tokens=32,
        temperature=0.0,
        enable_audio=False,
        talker_max_new_tokens=None,
    )

    async def send_all():
        return await asyncio.gather(*(send_fn(session, item) for item in duplicated))

    results = asyncio.run(send_all())

    assert all(result.is_success for result in results)
    assert [payload["images"] for payload in session.payloads] == [
        ["data:image/png;base64,cG5n"],
        ["data:image/png;base64,cG5n"],
    ]
