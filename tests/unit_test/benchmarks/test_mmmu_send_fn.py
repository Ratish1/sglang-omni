from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from benchmarks.tasks import visual_understand


class _Response:
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
        self.payload = None

    def post(self, url: str, *, json: dict) -> _Response:
        self.payload = json
        return _Response()


def test_mmmu_image_encoding_runs_off_event_loop_thread(monkeypatch) -> None:
    event_loop_thread = threading.get_ident()
    encoding_threads: list[int] = []

    def encode_images(images) -> list[str]:
        encoding_threads.append(threading.get_ident())
        return ["data:image/png;base64,cG5n"]

    monkeypatch.setattr(visual_understand, "images_to_data_uris", encode_images)
    send_fn = visual_understand.make_mmmu_send_fn(
        "qwen3-omni",
        "http://127.0.0.1:8000/v1/chat/completions",
    )
    sample = SimpleNamespace(
        sample_id="sample-0",
        prompt="Question",
        images=[object()],
    )
    session = _Session()

    result = asyncio.run(send_fn(session, sample))

    assert result.is_success
    assert len(encoding_threads) == 1
    assert encoding_threads[0] != event_loop_thread
    assert session.payload["images"] == ["data:image/png;base64,cG5n"]
