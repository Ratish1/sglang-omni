"""Independent transport and real-serving checks for PR 2730."""

import asyncio
import hashlib
import io
import json
import multiprocessing
import pickle
import sys
import time
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import soundfile

from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.stage_io import read_payload, write_payload
from sglang_omni.models.moss_tts_local.request_builders import (
    build_sglang_moss_tts_local_request,
    clear_moss_tts_local_preprocessing_context,
    preprocess_moss_tts_local_payload,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.relay.shm import ShmRelay
from tests.unit_test.moss_tts_local.test_pipeline import FakeProcessor

RUN_DIRECTORY = Path("/data/pr2730-review")
MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
REFERENCE_URL = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"


def produce_prepared_payload(connection: Connection) -> None:
    async def produce() -> None:
        set_moss_tts_local_preprocessing_context(processor=FakeProcessor())
        relay = ShmRelay(engine_id="producer")
        try:
            for index in range(8):
                request_id = f"wire-{index}"
                payload = preprocess_moss_tts_local_payload(
                    StagePayload(
                        request_id=request_id,
                        request=OmniRequest(
                            inputs={"text": "different prompt " * (index + 1)},
                            params={
                                "stream": index % 2 == 0,
                                "seed": index,
                                "audio_top_k": index + 2,
                                "max_new_tokens": 50,
                            },
                            metadata={},
                        ),
                        data={},
                    )
                )
                reference, operation = await write_payload(
                    relay,
                    request_id,
                    payload,
                    transport=TransportKind.SHM,
                    from_stage="preprocessing",
                    to_stage="tts_engine",
                )
                clear_moss_tts_local_preprocessing_context()
                connection.send_bytes(pickle.dumps(reference))
                assert connection.recv_bytes() == b"consumed"
                operation.mark_receiver_done()
                await operation.wait_for_completion()
                set_moss_tts_local_preprocessing_context(processor=FakeProcessor())
        finally:
            clear_moss_tts_local_preprocessing_context()
            connection.close()

    asyncio.run(produce())


def check_shared_memory() -> None:
    context = multiprocessing.get_context("spawn")
    consumer_connection, producer_connection = context.Pipe()
    producer = context.Process(
        target=produce_prepared_payload, args=(producer_connection,)
    )
    producer.start()
    producer_connection.close()

    async def consume() -> None:
        for index in range(8):
            assert consumer_connection.poll(60)
            reference = pickle.loads(consumer_connection.recv_bytes())
            payload = await read_payload(
                ShmRelay(engine_id="consumer"),
                f"wire-{index}",
                reference,
                local_device="cuda:0",
            )
            request = build_sglang_moss_tts_local_request(
                payload,
                model=SimpleNamespace(config=FakeProcessor.model_config),
            )
            text_length = len("different prompt " * (index + 1))
            assert request.prompt_rows.shape == (max(4, text_length % 7 + 4), 13)
            assert request.prompt_rows.device.type == "cpu"
            assert request.prompt_rows[-1, 0] == 151669
            assert request.req.origin_input_ids == request.input_ids.tolist()
            assert request.seed == index
            assert request.audio_top_k == index + 2
            assert bool(request.stream_metadata) == (index % 2 == 0)
            assert request.state.text == "different prompt " * (index + 1)
            consumer_connection.send_bytes(b"consumed")

    try:
        asyncio.run(consume())
        producer.join(60)
        assert producer.exitcode == 0
    finally:
        consumer_connection.close()
        if producer.is_alive():
            producer.terminate()
            producer.join()
        else:
            pass
    print(json.dumps({"shared_memory_payloads": 8, "producer_context_cleared": True}))


def check_serving(label: str) -> None:
    reference_path = RUN_DIRECTORY / "reference.wav"
    if not reference_path.exists():
        response = httpx.get(REFERENCE_URL, follow_redirects=True, timeout=120)
        response.raise_for_status()
        reference_path.write_bytes(response.content)
    else:
        pass
    samples = []
    with httpx.Client(timeout=180) as client:
        for index in range(16):
            streaming = index % 2 == 0
            with_reference = index % 4 < 2
            body = {
                "model": MODEL,
                "voice": "default",
                "input": "Get the trust fund to the bank early.",
                "stream": streaming,
                "response_format": "pcm" if streaming else "wav",
                "seed": 42,
                "max_new_tokens": 128,
                "token_count": 50,
            }
            if with_reference:
                body["ref_audio"] = str(reference_path)
                body["ref_text"] = (
                    "We asked over twenty different people, and they all said it was his."
                )
            else:
                pass
            started = time.perf_counter()
            chunks = []
            first_ms = None
            with client.stream(
                "POST", "http://127.0.0.1:8273/v1/audio/speech", json=body
            ) as response:
                assert response.status_code == 200, response.read()
                for chunk in response.iter_bytes():
                    if chunk and first_ms is None:
                        first_ms = (time.perf_counter() - started) * 1000
                    else:
                        pass
                    chunks.append(chunk)
            blob = b"".join(chunks)
            total_ms = (time.perf_counter() - started) * 1000
            assert len(blob) > 0
            if streaming:
                waveform = np.frombuffer(blob, dtype="<i2")
                sample_rate = 48000
            else:
                waveform, sample_rate = soundfile.read(io.BytesIO(blob), always_2d=True)
                assert waveform.shape[1] == 2
            assert sample_rate == 48000
            assert np.isfinite(waveform).all()
            assert np.any(waveform != 0)
            record = {
                "index": index,
                "streaming": streaming,
                "reference": with_reference,
                "first_ms": first_ms,
                "total_ms": total_ms,
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
            samples.append(record)
            print(json.dumps(record), flush=True)
    (RUN_DIRECTORY / f"{label}-requests.json").write_text(json.dumps(samples, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "wire":
        check_shared_memory()
    else:
        check_serving(sys.argv[1])
