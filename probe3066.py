"""Exercise the actual Qwen preprocessing and reference thread device contracts."""

import argparse
import ctypes
import json
import threading
from types import SimpleNamespace

import numpy as np
import torch

from sglang_omni.models.qwen3_tts import request_builders, stages
from sglang_omni.scheduling.message import IncomingMessage


def primary_context_active(index):
    driver = ctypes.CDLL("libcuda.so.1")
    flags, active = ctypes.c_uint(), ctypes.c_int()
    status = driver.cuDevicePrimaryCtxGetState(
        index, ctypes.byref(flags), ctypes.byref(active)
    )
    assert status == 0, status
    return bool(active.value)


def preprocessing():
    torch.cuda.set_device(1)
    stream = torch.cuda.Stream(device=1)
    values = torch.arange(256, device="cuda:1", dtype=torch.float32)
    expected = values.square() + 3
    stream.wait_stream(torch.cuda.current_stream(1))
    barrier = threading.Barrier(2)
    request_builders._PREPROCESSING_CONTEXT = SimpleNamespace(stream=stream)

    def compute(payload, **kwargs):
        before = torch.cuda.current_device()
        barrier.wait(timeout=10)
        with torch.cuda.stream(stream):
            actual = values.square() + 3
        after = torch.cuda.current_device()
        return before, after, actual

    stages.preprocess_qwen3_tts_payload = compute
    scheduler = stages.create_preprocessing_executor("unused", max_concurrency=2)
    worker = threading.Thread(target=scheduler.start)
    initial = primary_context_active(0)
    worker.start()
    try:
        for index in range(2):
            scheduler.enqueue(
                IncomingMessage(request_id=str(index), type="new_request", data=None)
            )
        results = [scheduler.outbox.get(timeout=20) for _ in range(2)]
        for result in results:
            assert result.type == "result", result.data
            assert torch.equal(result.data[2], expected)
        stream.synchronize()
        print(
            json.dumps(
                dict(
                    path="preprocessing",
                    thread_devices=[result.data[:2] for result in results],
                    other_context_before=initial,
                    other_context_after=primary_context_active(0),
                    exact_outputs=True,
                )
            ),
            flush=True,
        )
    finally:
        scheduler.stop()
        worker.join(timeout=10)
        assert not worker.is_alive()
        scheduler.executor.shutdown(wait=True)


def reference():
    torch.cuda.set_device(1)
    device = torch.device("cuda", 1)
    encoder = torch.nn.Linear(1, 1, bias=False, device=device)
    recorded = []

    def encode(values, **kwargs):
        recorded.append(torch.cuda.current_device())
        frames = values.shape[-1] // 8
        return SimpleNamespace(
            audio_codes=torch.arange(frames, device=device)
            .view(1, 1, frames)
            .expand(1, 2, frames)
        )

    encoder.encode = encode
    tokenizer = SimpleNamespace(
        model=SimpleNamespace(
            encoder=encoder, encode_downsample_rate=8, encoder_valid_num_quantizers=2
        ),
        _normalize_audio_inputs=lambda waveforms, sr: [
            np.asarray(value, dtype=np.float32) for value in waveforms
        ],
    )
    initial = primary_context_active(0)
    batcher = request_builders.Qwen3TTSRefCodeBatcher(tokenizer, max_batch_wait_ms=0)
    try:
        actual = batcher.encode(np.arange(17, dtype=np.float32), 24000)
        expected = torch.arange(3, device=device).view(3, 1).expand(3, 2)
        assert torch.equal(actual, expected)
        print(
            json.dumps(
                dict(
                    path="reference",
                    encode_devices=recorded,
                    other_context_before=initial,
                    other_context_after=primary_context_active(0),
                    exact_outputs=True,
                )
            ),
            flush=True,
        )
    finally:
        batcher.close()
        assert not batcher.thread.is_alive()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", choices=("preprocessing", "reference"))
    args = parser.parse_args()
    if args.path == "preprocessing":
        preprocessing()
    else:
        reference()
