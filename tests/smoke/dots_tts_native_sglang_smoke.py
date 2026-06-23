# SPDX-License-Identifier: Apache-2.0
"""GPU smoke for the SGLang-backed dots TTS latent engine."""

from __future__ import annotations

import argparse
import queue
import threading
import time

import torch

from sglang_omni.models.dots_tts.stages import (
    create_sglang_latent_engine_executor,
    create_vocoder_executor,
    preprocess_dots_tts_payload,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text", default="Hello.")
    parser.add_argument("--max-generate-length", type=int, default=12)
    parser.add_argument("--concurrent-requests", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native dots TTS smoke")
    if args.concurrent_requests <= 0:
        raise ValueError("--concurrent-requests must be positive")

    payloads = []
    for request_idx in range(args.concurrent_requests):
        payload = StagePayload(
            request_id=f"dots-native-smoke-{request_idx}",
            request=OmniRequest(
                inputs={
                    "text": args.text,
                    "max_generate_length": args.max_generate_length,
                },
                params={
                    "stream": False,
                    "max_new_tokens": args.max_generate_length,
                },
            ),
            data={},
        )
        payloads.append(preprocess_dots_tts_payload(payload))

    scheduler = create_sglang_latent_engine_executor(
        args.model_path,
        precision="bfloat16",
        max_generate_length=args.max_generate_length,
        gpu_id=0,
        server_args_overrides={
            "max_running_requests": args.concurrent_requests,
            "mem_fraction_static": 0.55,
            "disable_cuda_graph": True,
        },
    )
    native_model = scheduler._model_runner.model.native_model

    def _forbid_runtime_stream(*args, **kwargs):
        raise AssertionError("_generate_latents_stream must not run in native smoke")

    native_model._generate_latents_stream = _forbid_runtime_stream

    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for payload in payloads:
            scheduler.inbox.put(
                IncomingMessage(payload.request_id, "new_request", payload)
            )
        deadline = time.monotonic() + args.timeout_s
        streams_by_request = {payload.request_id: [] for payload in payloads}
        results_by_request = {}
        while time.monotonic() < deadline:
            try:
                msg = scheduler.outbox.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg.type == "error":
                raise RuntimeError(f"scheduler emitted error: {msg.data!r}")
            if msg.type == "stream":
                streams_by_request[msg.request_id].append(msg)
                continue
            if msg.type == "result":
                result_payload = msg.data
                results_by_request[msg.request_id] = result_payload
                data = result_payload.data
                if data.get("modality") != "audio_latents":
                    raise RuntimeError(f"unexpected result payload: {data!r}")
                latent_patches = data.get("latent_patches") or []
                if not latent_patches:
                    raise RuntimeError(
                        f"native dots latent smoke produced no patches for {msg.request_id}"
                    )
                first_patch = torch.as_tensor(latent_patches[0])
                if not bool(torch.isfinite(first_patch).all()):
                    raise RuntimeError(
                        "native dots latent patch contains non-finite values "
                        f"for {msg.request_id}"
                    )
                if tuple(first_patch.shape) != (1, 4, 128):
                    raise RuntimeError(
                        f"unexpected latent patch shape for {msg.request_id}: "
                        f"{tuple(first_patch.shape)}"
                    )
                stream_count = len(streams_by_request[msg.request_id])
                if stream_count != args.max_generate_length:
                    raise RuntimeError(
                        f"unexpected stream chunk count for {msg.request_id}: "
                        f"{stream_count}"
                    )
                if len(results_by_request) != len(payloads):
                    continue

                vocoder = create_vocoder_executor(
                    args.model_path,
                    precision="bfloat16",
                    gpu_id=0,
                )
                for request_id, request_payload in results_by_request.items():
                    decoded = vocoder._fn(request_payload)
                    if decoded.data.get("modality") != "audio":
                        raise RuntimeError(
                            f"unexpected vocoder payload for {request_id}: "
                            f"{decoded.data!r}"
                        )
                    if int(decoded.data.get("sample_rate", 0)) != 48000:
                        raise RuntimeError(
                            f"unexpected sample rate for {request_id}: "
                            f"{decoded.data.get('sample_rate')!r}"
                        )
                print(
                    "dots native sglang smoke passed "
                    f"requests={len(payloads)} "
                    f"patch_shape={tuple(first_patch.shape)} "
                    "sample_rate=48000 "
                    f"stream_chunks={args.max_generate_length}"
                )
                return 0
        raise TimeoutError("timed out waiting for dots native sglang smoke result")
    finally:
        scheduler.stop()
        thread.join(timeout=5.0)


if __name__ == "__main__":
    raise SystemExit(main())
