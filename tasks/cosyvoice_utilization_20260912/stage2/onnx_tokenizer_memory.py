"""Device memory the ONNX speech tokenizer takes, step by step.

The serving process builds the session before SGLang sizes the KV pool and
first runs it from request threads afterwards, so whatever a run allocates
lands in the slack the pool left. This probe reads the device's free memory
after each step a server goes through: construction, a first run, the longest
run the tokenizer accepts, eight runs at once, and runs from many threads.

One provider option set per process, on a card nothing else is using:

  python onnx_tokenizer_memory.py --onnx .../speech_tokenizer_v3.onnx
  python onnx_tokenizer_memory.py --onnx ... --option use_ep_level_unified_stream=1
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime
import torch

from sglang_omni.models.fun_cosyvoice3.utils import SpeechTokenizerV3

SAMPLE_RATE = 16000
MIB = 1 << 20


def used_mib() -> float:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return (total - free) / MIB


def build(onnx_path: str, options: dict[str, str]) -> SpeechTokenizerV3:
    # The session is built as utils.py builds it, with the extra provider
    # options under test merged in.
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session_options.intra_op_num_threads = 16
    tokenizer = SpeechTokenizerV3.__new__(SpeechTokenizerV3)
    tokenizer.session = onnxruntime.InferenceSession(
        onnx_path,
        sess_options=session_options,
        providers=[
            (
                "CUDAExecutionProvider",
                {"cudnn_conv_algo_search": "HEURISTIC", **options},
            ),
            "CPUExecutionProvider",
        ],
    )
    tokenizer.device = "cuda:0"
    return tokenizer


def audio(seconds: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.standard_normal(int(seconds * SAMPLE_RATE)).astype(np.float32)


def run_together(tokenizer: SpeechTokenizerV3, threads: int, seconds: float) -> int:
    barrier = threading.Barrier(threads)
    names: set[str] = set()

    def one(index: int) -> None:
        barrier.wait()
        names.add(threading.current_thread().name)
        tokenizer.extract_speech_token(audio(seconds, index), SAMPLE_RATE)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(one, range(threads)))
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--option", action="append", default=[])
    args = parser.parse_args()
    options = dict(item.split("=", 1) for item in args.option)

    torch.zeros(1, device="cuda")
    rows: list[tuple[str, float, float]] = [("cuda context", used_mib(), 0.0)]
    started = time.monotonic()

    def mark(step: str) -> None:
        nonlocal started
        rows.append((step, used_mib(), time.monotonic() - started))
        started = time.monotonic()

    tokenizer = build(args.onnx, options)
    mark("session built")

    tokenizer.extract_speech_token(audio(3, 0), SAMPLE_RATE)
    mark("first run, 3 s, main thread")

    tokenizer.extract_speech_token(audio(3, 1), SAMPLE_RATE)
    mark("second run, 3 s, main thread")

    tokenizer.extract_speech_token(audio(30, 2), SAMPLE_RATE)
    mark("run, 30 s, main thread")

    distinct = run_together(tokenizer, 8, 10)
    mark(f"8 runs at once, 10 s, {distinct} threads")

    distinct = run_together(tokenizer, 8, 30)
    mark(f"8 runs at once, 30 s, {distinct} threads")

    distinct = run_together(tokenizer, 32, 3)
    mark(f"32 runs at once, 3 s, {distinct} threads")

    # The steady state a server reaches: the same mixed load again, which
    # should allocate nothing and shows what a warm run costs in time.
    distinct = run_together(tokenizer, 8, 10)
    mark(f"again, 8 runs at once, 10 s, {distinct} threads")

    for index in range(8):
        tokenizer.extract_speech_token(audio(10, index), SAMPLE_RATE)
    mark("again, 8 runs in a row, 10 s, main thread")

    report = {
        "onnxruntime": onnxruntime.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "options": options,
        "rows": [
            {
                "step": step,
                "used_mib": round(used, 1),
                "delta_mib": round(used - rows[max(index - 1, 0)][1], 1),
                "seconds": round(seconds, 3),
            }
            for index, (step, used, seconds) in enumerate(rows)
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
