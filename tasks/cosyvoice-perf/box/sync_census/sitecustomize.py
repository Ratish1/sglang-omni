# SPDX-License-Identifier: Apache-2.0
"""Sync census: every synchronizing CUDA call PyTorch makes in the server process, by thread, call site and
stream. Once CUDA is initialized, torch.cuda.set_sync_debug_mode("warn") makes each synchronizing call
(non_blocking=False copies in either direction, .item(), .tolist(), stream and device synchronize) raise a
warning, and a showwarning hook counts it by (thread name, the innermost sglang_omni or sglang frame and the
innermost frame, whether the current stream is the legacy default stream). The counts and the decode steps
seen so far (FunCosyVoice3ModelRunner.sample_next_token_ids calls) go to CENSUS_OUT.<pid>.json every 5 s,
counted from the moment http://127.0.0.1:CENSUS_PORT/health answers 200.
"""

import json
import os
import sys
import threading
import time
import traceback
import warnings

OUT = os.environ.get("CENSUS_OUT")


def healthy(port: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def install() -> None:
    import torch

    while not torch.cuda.is_initialized():
        time.sleep(0.5)
    counts: dict[str, int] = {}
    samples = {"steps": 0}
    lock = threading.Lock()
    original_show = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if "synchronizing CUDA operation" not in text:
            return original_show(message, category, filename, lineno, file, line)
        stack = traceback.extract_stack()[:-1]
        own = next(
            (
                f"{frame.filename.split('site-packages/')[-1].split('/workspace/')[-1]}:{frame.lineno} {frame.name}"
                for frame in reversed(stack)
                if "sglang_omni/" in frame.filename or "/sglang/" in frame.filename
            ),
            "none",
        )
        inner = stack[-1] if stack else None
        inner_text = f"{os.path.basename(inner.filename)}:{inner.lineno}" if inner else "none"
        legacy = torch.cuda.current_stream().cuda_stream == 0
        key = f"{threading.current_thread().name} | {own} | {inner_text} | {'legacy' if legacy else 'own stream'}"
        with lock:
            counts[key] = counts.get(key, 0) + 1

    warnings.showwarning = show
    warnings.filterwarnings("always", message=".*synchronizing CUDA operation.*")
    while "sglang_omni.models.fun_cosyvoice3.model_runner" not in sys.modules:
        time.sleep(0.5)
    time.sleep(2.0)
    runner = sys.modules["sglang_omni.models.fun_cosyvoice3.model_runner"].FunCosyVoice3ModelRunner
    sample = runner.sample_next_token_ids

    def sample_next_token_ids(self, *args, **kwargs):
        samples["steps"] += 1
        return sample(self, *args, **kwargs)

    runner.sample_next_token_ids = sample_next_token_ids
    started = False
    while True:
        time.sleep(5.0)
        if not started and healthy(os.environ.get("CENSUS_PORT", "")):
            # note(ratish): startup's capture syncs are not serving; count from readiness on.
            with lock:
                counts.clear()
                samples["steps"] = 0
            torch.cuda.set_sync_debug_mode("warn")
            started = True
        with lock:
            snapshot = {"steps": samples["steps"], "counts": dict(counts)}
        with open(f"{OUT}.{os.getpid()}.json", "w") as handle:
            json.dump(snapshot, handle, indent=1)


if OUT:
    threading.Thread(target=install, name="sync-census", daemon=True).start()
