"""Count the Python call sites of synchronizing CUDA operations in every process that initializes CUDA.

Enabled by OMNI_SYNC_PROBE=<output prefix>. A watcher thread waits for CUDA to initialize, then turns on torch's sync debug
mode in warn mode; each warning records the stack's frames inside sglang_omni and sglang. Counts go to <prefix>.<pid>.json
every few seconds, so a killed process still leaves its counts.
"""

import collections
import json
import os
import threading
import time
import traceback
import warnings

PREFIX = os.environ.get("OMNI_SYNC_PROBE")
FRAMES = 6
DUMP_SECONDS = 5.0


def start() -> None:
    counts: collections.Counter = collections.Counter()
    lock = threading.Lock()
    original_showwarning = warnings.showwarning

    def record(message, category, filename, lineno, file=None, line=None) -> None:
        if "synchronizing CUDA operation" not in str(message):
            original_showwarning(message, category, filename, lineno, file, line)
            return
        else:
            pass
        frames = [
            f"{frame.filename.split('site-packages/')[-1].split('/workspace/')[-1]}:{frame.lineno} {frame.name}"
            for frame in traceback.extract_stack()[:-1]
            if ("sglang_omni" in frame.filename or "/sglang/" in frame.filename)
            and "sync_probe" not in frame.filename
        ]
        signature = " <- ".join(reversed(frames[-FRAMES:])) or f"{filename}:{lineno}"
        with lock:
            counts[signature] += 1

    def dump() -> None:
        with lock:
            snapshot = dict(counts)
        path = f"{PREFIX}.{os.getpid()}.json"
        with open(path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(
                {"pid": os.getpid(), "counts": snapshot, "time": time.time()}, handle
            )
        os.replace(path + ".tmp", path)

    def watch() -> None:
        while True:
            torch_module = __import__("sys").modules.get("torch")
            if torch_module is not None and torch_module.cuda.is_initialized():
                break
            else:
                pass
            time.sleep(1.0)
        warnings.showwarning = record
        warnings.simplefilter("always", UserWarning)
        torch_module.cuda.set_sync_debug_mode("warn")
        while True:
            time.sleep(DUMP_SECONDS)
            dump()

    threading.Thread(target=watch, name="sync-probe", daemon=True).start()


if PREFIX:
    start()
else:
    pass
