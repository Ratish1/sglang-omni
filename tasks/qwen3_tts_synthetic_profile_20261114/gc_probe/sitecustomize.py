"""Garbage collection pauses of every Python process started with this directory on PYTHONPATH.

With GC_PROBE_OUT set, each process appends to GC_PROBE_OUT.<pid>.gc.tsv one row per collection that
is a full one or lasts at least GC_PROBE_MIN_MS: wall time at its start, generation, milliseconds,
objects collected, and the process's tracked objects per generation after it.
"""

import gc
import os
import time

GC_PROBE_MIN_MS = 2.0


def install(prefix: str) -> None:
    handle = open(f"{prefix}.{os.getpid()}.gc.tsv", "a", buffering=1)
    started = {}

    def on_collection(phase: str, info: dict) -> None:
        if phase == "start":
            started["wall"] = time.time()
            started["clock"] = time.perf_counter()
        else:
            elapsed_ms = (time.perf_counter() - started["clock"]) * 1000.0
            if info["generation"] == 2 or elapsed_ms >= GC_PROBE_MIN_MS:
                counts = gc.get_count()
                handle.write(
                    f"{started['wall']:.3f}\t{info['generation']}\t{elapsed_ms:.2f}\t{info['collected']}\t"
                    f"{counts[0]},{counts[1]},{counts[2]}\t{gc.get_freeze_count()}\n"
                )
            else:
                pass

    gc.callbacks.append(on_collection)


if os.environ.get("GC_PROBE_OUT"):
    install(os.environ["GC_PROBE_OUT"])
else:
    pass
