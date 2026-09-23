"""Download every benchmark corpus into the container cache through the benchmark loaders.

Prints the row count each loader returns for its full corpus.

usage: python prefetch.py seedtts mmmu mmsu videoamme videomme
"""

from __future__ import annotations

import sys
import time


def load(name: str) -> int:
    if name == "seedtts":
        from benchmarks.dataset.seedtts import load_seedtts_samples

        return len(load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en"))
    if name == "mmmu":
        from benchmarks.dataset.mmmu import load_mmmu_samples

        return len(load_mmmu_samples())
    if name == "mmsu":
        from benchmarks.dataset.mmsu import load_mmsu_samples

        return len(load_mmsu_samples())
    if name == "videoamme":
        from benchmarks.dataset.videomme import load_videoamme_samples

        return len(load_videoamme_samples())
    from benchmarks.dataset.videomme import load_videomme_samples

    return len(load_videomme_samples())


for name in sys.argv[1:]:
    began = time.time()
    print(f"{name} rows {load(name)} in {time.time() - began:.0f} s", flush=True)
