"""Copies the SeedTTS EN reference clips into one directory for the reference-code checks.

usage: python3 stage_seedtts_wavs.py OUT_DIR
"""

import shutil
import sys
from pathlib import Path

from benchmarks.dataset.seedtts import load_seedtts_samples

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
for sample in samples:
    shutil.copy(sample.ref_audio, out / f"{sample.sample_id}.wav")
print(f"staged {len(samples)} clips into {out}")
