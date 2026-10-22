"""A seed-tts-eval meta.lst repeating the SeedTTS EN prompts that have run to the frame cap, for a cap-rate A/B.

usage: python3 runaway_meta.py OUT_DIR

Each prompt's reference clip is written to OUT_DIR/<sample id>.wav and its line repeated, ids suffixed _r<k> so every
request writes its own output.
"""

import shutil
import sys
from pathlib import Path

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples

REPEATS = {
    "common_voice_en_19916471-common_voice_en_19916474": 100,
    "common_voice_en_24985801-common_voice_en_24985802": 40,
    "common_voice_en_19770519-common_voice_en_19770516": 40,
    "common_voice_en_25745884-common_voice_en_25745889": 40,
    "common_voice_en_20095636-common_voice_en_20095638": 40,
    "common_voice_en_17275818-common_voice_en_17275816": 40,
}


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = {sample.sample_id: sample for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")}
    lines = []
    for sample_id, repeats in REPEATS.items():
        sample = samples[sample_id]
        shutil.copyfile(sample.ref_audio, out_dir / f"{sample_id}.wav")
        lines += [f"{sample_id}_r{k:03d}|{sample.ref_text}|{sample_id}.wav|{sample.target_text}" for k in range(repeats)]
    (out_dir / "meta.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} requests over {len(REPEATS)} prompts in {out_dir / 'meta.lst'}")


if __name__ == "__main__":
    main()
