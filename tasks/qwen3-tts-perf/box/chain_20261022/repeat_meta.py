"""A seed-tts-eval meta.lst repeating chosen SeedTTS EN prompts, for a cap-rate A/B.

usage: python3 repeat_meta.py OUT_DIR SAMPLE_ID:REPEATS [SAMPLE_ID:REPEATS ...]   (from an omni tree)

Each prompt's reference clip is written to OUT_DIR/<sample id>.wav and its line repeated, ids suffixed _r<k> so every
request writes its own output.
"""

import shutil
import sys
from pathlib import Path

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    repeats = {item.rsplit(":", 1)[0]: int(item.rsplit(":", 1)[1]) for item in sys.argv[2:]}
    samples = {sample.sample_id: sample for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")}
    lines = []
    for sample_id, count in repeats.items():
        sample = samples[sample_id]
        shutil.copyfile(sample.ref_audio, out_dir / f"{sample_id}.wav")
        lines += [f"{sample_id}_r{k:03d}|{sample.ref_text}|{sample_id}.wav|{sample.target_text}" for k in range(count)]
    (out_dir / "meta.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} requests over {len(repeats)} prompts in {out_dir / 'meta.lst'}")


if __name__ == "__main__":
    main()
