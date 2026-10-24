"""A SeedTTS EN request list with long references at new widths: SHORT_ROWS SeedTTS EN rows with their own clips and
targets, and one long row after every LONG_EVERY of them, whose reference repeats another sample's clip (0.25 s gaps)
to LONG_SECONDS[i] seconds. Each long reference lands in its own multiple of the largest graph bucket (about 20.5 s
for both the speaker mel and the reference encoder), so on a fresh server each is the first clip of its width past
the graphs.

usage: python3 fresh_width_meta.py OUT_DIR   (from an omni tree, so benchmarks/ imports)
"""

import sys
from pathlib import Path

import numpy as np
import soundfile

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples

SHORT_ROWS = 1000
LONG_EVERY = 200
LONG_SECONDS = (25.0, 46.0, 66.0, 87.0, 107.0)
GAP_SECONDS = 0.25


def main() -> None:
    out_dir = Path(sys.argv[1])
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    samples = load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")
    shorts, longs = samples[:SHORT_ROWS], samples[SHORT_ROWS : SHORT_ROWS + len(LONG_SECONDS)]
    rows = []
    long_rows = iter(zip(longs, LONG_SECONDS))
    for index, sample in enumerate(shorts):
        wav_name = f"wavs/{sample.sample_id}.wav"
        (out_dir / wav_name).write_bytes(Path(sample.ref_audio).read_bytes())
        rows.append(f"{sample.sample_id}|{sample.ref_text}|{wav_name}|{sample.target_text}")
        if (index + 1) % LONG_EVERY == 0:
            long_sample, seconds = next(long_rows, (None, None))
            if long_sample is None:
                continue
            else:
                pass
            audio, rate = soundfile.read(long_sample.ref_audio, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            else:
                pass
            unit = np.concatenate([audio, np.zeros(int(GAP_SECONDS * rate), dtype=np.float32)])
            repeats = max(1, round(seconds * rate / len(unit)))
            long_audio = np.concatenate([unit] * repeats)
            wav_name = f"wavs/long_{long_sample.sample_id}_{int(seconds)}s.wav"
            soundfile.write(out_dir / wav_name, long_audio, rate, subtype="PCM_16")
            ref_text = " ".join([long_sample.ref_text.strip()] * repeats)
            rows.append(f"long_{int(seconds)}s_{long_sample.sample_id}|{ref_text}|{wav_name}|{long_sample.target_text}")
            duration_s = len(long_audio) / rate
            print(f"long {wav_name}: {duration_s:.1f} s, speaker bin {int(duration_s * 24000 / 256) // 1920}, "
                  f"reference bin {int(np.ceil(duration_s * 12.5)) // 256}")
        else:
            pass
    (out_dir / "fresh_width.lst").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows, long rows {sum(row.startswith('long_') for row in rows)}, to {out_dir}")


if __name__ == "__main__":
    main()
