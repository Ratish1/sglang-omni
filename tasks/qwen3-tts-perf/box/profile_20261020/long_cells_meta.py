"""Request lists for the long cells of profile_20261022/PLAN.md 3, as seed-tts meta.lst files.

usage: python3 long_cells_meta.py OUT_DIR   (from an omni tree, so benchmarks/ imports)

Long prefill rows repeat a sample's own reference clip (0.25 s of silence between repeats, the transcript repeated) up
to about LONG_PREFILL_POSITIONS prompt positions, one speaker, and keep the sample's own target. Long decode rows keep a
sample's own reference and join corpus targets, taken in order and wrapping, up to about LONG_DECODE_WORDS words,
about 1,900 codec frames of speech. Every row takes its own sample's reference, and the reference opens the prompt, so
no prompt is already in the prefix cache when its run is captured. Writes OUT_DIR/<cell>_{warm,run}.lst, the long
reference wavs and lengths.tsv.
"""

import math
import sys
from pathlib import Path

import numpy as np
import soundfile

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples

FRAMES_PER_SECOND = 12.5
SPECIAL_POSITIONS = 10
GAP_SECONDS = 0.25
LONG_PREFILL_POSITIONS = 1100
LONG_DECODE_WORDS = 400
# cell: (long prefill, long decode, warm requests, measured requests)
CELLS = {
    "L1_c1": (True, False, 2, 8),
    "L1_c16": (True, False, 32, 96),
    "L2_c1": (False, True, 1, 2),
    "L2_c16": (False, True, 16, 32),
    "L2_c64": (False, True, 16, 64),
    "L3_c16": (True, True, 16, 32),
}


def long_reference(sample: SampleInput, out_dir: Path) -> tuple[str, str, float]:
    audio, rate = soundfile.read(sample.ref_audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    else:
        pass
    clip_seconds = len(audio) / rate
    repeats = math.ceil(
        (LONG_PREFILL_POSITIONS - SPECIAL_POSITIONS) / FRAMES_PER_SECOND / (clip_seconds + GAP_SECONDS)
    )
    gap = np.zeros(int(GAP_SECONDS * rate), dtype=np.float32)
    long_audio = np.concatenate([np.concatenate([audio, gap]) for _ in range(repeats)])
    wav_name = f"wavs/{sample.sample_id}_x{repeats}.wav"
    soundfile.write(out_dir / wav_name, long_audio, rate, subtype="PCM_16")
    return wav_name, " ".join([sample.ref_text.strip()] * repeats), len(long_audio) / rate


def main() -> None:
    out_dir = Path(sys.argv[1])
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    samples = load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")
    cursor = 0
    text_cursor = 0
    lengths = ["cell\trow\tsample\treference_s\tprompt_positions\ttarget_words"]
    for cell, (long_prefill, long_decode, warm_count, run_count) in CELLS.items():
        for part, count in (("warm", warm_count), ("run", run_count)):
            rows = []
            for _ in range(count):
                sample = samples[cursor]
                cursor += 1
                if long_prefill:
                    wav_name, ref_text, reference_s = long_reference(sample, out_dir)
                else:
                    wav_name = f"wavs/{sample.sample_id}.wav"
                    (out_dir / wav_name).write_bytes(Path(sample.ref_audio).read_bytes())
                    ref_text = sample.ref_text
                    reference_s = soundfile.info(sample.ref_audio).duration
                if long_decode:
                    words = []
                    while len(words) < LONG_DECODE_WORDS:
                        words += samples[text_cursor % len(samples)].target_text.split()
                        text_cursor += 1
                    target_text = " ".join(words)
                else:
                    target_text = sample.target_text
                row_id = f"{cell}_{part}_{len(rows)}"
                rows.append(f"{row_id}|{ref_text}|{wav_name}|{target_text}")
                lengths.append(
                    f"{cell}\t{row_id}\t{sample.sample_id}\t{reference_s:.1f}\t"
                    f"{round(reference_s * FRAMES_PER_SECOND) + SPECIAL_POSITIONS}\t{len(target_text.split())}"
                )
            (out_dir / f"{cell}_{part}.lst").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out_dir / "lengths.tsv").write_text("\n".join(lengths) + "\n", encoding="utf-8")
    print(f"wrote {len(CELLS)} cells over {cursor} corpus samples to {out_dir}")


if __name__ == "__main__":
    main()
