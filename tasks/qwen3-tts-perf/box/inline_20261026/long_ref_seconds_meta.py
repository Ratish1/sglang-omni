"""A seed-tts meta list whose references are each SeedTTS EN sample's own clip repeated to at least SECONDS (0.25 s of
silence between repeats, the transcript repeated), each with the sample's own target, written as 16-bit PCM; a
reference past the API's 10 MiB ref_audio limit stops the build.

usage: python3 long_ref_seconds_meta.py OUT_DIR SECONDS COUNT   (from an omni tree, so benchmarks/ imports)

Writes OUT_DIR/ref<SECONDS>s.lst, the wavs under OUT_DIR/wavs and OUT_DIR/ref<SECONDS>s_lengths.tsv.
"""

import math
import sys
from pathlib import Path

import numpy as np
import soundfile

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.serve.speech_limits import MAX_REFERENCE_AUDIO_BYTES

GAP_SECONDS = 0.25


def main() -> None:
    out_dir = Path(sys.argv[1])
    seconds = float(sys.argv[2])
    count = int(sys.argv[3])
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    rows, lengths = [], ["sample\trepeats\treference_s\trate\tbytes"]
    for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, count, split="en"):
        audio, rate = soundfile.read(sample.ref_audio, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        else:
            pass
        repeats = math.ceil(seconds / (len(audio) / rate + GAP_SECONDS))
        gap = np.zeros(int(GAP_SECONDS * rate), dtype=np.float32)
        long_audio = np.concatenate([np.concatenate([audio, gap]) for _ in range(repeats)])
        wav_path = out_dir / "wavs" / f"{sample.sample_id}_x{repeats}.wav"
        soundfile.write(wav_path, long_audio, rate, subtype="PCM_16")
        wav_bytes = wav_path.stat().st_size
        assert wav_bytes <= MAX_REFERENCE_AUDIO_BYTES, f"{wav_path} is {wav_bytes} bytes"
        ref_text = " ".join([sample.ref_text.strip()] * repeats)
        rows.append(f"{sample.sample_id}|{ref_text}|wavs/{wav_path.name}|{sample.target_text}")
        lengths.append(f"{sample.sample_id}\t{repeats}\t{len(long_audio) / rate:.1f}\t{rate}\t{wav_bytes}")
    name = f"ref{seconds:g}s"
    (out_dir / f"{name}.lst").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out_dir / f"{name}_lengths.tsv").write_text("\n".join(lengths) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} references of at least {seconds:g} s to {out_dir}")


if __name__ == "__main__":
    main()
