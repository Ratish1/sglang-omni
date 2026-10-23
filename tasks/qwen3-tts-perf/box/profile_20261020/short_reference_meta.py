"""The long cells' run lists with each long reference replaced by its sample's own SeedTTS EN clip, for speaker
similarity: a long reference is that clip repeated, and the scorer's WavLM runs out of memory on 88 s references
(its batch pads every clip to the longest). Row ids, reference texts and targets keep their sample's values.

usage: python3 short_reference_meta.py LONG_DIR OUT_DIR CELL [CELL ...]   (from an omni tree, so benchmarks/ imports)
Writes OUT_DIR/<cell>_run.lst and the clips under OUT_DIR/wavs/.
"""

import sys
from pathlib import Path

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples


def main() -> None:
    long_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    samples = {sample.sample_id: sample for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")}
    sample_by_row = {}
    for line in (long_dir / "lengths.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        _, row_id, sample_id = line.split("\t")[:3]
        sample_by_row[row_id] = sample_id
    for cell in sys.argv[3:]:
        rows = []
        for line in (long_dir / f"{cell}_run.lst").read_text(encoding="utf-8").splitlines():
            row_id, _, _, target_text = line.split("|", 3)
            sample = samples[sample_by_row[row_id]]
            wav_name = f"wavs/{sample.sample_id}.wav"
            (out_dir / wav_name).write_bytes(Path(sample.ref_audio).read_bytes())
            rows.append(f"{row_id}|{sample.ref_text}|{wav_name}|{target_text}")
        (out_dir / f"{cell}_run.lst").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print(f"{cell}: {len(rows)} rows")


if __name__ == "__main__":
    main()
