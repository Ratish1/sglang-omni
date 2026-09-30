"""Build the long-form cell S2 of the workload matrix as a seed-tts meta.lst.

Each entry keeps one seed-tts sample's reference (audio copied next to the list) and
joins the target texts of that sample and the next ones, so one request speaks for
tens of seconds with a real voice-clone prompt. Deterministic: the first entries of
the seed-tts English split, in order.

usage: python make_longform_meta.py --out DIR [--entries 48 --texts-per-entry 10]
"""

from __future__ import annotations

import argparse
import os
import shutil

from benchmarks.dataset.seedtts import load_seedtts_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--entries", type=int, default=48)
    parser.add_argument("--texts-per-entry", type=int, default=10)
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()
    samples = load_seedtts_samples(args.meta, split=args.lang)
    needed = args.entries + args.texts_per_entry - 1
    assert len(samples) >= needed, f"{len(samples)} samples, need {needed}"
    os.makedirs(args.out, exist_ok=True)
    lines = []
    for index in range(args.entries):
        sample = samples[index]
        audio_name = f"ref_{index:03d}{os.path.splitext(sample.ref_audio)[1]}"
        shutil.copyfile(sample.ref_audio, os.path.join(args.out, audio_name))
        text = " ".join(
            s.target_text.strip() for s in samples[index : index + args.texts_per_entry]
        )
        ref_text = sample.ref_text.replace("|", " ")
        lines.append(
            f"longform_{index:03d}|{ref_text}|{audio_name}|{text.replace('|', ' ')}"
        )
    with open(os.path.join(args.out, "meta.lst"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} entries to {args.out}/meta.lst")


if __name__ == "__main__":
    main()
