# SPDX-License-Identifier: Apache-2.0
"""The vocoder's admissible worst step as a seed-tts meta.lst: N identical requests, each the first
seed-tts English sample's reference with the target texts of the following samples joined up to the
API's text limit (4,096 characters, kept under 4,060), so with one seed their finals land in one step
near the AR's context. Rebuilds the 10-10 worst/ and worst32/ lists.

usage: python make_worst_meta_c20261011.py --out DIR --requests 16
"""

from __future__ import annotations

import argparse
import os
import shutil

from benchmarks.dataset.seedtts import load_seedtts_samples

TEXT_LIMIT = 4060


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--requests", type=int, required=True)
    args = parser.parse_args()
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
    os.makedirs(args.out, exist_ok=True)
    reference = samples[0]
    audio_name = f"ref_000{os.path.splitext(reference.ref_audio)[1]}"
    shutil.copyfile(reference.ref_audio, os.path.join(args.out, audio_name))
    text = ""
    for sample in samples[1:]:
        piece = sample.target_text.strip().replace("|", " ")
        if len(text) + 1 + len(piece) > TEXT_LIMIT:
            break
        text = f"{text} {piece}".strip()
    ref_text = reference.ref_text.replace("|", " ")
    lines = [
        f"worst_{index:03d}|{ref_text}|{audio_name}|{text}"
        for index in range(args.requests)
    ]
    with open(os.path.join(args.out, "meta.lst"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} requests of {len(text)} characters to {args.out}/meta.lst")


if __name__ == "__main__":
    main()
