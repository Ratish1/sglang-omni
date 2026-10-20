# SPDX-License-Identifier: Apache-2.0
"""Request lists beyond the SeedTTS English cells, as seed-tts meta.lst files, for E5's memory attribution:
  zh       the first 128 SeedTTS Chinese samples as they are
  longref  64 English entries whose reference joins consecutive references up to 28 s (the speech
           tokenizer refuses more than 30 s)
  shortref 64 English entries whose reference is cut to its first 1.5 s
  sr48k    64 English entries with the reference resampled to 48 kHz
  sr8k     64 English entries with the reference resampled to 8 kHz
Texts are the samples' own target texts. Deterministic.

usage: python make_mixed_meta_c20261011.py --out DIR
"""

from __future__ import annotations

import argparse
import os
import shutil

import soundfile
import torch
import torchaudio

from benchmarks.dataset.seedtts import load_seedtts_samples

LONG_SECONDS = 28.0
SHORT_SECONDS = 1.5
ENTRIES = 64


def write_list(directory: str, lines: list[str]) -> None:
    with open(os.path.join(directory, "meta.lst"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} entries to {directory}/meta.lst")


def load(path: str) -> tuple[torch.Tensor, int]:
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T[:1].copy()), rate


def save(directory: str, name: str, audio: torch.Tensor, rate: int) -> str:
    soundfile.write(os.path.join(directory, name), audio[0].numpy(), rate)
    return name


def clean(text: str) -> str:
    return text.replace("|", " ").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    meta = "zhaochenyang20/seed-tts-eval-arrow"

    zh = load_seedtts_samples(meta, split="zh")[:128]
    directory = os.path.join(args.out, "zh")
    os.makedirs(directory, exist_ok=True)
    lines = []
    for index, sample in enumerate(zh):
        name = f"ref_{index:03d}{os.path.splitext(sample.ref_audio)[1]}"
        shutil.copyfile(sample.ref_audio, os.path.join(directory, name))
        lines.append(f"zh_{index:03d}|{clean(sample.ref_text)}|{name}|{clean(sample.target_text)}")
    write_list(directory, lines)

    en = load_seedtts_samples(meta, split="en")
    for kind in ("longref", "shortref", "sr48k", "sr8k"):
        directory = os.path.join(args.out, kind)
        os.makedirs(directory, exist_ok=True)
        lines = []
        for index in range(ENTRIES):
            sample = en[index]
            audio, rate = load(sample.ref_audio)
            ref_text = clean(sample.ref_text)
            if kind == "longref":
                pieces, texts, offset = [audio], [ref_text], index + 1
                while sum(p.shape[1] for p in pieces) < LONG_SECONDS * rate:
                    extra, extra_rate = load(en[offset].ref_audio)
                    pieces.append(torchaudio.functional.resample(extra, extra_rate, rate))
                    texts.append(clean(en[offset].ref_text))
                    offset += 1
                audio = torch.cat(pieces, dim=1)[:, : int(LONG_SECONDS * rate)]
                ref_text = " ".join(texts)
            elif kind == "shortref":
                audio = audio[:, : int(SHORT_SECONDS * rate)]
            else:
                target_rate = 48000 if kind == "sr48k" else 8000
                audio = torchaudio.functional.resample(audio, rate, target_rate)
                rate = target_rate
            name = save(directory, f"ref_{index:03d}.wav", audio, rate)
            lines.append(f"{kind}_{index:03d}|{ref_text}|{name}|{clean(sample.target_text)}")
        write_list(directory, lines)


if __name__ == "__main__":
    main()
