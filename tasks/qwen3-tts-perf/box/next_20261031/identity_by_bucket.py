"""Which reference-encoder bucket each differing reference falls in: a seeded identity root's differing WAVs grouped by
their SeedTTS EN reference clip, each clip's length at 24 kHz in encoder frames (hop 1,920), and the smallest captured
bucket at or above it (32, 48, 64, 96, 128, 192, 256; past 256 eager). Run from a tree with benchmarks/ on the path.

usage: python3 identity_by_bucket.py --root /data/next_20261101/block59/identity_codec [--run c1n]
"""

import argparse
import collections
import hashlib
from pathlib import Path

import soundfile

from benchmarks.dataset.seedtts import SEEDTTS_DATASET_ID, load_seedtts_samples

BUCKETS = (32, 48, 64, 96, 128, 192, 256)
HOP = 1920
RATE = 24000


def bucket_of(frames: int) -> str:
    fitting = [bucket for bucket in BUCKETS if bucket >= frames]
    return str(fitting[0]) if fitting else "eager"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", default="c1n")
    args = parser.parse_args()
    first = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.root / args.run / "main").rglob("*.wav")}
    second = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.root / args.run / "branch").rglob("*.wav")}
    differing = {name[: -len(".wav")] for name in first if first[name] != second.get(name)}
    samples = load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")
    clip_bucket = {}
    clip_differs = collections.defaultdict(bool)
    for sample in samples:
        info = soundfile.info(sample.ref_audio)
        frames = -(-round(info.frames * RATE / info.samplerate) // HOP)
        clip_bucket[sample.ref_audio] = (bucket_of(frames), frames)
        clip_differs[sample.ref_audio] |= sample.sample_id in differing
    totals = collections.Counter(bucket for bucket, _ in clip_bucket.values())
    hits = collections.Counter(clip_bucket[clip][0] for clip, differs in clip_differs.items() if differs)
    print(f"{args.run}: differing samples {len(differing)}, clips {sum(clip_differs.values())} of {len(clip_bucket)}")
    print("bucket | clips | clips with differing audio")
    for bucket in [str(item) for item in BUCKETS] + ["eager"]:
        print(f"{bucket} | {totals[bucket]} | {hits[bucket]}")
    edges = sorted(frames for clip, (bucket, frames) in clip_bucket.items() if clip_differs[clip])
    print(f"differing clips' frames: {edges}")


if __name__ == "__main__":
    main()
