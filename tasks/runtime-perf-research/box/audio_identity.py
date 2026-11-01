"""Byte identity of every request's audio between seeded passes of two or more boots.

usage: python3 audio_identity.py <root> <cell> <boot> <boot> [<boot> ...]

Each boot's <cell>/bench/audio/*.wav is hashed; every pair of boots is compared over the sample ids both hold: how
many files are byte identical, how many differ, and how many of the differing ones also differ in length. Comparing
two boots of one tree gives the boot-state baseline a change between trees is read against.
"""

import hashlib
import itertools
import sys
from pathlib import Path


def hashes(root: Path, boot: str, cell: str) -> dict[str, tuple[str, int]]:
    out = {}
    for wav in sorted((root / boot / cell / "bench" / "audio").glob("*.wav")):
        data = wav.read_bytes()
        out[wav.stem] = (hashlib.sha256(data).hexdigest(), len(data))
    return out


def main() -> None:
    root, cell, boots = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    by_boot = {boot: hashes(root, boot, cell) for boot in boots}
    for boot, files in by_boot.items():
        print(f"{boot}: {len(files)} files")
    for a, b in itertools.combinations(boots, 2):
        shared = sorted(set(by_boot[a]) & set(by_boot[b]))
        same = sum(1 for k in shared if by_boot[a][k][0] == by_boot[b][k][0])
        length = sum(1 for k in shared if by_boot[a][k][1] != by_boot[b][k][1])
        print(f"{a} against {b}: {len(shared)} shared, {same} identical, {len(shared) - same} differ "
              f"({length} of them in length)")


if __name__ == "__main__":
    main()
