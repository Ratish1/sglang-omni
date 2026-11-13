"""Byte identity of the wavs two boots wrote for the same prompts.

usage: python3 wav_identity.py <run dir> <cell directory, e.g. c1_clean> <boot a> <boot b>
"""

import hashlib
import sys
from pathlib import Path


def digests(root: Path) -> dict[str, str]:
    found = {}
    for path in root.rglob("*.wav"):
        found[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return found


def main() -> None:
    run_dir, cell, first, second = (
        Path(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
    )
    left = digests(run_dir / first / cell / "bench")
    right = digests(run_dir / second / cell / "bench")
    shared = sorted(set(left) & set(right))
    differing = [name for name in shared if left[name] != right[name]]
    print(
        f"{first} {len(left)} wavs, {second} {len(right)} wavs, shared {len(shared)}, "
        f"identical {len(shared) - len(differing)}, differ {len(differing)}"
    )
    for name in differing[:5]:
        print(f"  differs {name}")


if __name__ == "__main__":
    main()
