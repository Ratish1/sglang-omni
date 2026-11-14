"""GIL holders of a stage process from py-spy raw records (--gil --threads), base against head, per order.

Each sample is a stack held while owning the GIL. Its owner is the first of OWNERS whose marker appears
in the stack; the share of samples against py-spy's rate times the duration is how busy the GIL was.

usage: python3 gil_owners.py --root /data/.../gil_owners_c64 --duration 12
"""

import argparse
from collections import Counter
from pathlib import Path

SAMPLE_RATE_HZ = 100
OWNERS = (
    ("preprocessing", ("request_builders.py", "reference_encoder.py", "cache_key.py")),
    ("vocoder", ("streaming_vocoder.py", "codec_state_arena.py", "incremental_codec")),
    (
        "engine",
        ("omni_scheduler.py", "model_runner.py", "sglang_model.py", "sglang_backend"),
    ),
    ("stage runtime", ("pipeline/stage", "local_dispatch.py", "relay", "asyncio")),
)
TOP_FUNCTIONS = 12


def read_samples(path: Path) -> list[tuple[list[str], int]]:
    samples = []
    for line in path.read_text().splitlines():
        stack, _, count = line.rpartition(" ")
        if not count.isdigit():
            continue
        else:
            pass
        samples.append((stack.split(";"), int(count)))
    return samples


def owner_of(frames: list[str]) -> str:
    text = ";".join(frames)
    for owner, markers in OWNERS:
        if any(marker in text for marker in markers):
            return owner
        else:
            pass
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--duration", type=float, default=12)
    arguments = parser.parse_args()
    possible = SAMPLE_RATE_HZ * arguments.duration
    for order_directory in sorted(Path(arguments.root).glob("order*")):
        print(order_directory.name)
        for side in ("base", "head"):
            samples = read_samples(order_directory / f"gil_{side}.txt")
            total = sum(count for _, count in samples)
            owners = Counter()
            leaves = Counter()
            for frames, count in samples:
                owners[owner_of(frames)] += count
                leaves[frames[-1]] += count
            shares = ", ".join(
                f"{owner} {100 * count / possible:.1f} %"
                for owner, count in owners.most_common()
            )
            print(
                f"  {side}: GIL held {100 * total / possible:.1f} % of samples; {shares}"
            )
            for leaf, count in leaves.most_common(TOP_FUNCTIONS):
                print(f"      {100 * count / possible:5.1f} %  {leaf[:120]}")


if __name__ == "__main__":
    main()
