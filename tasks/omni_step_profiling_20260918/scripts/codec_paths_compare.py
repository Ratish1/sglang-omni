"""Compare codec_paths_dump.py outputs of two trees: per shape, whether each path's waveform is bit identical across
the trees, and each tree's distance between its compiled and eager waveform.

usage: python3 codec_paths_compare.py A.pt B.pt
"""

import sys

import torch


def distance(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).norm() / expected.norm()).item()


def main() -> None:
    first = torch.load(sys.argv[1])
    second = torch.load(sys.argv[2])
    for name in sorted(first):
        if not name.startswith("eager "):
            continue
        else:
            pass
        shape = name.removeprefix("eager ")
        compiled = f"compiled {shape}"
        print(
            f"width, bucket {shape}: eager identical {torch.equal(first[name], second[name])}, "
            f"compiled identical {torch.equal(first[compiled], second[compiled])}, "
            f"compiled to eager {distance(first[compiled], first[name]):.2e} (A) {distance(second[compiled], second[name]):.2e} (B), "
            f"A to B eager {distance(second[name], first[name]):.2e} compiled {distance(second[compiled], first[compiled]):.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
