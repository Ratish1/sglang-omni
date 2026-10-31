"""How a long talker reply repeats: for each capture past MIN_FRAMES frames (and a sample of normal ones), the share of
frames whose first code, and whose whole 16-code frame, already occurred within the previous WINDOW frames, the
first frame of the longest run of such repeats, and the reply's text length.

usage: python3 loop_read.py /data/q104/capture [MIN_FRAMES]
"""

import glob
import os
import random
import sys

import torch

WINDOW = 12
MIN_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 150


def repeats(codes: torch.Tensor) -> tuple[float, float, int, int]:
    first = codes[:, 0].tolist()
    frames = [tuple(row) for row in codes.tolist()]
    first_hits = 0
    frame_hits = 0
    run = 0
    best = (0, 0)
    for index in range(len(frames)):
        window = range(max(0, index - WINDOW), index)
        first_hit = any(first[j] == first[index] for j in window)
        frame_hit = any(frames[j] == frames[index] for j in window)
        first_hits += first_hit
        frame_hits += frame_hit
        run = run + 1 if first_hit else 0
        if run > best[0]:
            best = (run, index - run + 1)
        else:
            pass
    total = max(len(frames), 1)
    return first_hits / total, frame_hits / total, best[0], best[1]


paths = glob.glob(os.path.join(sys.argv[1], "*.pt"))
long_paths = []
normal_paths = []
for path in paths:
    frames = torch.load(path, map_location="cpu", weights_only=False)["codes"].shape[0]
    if frames >= MIN_FRAMES:
        long_paths.append(path)
    else:
        normal_paths.append(path)
random.seed(0)
for label, chosen in (
    ("long", long_paths),
    ("normal sample", random.sample(normal_paths, min(40, len(normal_paths)))),
):
    first_shares = []
    frame_shares = []
    for path in chosen:
        record = torch.load(path, map_location="cpu", weights_only=False)
        codes = record["codes"]
        first_share, frame_share, run, start = repeats(codes)
        first_shares.append(first_share)
        frame_shares.append(frame_share)
        if label == "long":
            print(
                f"{os.path.basename(path)} frames {codes.shape[0]} reply tokens {len(record['thinker_token_ids'])} "
                f"first code repeats {first_share:.2f} whole frame repeats {frame_share:.3f} "
                f"longest first-code repeat run {run} from frame {start}"
            )
        else:
            pass
    if chosen:
        print(
            f"{label}: {len(chosen)} captures, mean first code repeat share {sum(first_shares) / len(chosen):.2f}, "
            f"mean whole frame repeat share {sum(frame_shares) / len(chosen):.3f}"
        )
    else:
        print(f"{label}: none")
