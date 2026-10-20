"""Prefix graph tier ladders scored on the frames real hop steps replayed (10-20, L6 of DESIGN_PR_STACK_20261020.md):
the frames= of every flow.prefix_graph range of a profile (cosy_profile_probe), each ladder's padded share of the
replayed frames, the device time the padding costs at the measured 0.021 ms a frame (graph_padding_c20261020.py,
both graph kinds), and its tier count (about 0.5 s of capture each). Ladders: today's (the decode row ladder times
the longest hop) and SGLang's prefill ladder over frames from the shortest hop.
usage: python3 prefix_ladder_c20261020.py <name>=<profile.sqlite> ... [--max-rows 16 --hop-frames 200 --min-hop-frames 50]"""

import argparse
import bisect
import re
import sqlite3

from sglang.srt.arg_groups.cuda_graph_hook import generate_prefill_cuda_graph_batch_sizes

RANGE = re.compile(r"flow\.prefix_graph rows=(\d+) frames=(\d+)")
MS_PER_FRAME = 0.021


def row_tiers(max_rows: int) -> list[int]:
    return sorted({rows for rows in (1, 2, 4, 8, 12, *range(16, max_rows + 1, 8)) if rows <= max_rows} | {max_rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+")
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--hop-frames", type=int, default=200)
    parser.add_argument("--min-hop-frames", type=int, default=50)
    args = parser.parse_args()
    top = args.max_rows * args.hop_frames
    ladders = {
        "today": [rows * args.hop_frames for rows in row_tiers(args.max_rows)],
        "prefill": [
            *(size for size in generate_prefill_cuda_graph_batch_sizes(top - 1) if size >= args.min_hop_frames),
            top,
        ],
    }
    for argument in args.profiles:
        name, path = argument.split("=", 1)
        cursor = sqlite3.connect(path).cursor()
        frames = [
            int(match.group(2))
            for (text,) in cursor.execute(
                "select COALESCE(e.text, s.value) from NVTX_EVENTS e left join StringIds s on e.textId = s.id"
                " where COALESCE(e.text, s.value) like 'flow.prefix_graph%'"
            )
            if (match := RANGE.search(text))
        ]
        print(f"{name}: {len(frames)} prefix replays, real frames {sum(frames)}")
        for ladder_name, ladder in ladders.items():
            tiers = [ladder[bisect.bisect_left(ladder, count)] for count in frames if count <= ladder[-1]]
            padded = sum(tiers) - sum(count for count in frames if count <= ladder[-1])
            print(
                f"  {ladder_name:8s} tiers {len(ladder):3d}: padded {100 * padded / sum(tiers):.1f} % of replayed frames, "
                f"{MS_PER_FRAME * padded / len(frames):.2f} ms of device time a replay"
            )


if __name__ == "__main__":
    main()
