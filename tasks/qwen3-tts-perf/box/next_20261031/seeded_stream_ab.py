"""ab_runs.py's two-order A/B with seeded streaming c16 runs only (seeds 1 and 2): each prompt draws the same random
numbers in both arms, so the arms differ only by their numerics and by batch composition, and the paired speaker
similarity bounds a change to the vocoder's numerics far tighter than unseeded runs.

usage: python3 seeded_stream_ab.py --a LABEL=TREE --b LABEL=TREE --root /data/next_20261031/<block> [--port 11000]
"""

import ab_runs

ab_runs.RUNS = (("s16seed1", 16, 1, True), ("s16seed2", 16, 2, True))

if __name__ == "__main__":
    ab_runs.main()
