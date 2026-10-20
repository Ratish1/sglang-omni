"""run185: does 1c slow the talker? Talker decode steps with and without code2wav on the device, each arm on both cards.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/talker_overlap_queue.py
gil_wait_check.py at c16 (nsys, CUDA and OS runtime, graph-level, 15 s from 40 s): card 0 serves main (upn) then 1c
(s3z), card 1 serves 1c (s3zb) then main (upnb), the cards at once, so each card compares both arms on itself in
opposite orders. talker_overlap_read.py then gives, per arm, the talker's decode steps that overlap a code2wav
execution and those that do not. If 1c's code2wav slowed the talker, its overlapped steps would be longer than main's
on the same card while the steps without code2wav stay equal. Writes /data/pr2800/run185/ and run185_read.txt.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
from followups_merge_queue import card_pins  # noqa: E402

DATA = Path("/data/pr2800")


def main() -> None:
    cpus, node = card_pins()
    env = {**os.environ, "PR2800_CARD_CPUS": cpus, "PR2800_NODE": node}
    out = DATA / "run185"
    subprocess.run(
        ["python3", str(BOX / "gil_wait_check.py"), "--out", str(out), "--arms", "upn@0,s3z@0,s3zb@1,upnb@1"],
        env=env, stdout=open(DATA / "run185.out", "w"), stderr=subprocess.STDOUT,
    )
    subprocess.run(
        ["python3", str(BOX / "talker_overlap_read.py"), str(out)],
        cwd=BOX, env=env, stdout=open(DATA / "run185_read.txt", "w"), stderr=subprocess.STDOUT,
    )


if __name__ == "__main__":
    main()
