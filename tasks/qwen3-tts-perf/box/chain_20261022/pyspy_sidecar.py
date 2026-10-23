"""py-spy on both arms' stage processes (preprocessing, talker and vocoder share one) while a pair.py run serves:
waits for each arm's events_preprocessing_<pid>.jsonl, the stage process's first flushed events, then records that pid
without pausing it (--nonblocking) at --rate Hz for --duration seconds, idle frames included, as raw collapsed stacks
in PAIR_ROOT/<label>/<arm>/pyspy.txt. Start it alongside pair.py --events.

usage: PAIR_ROOT=<pairs> python3 pyspy_sidecar.py LABEL [--rate 100] [--duration 60] [--timeout 600]
"""

import argparse
import os
import subprocess
import time
from pathlib import Path


def stage_pid(events_dir: Path, deadline: float) -> int | None:
    while time.time() < deadline:
        found = sorted(events_dir.glob("events_preprocessing_*.jsonl")) if events_dir.exists() else []
        if found:
            return int(found[0].stem.rsplit("_", 1)[1])
        else:
            time.sleep(0.5)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--rate", type=int, default=100)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    root = Path(os.environ["PAIR_ROOT"]) / args.label
    deadline = time.time() + args.timeout
    recorders = []
    for arm in ("a", "b"):
        pid = stage_pid(root / arm / "events", deadline)
        if pid is None:
            print(f"{args.label} {arm}: no stage process within {args.timeout} s", flush=True)
            continue
        else:
            pass
        out = root / arm / "pyspy.txt"
        recorders.append(
            subprocess.Popen(
                ["py-spy", "record", "--pid", str(pid), "--rate", str(args.rate), "--duration", str(args.duration),
                 "--nonblocking", "--idle", "--threads", "--format", "raw", "--output", str(out)],
                stdout=subprocess.DEVNULL, stderr=open(root / arm / "pyspy.log", "w"),
            )
        )
        print(f"{args.label} {arm}: recording stage process {pid}", flush=True)
    for recorder in recorders:
        recorder.wait()


if __name__ == "__main__":
    main()
