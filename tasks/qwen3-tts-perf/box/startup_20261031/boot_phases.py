"""A server log's timestamped lines as a timeline: seconds since the first line, the gap to the previous line, the
logger and the message; with --min-gap only lines after a gap of at least that many seconds.

usage: python3 boot_phases.py SERVE_LOG [--min-gap 1.0]
"""

import argparse
import datetime
import re

LINE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[(\w+)\] (\S+): (.*)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--min-gap", type=float, default=0.0)
    args = parser.parse_args()
    start = previous = None
    for line in open(args.log, errors="replace"):
        match = LINE.match(line)
        if match is None:
            continue
        else:
            pass
        moment = datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
        start = start or moment
        gap = (moment - previous).total_seconds() if previous else 0.0
        previous = moment
        if gap >= args.min_gap or "ready" in match.group(4):
            print(f"{(moment - start).total_seconds():7.1f} +{gap:6.1f} {match.group(3)[-34:]:>34} | {match.group(4)[:140]}")
        else:
            pass


if __name__ == "__main__":
    main()
