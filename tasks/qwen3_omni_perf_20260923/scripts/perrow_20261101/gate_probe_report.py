"""The talker gate's figures per pass, from the GATEPROBE lines of a serve log (talker_gate_probe) and the pass
windows the queue logged ("window <label> <begin epoch> <end epoch>").

Per window, the deltas of the last GATEPROBE line at or before its end against the last at or before its begin (lines
come at most once a second, so each edge is within about a second): decode steps and their mean rows, deferred
iterations (each a prepared and rolled back decode) per decode step, episodes, the gated wall time and its share of
the window, and per episode the rows held and how many of them were unready (the rest could have run).

usage: python3 gate_probe_report.py SERVE_LOG QUEUE_LOG LABEL
"""

from __future__ import annotations

import argparse
import re

PROBE = re.compile(r"GATEPROBE wall=([\d.]+) pid=(\d+) (.*)$")
WINDOW = re.compile(r"window (\S+) ([\d.]+) ([\d.]+)")


def read_probe(path: str) -> list[tuple[float, dict[str, float]]]:
    lines = []
    with open(path, errors="replace") as handle:
        for line in handle:
            match = PROBE.search(line)
            if match is None:
                continue
            else:
                pass
            fields = dict(item.split("=") for item in match.group(3).split())
            fields["pid"] = match.group(2)
            lines.append((float(match.group(1)), fields))
    lines.sort(key=lambda row: row[0])
    return lines


def at_or_before(
    lines: list[tuple[float, dict[str, float]]], moment: float
) -> dict[str, float]:
    chosen = None
    for wall, fields in lines:
        if wall <= moment:
            chosen = fields
        else:
            break
    return chosen or {key: 0.0 for key in (lines[0][1] if lines else {})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("serve_log")
    parser.add_argument("queue_log")
    parser.add_argument(
        "label",
        help="only windows whose label starts with this (the boot the serve log belongs to)",
    )
    args = parser.parse_args()
    lines = read_probe(args.serve_log)
    pids = sorted({fields.pop("pid") for _, fields in lines})
    if len(pids) > 1:
        raise SystemExit(
            f"GATEPROBE lines from {len(pids)} processes {pids}; one talker scheduler expected"
        )
    else:
        pass
    lines = [
        (wall, {key: float(value) for key, value in fields.items()})
        for wall, fields in lines
    ]
    print(f"{len(lines)} GATEPROBE lines, talker scheduler pid {pids}")
    with open(args.queue_log) as handle:
        windows = [WINDOW.search(line) for line in handle]
    for match in filter(None, windows):
        label, begin, end = match.group(1), float(match.group(2)), float(match.group(3))
        if not label.startswith(args.label):
            continue
        else:
            pass
        first, last = at_or_before(lines, begin), at_or_before(lines, end)
        delta = {key: last.get(key, 0.0) - first.get(key, 0.0) for key in last}
        steps = delta.get("decode_steps", 0.0)
        episodes = delta.get("episodes", 0.0)
        span = end - begin
        print(
            f"{label}: window {span:.1f} s; decode steps {steps:.0f} (mean rows "
            f"{delta.get('rows_run', 0) / steps if steps else 0:.1f}); deferred iterations {delta.get('deferred', 0):.0f} "
            f"({delta.get('deferred', 0) / steps if steps else 0:.2f} per decode step); episodes {episodes:.0f} "
            f"({episodes / steps if steps else 0:.1%} of decode steps); gated {delta.get('gated_s', 0):.2f} s "
            f"({delta.get('gated_s', 0) / span if span else 0:.1%} of the window, "
            f"{1e3 * delta.get('gated_s', 0) / episodes if episodes else 0:.1f} ms an episode); per episode rows held "
            f"{delta.get('episode_rows', 0) / episodes if episodes else 0:.1f}, unready "
            f"{delta.get('episode_unready', 0) / episodes if episodes else 0:.2f}"
        )


if __name__ == "__main__":
    main()
