# SPDX-License-Identifier: Apache-2.0
"""Window stamps for cell logs written while the seedtts client logged nothing: appends a
Benchmarking and a Results saved line from the cell's experiment.json, in the format the
census tools read. Its started_utc precedes the warmup request, so such a window opens
up to one warmup request early. Times are written in the box's local zone (UTC).

usage: python stamp_cells.py <run dir>
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

run = Path(sys.argv[1])
for experiment in sorted(run.glob("*/experiment.json")):
    cell = experiment.parent.name
    log = run / f"{cell}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    if "Benchmarking" in text and "Results saved to" in text:
        print(f"{cell}: stamped by the client")
        continue
    else:
        pass
    record = json.loads(experiment.read_text())
    started = datetime.fromisoformat(record["started_utc"]).astimezone()
    finished = datetime.fromisoformat(record["finished_utc"]).astimezone()
    stamp = "%Y-%m-%d %H:%M:%S"
    with log.open("a") as handle:
        handle.write(
            f"{started.strftime(stamp)},{started.microsecond // 1000:03d} stamp_cells INFO "
            f"Benchmarking {record['sample_count']} requests (from experiment.json)\n"
            f"{finished.strftime(stamp)},{finished.microsecond // 1000:03d} stamp_cells INFO "
            f"Results saved to {experiment.parent} (from experiment.json)\n"
        )
    print(
        f"{cell}: stamped from experiment.json {started.isoformat()} .. {finished.isoformat()}"
    )
