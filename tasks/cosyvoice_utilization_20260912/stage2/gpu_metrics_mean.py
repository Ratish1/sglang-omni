"""Mean of every Nsight GPU metric over a capture, from its sqlite export.

The capture window of g1_stream_identity.sh's Nsight arm is the measured
benchmark, so the mean over all samples is the mean over the benchmark. GR
Active is the share of time any kernel was resident; SM Active is the share of
the streaming multiprocessors that were busy.

  nsys export --type sqlite --output trace.sqlite trace.nsys-rep
  python gpu_metrics_mean.py trace.sqlite
"""

from __future__ import annotations

import json
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
report = {}
for type_id, metric_id, name in db.execute(
    "SELECT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS"
):
    count, mean = db.execute(
        "SELECT COUNT(*), AVG(value) FROM GPU_METRICS WHERE typeId=? AND metricId=?",
        (type_id, metric_id),
    ).fetchone()
    if count:
        report[name] = {"samples": count, "mean": round(mean, 2)}
print(json.dumps(report, indent=1))
