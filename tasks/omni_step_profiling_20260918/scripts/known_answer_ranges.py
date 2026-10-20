"""Mean GPU metrics inside each NVTX range of a known-answer report.

usage: python known_answer_ranges.py REPORT.sqlite
"""

import sqlite3
import sys

METRICS = (
    "GR Active [Throughput %]",
    "SMs Active [Throughput %]",
    "SM Issue [Throughput %]",
    "Tensor Active [Throughput %]",
    "DRAM Read Bandwidth [Throughput %]",
)

db = sqlite3.connect(sys.argv[1])
strings = dict(db.execute("select id, value from StringIds"))
names = dict(db.execute("select metricName, metricId from TARGET_INFO_GPU_METRICS"))
for start, end, text, text_id in db.execute(
    "select start, end, text, textId from NVTX_EVENTS where eventType = 59 and end is not null order by start"
):
    label = text if text is not None else strings.get(text_id, "")
    values = []
    for name in METRICS:
        (mean,) = db.execute(
            "select avg(value) from GPU_METRICS where metricId = ? and timestamp between ? and ?",
            (names[name], start, end),
        ).fetchone()
        values.append(mean or 0.0)
    print(f"{label:<32} " + "  ".join(f"{name.split(' [')[0]} {value:5.1f}" for name, value in zip(METRICS, values)))
