"""How busy the card is while requests are in flight, from an nsys report taken with the probe: the span from the
first get from the AR stage's inbox (tts_engine.in) to the last stream put from the vocoder's outbox (vocoder.out),
the union of every kernel, copy, memset and graph replay in it, and the idle intervals inside it by length.

usage: python3 card_busy_span_c20261017.py NAME=REPORT.sqlite [NAME=REPORT.sqlite ...]"""

import re
import sqlite3
import sys

QUEUE = re.compile(r"q (tts_engine\.in\.get|vocoder\.out\.put) ")
DEVICE_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET",
                 "CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
BUCKETS_US = (10, 50, 100, 500, 1000, 5000, 10000, float("inf"))


def span(db):
    columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    query = ("SELECT e.start, COALESCE(e.text, s.value) FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id"
             if "textId" in columns else "SELECT start, text FROM NVTX_EVENTS")
    times = {"tts_engine.in.get": [], "vocoder.out.put": []}
    for start, text in db.execute(query):
        if text and (match := QUEUE.match(text)):
            times[match[1]].append(start)
    return min(times["tts_engine.in.get"]), max(times["vocoder.out.put"])


def busy_intervals(db, t0, t1):
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    intervals = []
    for table in DEVICE_TABLES:
        if table in tables:
            intervals += db.execute(f"SELECT start, end FROM {table} WHERE end > ? AND start < ?", (t0, t1)).fetchall()
    merged = []
    for start, end in sorted(intervals):
        start, end = max(start, t0), min(end, t1)
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


for report in sys.argv[1:]:
    name, _, path = report.partition("=")
    db = sqlite3.connect(path)
    t0, t1 = span(db)
    merged = busy_intervals(db, t0, t1)
    busy = sum(end - start for start, end in merged)
    gaps = [b[0] - a[1] for a, b in zip(merged, merged[1:])]
    total = t1 - t0
    print(f"\n# {name}: span {total / 1e9:.2f} s, card busy {100 * busy / total:.1f} %, idle {(total - busy) / 1e9:.2f} s")
    low = 0
    for high in BUCKETS_US:
        own = [gap for gap in gaps if low * 1e3 <= gap < high * 1e3]
        print(f"  idle {low:>6} to {high:>6} us: {len(own):7d} intervals, {sum(own) / 1e9:6.2f} s "
              f"({100 * sum(own) / total:4.1f} % of the span)")
        low = high
