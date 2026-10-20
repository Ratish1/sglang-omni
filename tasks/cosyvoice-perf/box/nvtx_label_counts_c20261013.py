# Count every NVTX range and mark in an nsys SQLite export by its label's first word (loop.step, flow.hop, ...), so a
# probe target that a code change renamed shows up as a missing or zero label.
# usage: python3 nvtx_label_counts_c20261013.py <trace.sqlite>
import sqlite3
import sys
from collections import Counter

connection = sqlite3.connect(sys.argv[1])
columns = {row[1] for row in connection.execute("PRAGMA table_info(NVTX_EVENTS)")}
if "textId" in columns:
    query = (
        "SELECT COALESCE(e.text, s.value) FROM NVTX_EVENTS e "
        "LEFT JOIN StringIds s ON e.textId = s.id"
    )
else:
    query = "SELECT text FROM NVTX_EVENTS"
counts = Counter()
for (text,) in connection.execute(query):
    if text:
        counts[text.split(" ", 1)[0]] += 1
for label, count in sorted(counts.items()):
    print(f"{label} {count}")
print(f"labels {len(counts)} events {sum(counts.values())}")
