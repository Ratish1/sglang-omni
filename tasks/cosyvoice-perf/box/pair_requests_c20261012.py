"""Paired per request differences (B minus A, same request id) of one cell of a same time pair: first audio, the
request's mean inter chunk gap and latency, as median, mean, p10 and p90 in ms, and how many requests are later.
usage: python3 pair_requests_c20261012.py <label> <arm dir A> <arm dir B> <cell>"""

import json
import statistics
import sys
from pathlib import Path


def per_request(arm: Path, cell: str) -> dict[str, dict]:
    rows = json.loads((arm / cell / "speed_results.json").read_text())["per_request"]
    return {row["id"]: row for row in rows if row["is_success"]}


def mean_gap(row: dict) -> float | None:
    gaps = row.get("inter_chunk_s") or []
    return sum(gaps) / len(gaps) if gaps else None


label, arm_a, arm_b, cell = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
rows_a, rows_b = per_request(arm_a, cell), per_request(arm_b, cell)
shared = sorted(set(rows_a) & set(rows_b))
for metric, value in (
    ("first audio", lambda row: row["audio_ttfp_s"]),
    ("mean inter chunk", mean_gap),
    ("latency", lambda row: row["latency_s"]),
):
    diffs = []
    for request_id in shared:
        a, b = value(rows_a[request_id]), value(rows_b[request_id])
        if a is not None and b is not None:
            diffs.append(1000 * (b - a))
    diffs.sort()
    p10, p90 = diffs[len(diffs) // 10], diffs[9 * len(diffs) // 10]
    later = sum(1 for diff in diffs if diff > 0)
    print(
        f"{label} {cell} {metric}: median {statistics.median(diffs):+.1f} ms, mean {statistics.mean(diffs):+.1f} ms, "
        f"p10 {p10:+.1f}, p90 {p90:+.1f}, later {later} / {len(diffs)}"
    )
