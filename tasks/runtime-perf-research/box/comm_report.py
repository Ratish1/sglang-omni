"""Read SGLANG_OMNI_COMM_TRACE lines from a baseline.py run's serve.log and print, per cell window, each trace event's
timing fields (every field ending in _ms): p50, p90, p99 and count, grouped by event and edge.

usage: python3 comm_report.py /data/runtime-perf/b5_omni_speech [--cells c16_events]
"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--cells", default="")
    args = parser.parse_args()
    run = Path(args.run)
    records = []
    for raw in (run / "serve.log").open(errors="replace"):
        index = raw.find("COMM_TRACE ")
        if index < 0:
            continue
        else:
            pass
        try:
            records.append(json.loads(raw[index + len("COMM_TRACE "):]))
        except json.JSONDecodeError:
            continue
    wanted = set(args.cells.split(",")) if args.cells else None
    for cell in sorted(path for path in run.iterdir() if (path / "window.json").exists()):
        if wanted and cell.name not in wanted:
            continue
        else:
            pass
        window = json.loads((cell / "window.json").read_text())
        start, end = int(window["start"] * 1e9), int(window["end"] * 1e9)
        groups = collections.defaultdict(lambda: collections.defaultdict(list))
        counts = collections.Counter()
        for record in records:
            if not start <= record.get("ts_ns", 0) <= end:
                continue
            else:
                pass
            edge = f"{record.get('from_stage', record.get('src_device', '-'))}->" \
                   f"{record.get('to_stage', record.get('dst_device', record.get('device', '-')))}"
            key = (record["event"], edge, record.get("kind", record.get("transport", "")))
            counts[key] += 1
            for field, value in record.items():
                if field.endswith("_ms") and isinstance(value, (int, float)):
                    groups[key][field].append(float(value))
                else:
                    pass
        print(f"== {cell.name}")
        for key in sorted(counts, key=lambda k: -counts[k]):
            print(f"  {key[0]} {key[1]} {key[2]}: n={counts[key]}")
            for field, values in sorted(groups[key].items()):
                if max(values) <= 0:
                    continue
                else:
                    pass
                print(f"      {field:24s} p50 {statistics.median(values):8.3f} p90 {pct(values, 0.9):8.3f} "
                      f"p99 {pct(values, 0.99):8.3f} max {max(values):8.3f}")


if __name__ == "__main__":
    main()
