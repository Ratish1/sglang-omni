"""Stream a torch chrome trace and keep the events inside given time windows.

Usage: python trace_windows.py <trace.json.gz> <out.jsonl> t0:t1 [t0:t1 ...]
Times are the trace's own ts in microseconds. Events are written one per
line as JSON with ph, cat, name, pid, tid, ts, dur and args.
"""

import gzip
import json
import sys


def iter_events(path):
    buf = []
    depth = 0
    started = False
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not started:
                if '"traceEvents"' in line:
                    started = True
                continue
            stripped = line.strip()
            if depth == 0:
                if stripped.startswith("{"):
                    buf = [line]
                    depth = line.count("{") - line.count("}")
                    if depth == 0:
                        text = "".join(buf).rstrip().rstrip(",")
                        try:
                            yield json.loads(text)
                        except json.JSONDecodeError:
                            pass
                continue
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                depth = 0
                text = "".join(buf).rstrip().rstrip(",")
                try:
                    yield json.loads(text)
                except json.JSONDecodeError:
                    pass


def main():
    path, out = sys.argv[1], sys.argv[2]
    windows = []
    for spec in sys.argv[3:]:
        a, b = spec.split(":")
        windows.append((float(a), float(b)))
    kept = 0
    total = 0
    with open(out, "w") as sink:
        for ev in iter_events(path):
            total += 1
            ts = ev.get("ts")
            if ts is None:
                continue
            for a, b in windows:
                if a <= ts <= b:
                    sink.write(json.dumps(ev, separators=(",", ":")) + "\n")
                    kept += 1
                    break
    print(f"events total={total} kept={kept}", file=sys.stderr)


if __name__ == "__main__":
    main()
