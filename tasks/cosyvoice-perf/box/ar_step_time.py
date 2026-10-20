"""AR decode step time per arm inside one benchmark cell's window, read from the server log.

SGLang prints a Decode batch line every decode_log_interval steps with the running request count
and the generated token rate since the last line, so running / rate is the mean decode step time
over that interval. Binned by running count, the step time tells a slower AR forward (same batch,
longer step) apart from a busier AR stage (more running requests, same step).
usage: python3 ar_step_time.py <arm dir> <cell> [<arm dir> <cell> ...]
"""

import re
import statistics
import sys
from datetime import datetime

DECODE = re.compile(
    r"^(\S+ \S+) .*Decode batch, #running-req: (\d+), .*gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
)
PREFILL = re.compile(
    r"^(\S+ \S+) .*Prefill batch, #new-seq: (\d+), #new-token: (\d+), .*#running-req: (\d+), #queue-req: (\d+)"
)
CELL_START = re.compile(r"^(\S+ \S+) __main__ INFO Prepared")
CELL_DONE = re.compile(r"^cell (\S+) rc \d+ (\S+)")
BINS = ((1, 8), (9, 16), (17, 24), (25, 32), (33, 48), (49, 64))


def stamp(text):
    return datetime.strptime(text.split(",")[0], "%Y-%m-%d %H:%M:%S")


def window(arm, cell):
    with open(f"{arm}/{cell}.log") as f:
        start = next(stamp(m.group(1)) for m in map(CELL_START.match, f) if m)
    with open(f"{arm}/progress.txt") as f:
        done = next(m.group(2) for m in map(CELL_DONE.match, f) if m and m.group(1) == cell)
    end = datetime.combine(start.date(), datetime.strptime(done, "%H:%M:%S").time())
    return start, end


def main():
    args = sys.argv[1:]
    for arm, cell in zip(args[0::2], args[1::2]):
        start, end = window(arm, cell)
        decode, prefill = [], []
        with open(f"{arm}/serve.log") as f:
            for line in f:
                m = DECODE.match(line)
                if m and start <= stamp(m.group(1)) <= end and float(m.group(3)) > 0:
                    decode.append((int(m.group(2)), float(m.group(3)), int(m.group(4))))
                    continue
                m = PREFILL.match(line)
                if m and start <= stamp(m.group(1)) <= end:
                    prefill.append((int(m.group(2)), int(m.group(5))))
        running = [r for r, _, _ in decode]
        print(f"== {arm} {cell} {start:%H:%M:%S}-{end:%H:%M:%S}")
        print(
            f"decode lines={len(decode)} running mean={statistics.mean(running):.1f} "
            f"median={statistics.median(running):.0f} tokens/s mean={statistics.mean(t for _, t, _ in decode):.0f} "
            f"queue-req mean={statistics.mean(q for _, _, q in decode):.2f}"
        )
        for low, high in BINS:
            steps = [r / t * 1000 for r, t, _ in decode if low <= r <= high]
            if steps:
                print(
                    f"  running {low:>2}-{high:<2} lines={len(steps):>3} step ms median={statistics.median(steps):6.2f} "
                    f"mean={statistics.mean(steps):6.2f}"
                )
        if prefill:
            print(
                f"prefill lines={len(prefill)} new-seq mean={statistics.mean(s for s, _ in prefill):.2f} "
                f"queue-req mean={statistics.mean(q for _, q in prefill):.2f} max={max(q for _, q in prefill)}"
            )


if __name__ == "__main__":
    main()
