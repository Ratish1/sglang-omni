"""Busy CPUs per socket on the whole host, one line every interval, so a boot can be read against its neighbours' load.

usage: python3 cpu_load_sampler.py /data/run/cpu_load.log [--interval 20]
"""

import argparse
import time


def read_counters() -> dict[int, tuple[int, int]]:
    counters = {}
    with open("/proc/stat") as source:
        for line in source:
            if line.startswith("cpu") and line[3].isdigit():
                fields = line.split()
                counters[int(fields[0][3:])] = (
                    sum(map(int, fields[1:])),
                    int(fields[4]) + int(fields[5]),
                )
            else:
                pass
    return counters


def socket_cpus() -> list[list[int]]:
    sockets = []
    for node in range(16):
        try:
            with open(f"/sys/devices/system/node/node{node}/cpulist") as source:
                text = source.read().strip()
        except OSError:
            break
        cpus = []
        for part in text.split(","):
            first, _dash, last = part.partition("-")
            cpus.extend(range(int(first), int(last or first) + 1))
        sockets.append(cpus)
    return sockets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--interval", type=float, default=20.0)
    args = parser.parse_args()
    sockets = socket_cpus()
    before = read_counters()
    while True:
        time.sleep(args.interval)
        after = read_counters()
        busy = []
        for cpus in sockets:
            total = 0.0
            for cpu in cpus:
                elapsed = max(after[cpu][0] - before[cpu][0], 1)
                total += 1 - (after[cpu][1] - before[cpu][1]) / elapsed
            busy.append(f"{total:.0f}/{len(cpus)}")
        before = after
        with open(args.out, "a") as sink:
            sink.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {' '.join(busy)}\n")


if __name__ == "__main__":
    main()
