"""Block 6 reader: per server, the process memory timeline from probe_site against its 500s.

For each server label, prints every process that held device memory (its first log line names
it), its reserved bytes at start, peak and end, the device's lowest free bytes, and the card's
nvidia-smi used memory; then the free bytes and reserved bytes of each process in the 20 s before
the server's first HTTP 500, and the 500s per minute.

python3 memory_timeline.py <block 6 directory>
"""

import argparse
import collections
import csv
import datetime
import re
from pathlib import Path

LABELS = (("cudnn_on0", 0), ("cudnn_on1", 1), ("cudnn_off2", 2), ("cudnn_off3", 3))
GIB = 1024**3


def read_process_log(path: Path) -> tuple[str, list[dict[str, float]]]:
    name = ""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            name = line[2:]
            continue
        else:
            pass
        fields = line.split()
        values = dict(zip(fields[2::2], (float(value) for value in fields[3::2])))
        values["time"] = float(fields[0])
        values["device"] = float(fields[1])
        rows.append(values)
    return name, rows


def first_error_time(log_path: Path) -> float | None:
    pattern = re.compile(
        r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ \[ERROR\] sglang_omni.serve"
    )
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            return datetime.datetime.strptime(
                match.group(1), "%Y-%m-%d %H:%M:%S"
            ).timestamp()
        else:
            pass
    return None


def errors_per_minute(log_path: Path) -> dict[str, int]:
    counts: dict[str, int] = collections.Counter()
    pattern = re.compile(r"^\d{4}-\d\d-\d\d (\d\d:\d\d):\d\d,\d+ \[ERROR\] sglang_omni.serve")
    for line in log_path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            counts[match.group(1)] += 1
        else:
            pass
    return dict(sorted(counts.items()))


def card_used(directory: Path) -> dict[int, list[tuple[float, float]]]:
    used: dict[int, list[tuple[float, float]]] = collections.defaultdict(list)
    with open(directory / "gpu_memory.csv") as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            else:
                pass
            moment = datetime.datetime.strptime(
                row[0].strip(), "%Y/%m/%d %H:%M:%S.%f"
            ).timestamp()
            used[int(row[1])].append((moment, float(row[2].split()[0])))
    return used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    directory = arguments.directory
    used = card_used(directory)
    for label, card in LABELS:
        print(f"== {label} card {card}")
        error_time = first_error_time(directory / f"serve_{label}.log")
        print(
            "first 500",
            None
            if error_time is None
            else datetime.datetime.fromtimestamp(error_time).strftime("%H:%M:%S"),
        )
        print("500s per minute", errors_per_minute(directory / f"serve_{label}.log"))
        card_rows = used.get(card, [])
        if card_rows:
            print(
                f"card used MiB: start {card_rows[0][1]:.0f} max {max(value for _, value in card_rows):.0f} "
                f"end {card_rows[-1][1]:.0f}"
            )
        else:
            pass
        for path in sorted((directory / f"memory_{label}").glob("torchmem_*.log")):
            name, rows = read_process_log(path)
            if not rows:
                continue
            else:
                pass
            print(f"  {path.name} {name[:120]}")
            print(
                f"    reserved GiB: start {rows[0]['reserved'] / GIB:.2f} max {max(row['reserved'] for row in rows) / GIB:.2f} "
                f"end {rows[-1]['reserved'] / GIB:.2f}; device free GiB min {min(row['free'] for row in rows) / GIB:.2f}; "
                f"retries {rows[-1]['retries']:.0f} ooms {rows[-1]['ooms']:.0f}"
            )
            if error_time is not None:
                before = [
                    row for row in rows if error_time - 20 <= row["time"] <= error_time + 2
                ]
                print(
                    "    around the first 500 (s from it, reserved GiB, free GiB):",
                    [
                        (
                            round(row["time"] - error_time),
                            round(row["reserved"] / GIB, 2),
                            round(row["free"] / GIB, 2),
                        )
                        for row in before
                    ],
                )
            else:
                pass


if __name__ == "__main__":
    main()
