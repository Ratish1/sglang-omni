"""Rank the CUDA lines and SASS instructions of one kernel by Nsight Compute warp stall samples.

Input is `ncu -i <rep> --page source --csv --print-source cuda,sass`. CUDA line rows carry an
Address of "-"; the SASS rows under them carry their address.

Usage: python3 ncu_source_digest.py <source.csv> [--top N]
"""

import argparse
import csv

SAMPLES = "Warp Stall Sampling (All Samples)"
NOT_ISSUED = "Warp Stall Sampling (Not-issued Samples)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_csv")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    with open(args.source_csv, newline="") as handle:
        rows = list(csv.reader(handle))
    header_index = next(index for index, row in enumerate(rows) if row[:2] == ["Line No", "Source"])
    header = rows[header_index]
    samples_column = header.index(SAMPLES)
    not_issued_column = header.index(NOT_ISSUED)
    lines, instructions = [], []
    current_line = None
    for row in rows[header_index + 1 :]:
        if len(row) <= samples_column or not row[samples_column].replace(".", "").isdigit():
            continue
        samples = float(row[samples_column])
        not_issued = float(row[not_issued_column])
        if row[2] == "-":
            current_line = (row[0], row[1].strip())
            lines.append((samples, not_issued, row[0], row[1].strip()))
        else:
            instructions.append((samples, not_issued, row[2], row[3].strip(), current_line))
    total = sum(sample for sample, *_ in lines) or 1.0
    print(f"total samples {total:.0f}")
    print("CUDA lines by samples (share, not-issued share, line, source):")
    for samples, not_issued, line, source in sorted(lines, reverse=True)[: args.top]:
        print(f"  {100 * samples / total:5.1f}% {100 * not_issued / total:5.1f}%  {line:>5}  {source[:110]}")
    print("SASS instructions by samples (share, address, instruction, CUDA line):")
    for samples, not_issued, address, instruction, owner in sorted(instructions, reverse=True)[: args.top]:
        line = owner[0] if owner else "?"
        print(f"  {100 * samples / total:5.1f}%  {address}  {instruction[:70]:70s}  line {line}")


if __name__ == "__main__":
    main()
