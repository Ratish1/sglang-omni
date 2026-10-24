"""The decode kernel tables of a profile's BBuf pair reads, side by side: per cell (PROFILE/formal/<cell>/bbuf_pair.md),
each kernel of the first decode table with its share of decode GPU time, then one row per kernel name across cells.
Kernel names are cut to KERNEL_CHARS; the location is the first Python site the table gives.

usage: python3 kernel_shares.py PROFILE CELL [CELL ...]
"""

import re
import sys
from pathlib import Path

KERNEL_CHARS = 60
TOP = 25


def decode_rows(path: Path) -> list[tuple[str, float, float, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows = []
    in_table = False
    for line in lines:
        if line.startswith("##### decode") and not rows:
            in_table = True
        elif in_table and line.startswith("| ") and not line.startswith("| Kernel") and not line.startswith("| ---"):
            cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
            if len(cells) < 6:
                continue
            else:
                pass
            time_ms = float(re.sub(r"[^0-9.]", "", cells[2]) or 0)
            share = float(re.sub(r"[^0-9.]", "", cells[3]) or 0)
            site = cells[5].split("<br>")[0]
            rows.append((cells[0][:KERNEL_CHARS], share, time_ms, site))
        elif in_table and rows and not line.startswith("|"):
            break
        else:
            pass
    return rows


def main() -> None:
    profile = Path(sys.argv[1])
    cells = sys.argv[2:]
    by_cell = {cell: decode_rows(profile / "formal" / cell / "bbuf_pair.md") for cell in cells}
    for cell, rows in by_cell.items():
        print(f"## {cell}: {len(rows)} kernels, {sum(row[2] for row in rows):.1f} ms listed")
        for name, share, time_ms, site in rows[:TOP]:
            print(f"  {share:5.1f} % {time_ms:9.2f} ms  {name}  @ {site}")
    names = []
    for rows in by_cell.values():
        for name, *_ in rows:
            if name not in names:
                names.append(name)
            else:
                pass
    print("\n| kernel | " + " | ".join(cells) + " |")
    print("| --- | " + " | ".join("---" for _ in cells) + " |")
    for name in names:
        shares = []
        for cell in cells:
            match = [row[1] for row in by_cell[cell] if row[0] == name]
            shares.append(f"{sum(match):.1f} %" if match else "")
        print(f"| {name} | " + " | ".join(shares) + " |")


if __name__ == "__main__":
    main()
