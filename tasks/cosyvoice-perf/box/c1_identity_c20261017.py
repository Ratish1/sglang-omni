"""Seeded audio identity between the two arms of a pair, per order: how many samples of a seeded cell have byte
identical wav files on both servers (the cross server rate on main against main is about 58 of 64 at c1).
usage: python3 c1_identity_c20261017.py <pair dir> [cell, default c1]"""

import json
import sys
from pathlib import Path


def audio(cell):
    return {row["sample_id"]: Path(row["wav_path"]).read_bytes() for row in json.loads((cell / "generated.json").read_text())
            if row["is_success"]}


pair = Path(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else "c1"
for order in ("a", "b"):
    first, second = audio(pair / order / "gpu0" / name), audio(pair / order / "gpu1" / name)
    shared = first.keys() & second.keys()
    print(f"{name} order {order}: identical {sum(first[key] == second[key] for key in shared)} of {len(shared)}")
