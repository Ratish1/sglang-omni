"""Resident peak of the preprocessing process per boot and pass, from baseline.py's memory.json: the process whose
threads include the media decode pool. Also its anonymous and shared memory at the pass end, and the peak summed over
every serving process.

usage: python3 memory_table.py <root> <cell, e.g. c16_clean> <boot> [<boot> ...]
"""

import json
import sys
from pathlib import Path

PREPROCESSING_THREADS = ("media-decode", "preprocessing-v")

root, cell = Path(sys.argv[1]), sys.argv[2]
print(
    f"| {cell} | preprocessing peak (MiB) | anon at end | shmem at end | all processes peak |"
)
print("|---|---|---|---|---|")
for boot in sys.argv[3:]:
    memory = json.loads((root / boot / cell / "memory.json").read_text())
    preprocessing = [
        process
        for process in memory.values()
        if any(name.startswith(PREPROCESSING_THREADS) for name in process["threads"])
    ]
    assert len(preprocessing) == 1, [
        process["threads"][:8] for process in preprocessing
    ]
    process = preprocessing[0]
    total = sum(item["peak_mib"] for item in memory.values())
    print(
        f"| {boot} | {process['peak_mib']:.0f} | {process['anon_mib']:.0f} | "
        f"{process['shmem_mib']:.0f} | {total:.0f} |"
    )
