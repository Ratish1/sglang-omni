"""The first PASSES passes of a running cell, as a run directory scoring can read and thin without touching the cell.

WAVs are hard links: queue_score.sh removes a good sample's WAV from the snapshot and the cell's own copy stays.

usage: python3 cells_snapshot.py CELL_RUN OUT_RUN PASSES   (each run directory holds seedtts_en/)
"""

import json
import os
import sys
from pathlib import Path


def main() -> None:
    cell = Path(sys.argv[1]) / "seedtts_en"
    out = Path(sys.argv[2]) / "seedtts_en"
    passes = {f"__r{index}" for index in range(int(sys.argv[3]))}
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in json.loads((cell / "generated.json").read_text()) if row["sample_id"][row["sample_id"].rindex("__r") :] in passes
    ]
    for row in rows:
        link = out / "audio" / Path(row["wav_path"]).name
        os.link(row["wav_path"], link)
        row["wav_path"] = str(link)
    replies = json.loads((cell / "reply_text.json").read_text())
    (out / "generated.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (out / "reply_text.json").write_text(
        json.dumps({row["sample_id"]: replies[row["sample_id"]] for row in rows}, indent=1, ensure_ascii=False)
    )
    print(f"{out}: {len(rows)} rows, {len({row['sample_id'].split('__r')[0] for row in rows})} samples")


if __name__ == "__main__":
    main()
