"""WER (corpus and excluding samples above 50 %) and mean speaker similarity of scored cells.
usage: python3 quality.py <cell dir>..."""

import json
import sys
from pathlib import Path

for cell in map(Path, sys.argv[1:]):
    wer = json.loads((cell / "wer_results.json").read_text())
    sim = json.loads((cell / "similarity_results.json").read_text())
    summary = wer.get("summary", wer)
    sim_summary = sim.get("summary", sim)
    wer_keys = {k: v for k, v in summary.items() if "wer" in k.lower() and not isinstance(v, (list, dict))}
    sim_keys = {k: v for k, v in sim_summary.items() if "sim" in k.lower() and not isinstance(v, (list, dict))}
    print(cell, wer_keys, sim_keys)
