"""WER and SIM per arm over every c64 clean pass that has them.

usage: python3 quality_table.py <root> <arm>=<boot>,<boot> [<arm>=<boot>,<boot> ...]
"""

import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("| arm | passes | corpus WER % | below-50 WER % (sd) | above 50 % | SIM (boot means) |")
print("|---|---|---|---|---|---|")
for spec in sys.argv[2:]:
    arm, boots = spec.split("=")
    corpus, below, above, sims = [], [], 0, []
    for boot in boots.split(","):
        boot_sims = []
        for rep in ("rep0", "rep1"):
            cell = root / boot / "c64_clean/bench" / rep
            if (cell / "wer_results.json").exists():
                summary = json.loads((cell / "wer_results.json").read_text())["summary"]
                corpus.append(summary["wer_corpus"] * 100)
                below.append(summary["wer_below_50_corpus"] * 100)
                above += summary["n_above_50_pct_wer"]
            if (cell / "similarity_results.json").exists():
                boot_sims.append(
                    json.loads((cell / "similarity_results.json").read_text())["summary"]["speaker_similarity_mean"]
                )
        if boot_sims:
            sims.append(statistics.mean(boot_sims))
    if corpus:
        print(f"| {arm} | {len(corpus)} | {statistics.mean(corpus):.3f} | {statistics.mean(below):.3f} "
              f"({statistics.pstdev(below):.3f}) | {above} | {' / '.join(f'{x:.3f}' for x in sims) or '-'} |")
