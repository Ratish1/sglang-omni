"""Talker reply length per request over every SeedTTS pass of the given boots: audio seconds per target word, the
replies over three times the corpus median, the ones that ran to the frame cap, and which samples recur across passes.

usage: python3 long_replies.py <root> <boot> [<boot> ...]
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CAP_S = 327.0
root = Path(sys.argv[1])
rows = []
for boot in sys.argv[2:]:
    for path in sorted((root / boot).glob("c*_*/bench/**/generated.json")):
        cell = path.relative_to(root / boot).parts[0]
        for entry in json.loads(path.read_text()):
            words = max(len(entry["target_text"].split()), 1)
            duration = entry.get("audio_duration_s") or 0.0
            rows.append((boot, cell, str(path.parent.name), entry["sample_id"], words, duration, entry["is_success"]))
per_word = [duration / words for _, _, _, _, words, duration, ok in rows if ok and duration > 0]
median = statistics.median(per_word)
print(f"{len(rows)} replies, median {median:.3f} s per word, over-long above {3 * median:.3f}")
recur = defaultdict(list)
for boot in sys.argv[2:]:
    mine = [row for row in rows if row[0] == boot]
    passes = {(row[1], row[2]) for row in mine}
    long = [row for row in mine if row[6] and row[5] / row[4] > 3 * median]
    capped = [row for row in mine if row[5] >= CAP_S]
    failed = sum(1 for row in mine if not row[6])
    print(f"{boot}: {len(passes)} passes, {len(mine)} replies, over-long {len(long)}, capped {len(capped)}, failed {failed}")
    for row in capped:
        print(f"  capped {row[1]} {row[2]} {row[3]} words {row[4]} audio {row[5]:.1f} s")
    for row in long:
        recur[row[3]].append(f"{boot}/{row[1]}/{row[2]}:{row[5]:.1f}s")
print("samples over-long in more than one pass:")
for sample, hits in sorted(recur.items(), key=lambda item: -len(item[1])):
    if len(hits) > 1:
        print(f"  {sample} {len(hits)}: {' '.join(hits[:8])}")
counts = Counter(len(hits) for hits in recur.values())
print(f"over-long samples by number of passes: {dict(sorted(counts.items()))}")
