"""Summarize a py-spy --format raw --threads record: per thread, samples, top leaf
frames (self time) and top frames by inclusive presence (each frame counted once per
stack). usage: python pyspy_raw_summary.py FILE.raw [--top 25]"""

import argparse
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("raw")
ap.add_argument("--top", type=int, default=25)
args = ap.parse_args()

per_thread = defaultdict(lambda: {"n": 0, "leaf": Counter(), "incl": Counter()})
total = 0
with open(args.raw) as raw:
    for line in raw:
        line = line.rstrip("\n")
        if not line:
            continue
        stack, _, count = line.rpartition(" ")
        count = int(count)
        frames = stack.split(";")
        thread = frames[0]
        frames = frames[1:] if len(frames) > 1 else frames
        t = per_thread[thread]
        t["n"] += count
        total += count
        t["leaf"][frames[-1]] += count
        for f in set(frames):
            t["incl"][f] += count

print(f"{total} samples")
for thread, t in sorted(per_thread.items(), key=lambda kv: -kv[1]["n"]):
    print(f"\n== {thread}: {t['n']} samples ({100*t['n']/total:.1f} %)")
    print("  leaf (self):")
    for f, c in t["leaf"].most_common(args.top):
        print(f"   {100*c/t['n']:5.1f} %  {f[:110]}")
    print("  inclusive:")
    for f, c in t["incl"].most_common(args.top):
        print(f"   {100*c/t['n']:5.1f} %  {f[:110]}")
