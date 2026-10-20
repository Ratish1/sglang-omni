"""Where a pair's playback underruns land, per arm and card order, from its stored speed_results.json: the second chunk
when it is not the last (a first hop's follow up), the last chunk (the final step), or a chunk between. Playback starts
at the first chunk's arrival and a late chunk stalls it, as the benchmark's max_playback_underrun_s counts it. The
first tree of the pair is the slot 0 arm in order a and the slot 1 arm in order b (serve_pair_c20261017.py).
Reads only copied artifacts, so no box run.

usage: python3 underrun_sites_c20261017.py <pair dir> <cell> [<first name> <second name>]"""

import json
import sys
from collections import Counter
from pathlib import Path

pair, cell = Path(sys.argv[1]), sys.argv[2]
names = sys.argv[3:5] if len(sys.argv) > 4 else ["first", "second"]
slots = {names[0]: [("a", "gpu0"), ("b", "gpu1")], names[1]: [("a", "gpu1"), ("b", "gpu0")]}
for name, arm_slots in slots.items():
    for order, gpu in arm_slots:
        rows = json.load(open(pair / order / gpu / cell / "speed_results.json"))["per_request"]
        counts: Counter[str] = Counter()
        seconds: Counter[str] = Counter()
        request_count = 0
        for row in rows:
            if not row["is_success"]:
                continue
            request_count += 1
            gaps, durations = row["inter_chunk_s"] or [], row["chunk_audio_duration_s"]
            arrival, played_until = 0.0, durations[0]
            for k, gap in enumerate(gaps, start=1):
                arrival += gap
                underrun = max(0.0, arrival - played_until)
                if underrun > 0.0:
                    site = "last" if k == len(gaps) else ("second" if k == 1 else "between")
                    counts[site] += 1
                    seconds[site] += underrun
                played_until = max(played_until, arrival) + durations[k]
        sites = ", ".join(f"{site} {counts[site]} ({seconds[site]:.2f} s)" for site in ("second", "between", "last"))
        print(f"{cell} {name:12} {order}: {request_count} requests; underruns at {sites}")
