"""Where two clean passes of one server differ: per prompt, by request phase, and over the pass's time.

usage: python3 pass_shift.py <root> <load> <boot>[/repA,repB] [<boot> ...]

For each boot, rep1 against rep0 by sample id: the mean latency shift and how much of it the slowest-moving 1 % and
5 % of prompts carry, the shift split into first audio and the rest (first audio to the last chunk), its correlation
with the prompt's own reply length change, and the shift by tenth of the pass (requests ordered by start time), so a
slow phase shows as one tenth carrying it. Then the pass level: realized audio seconds, run-on replies (over three
times the corpus median audio length) and the pass's request rate.
"""

import json
import statistics
import sys
from pathlib import Path

root, load = Path(sys.argv[1]), sys.argv[2]


def load_pass(boot, rep):
    """Successful requests by sample id, in dispatch order (the benchmark lists them in sample order, and with a
    full in-flight window each starts when a slot frees, so list order is the pass's time order)."""
    data = json.loads((root / boot / f"{load}_clean/bench/{rep}/speed_results.json").read_text())
    return {r["id"]: r for r in data["per_request"] if r["is_success"] and r["audio_ttfp_s"] is not None}


for spec in sys.argv[3:]:
    boot, _, reps = spec.partition("/")
    rep_a, rep_b = (reps.split(",") if reps else ("rep0", "rep1"))
    a, b = load_pass(boot, rep_a), load_pass(boot, rep_b)
    ids = sorted(set(a) & set(b))
    lat = {i: (b[i]["latency_s"] - a[i]["latency_s"]) * 1e3 for i in ids}
    first = {i: (b[i]["audio_ttfp_s"] - a[i]["audio_ttfp_s"]) * 1e3 for i in ids}
    audio = {i: b[i]["audio_duration_s"] - a[i]["audio_duration_s"] for i in ids}
    mean = statistics.mean(lat.values())
    ranked = sorted(ids, key=lambda i: -abs(lat[i]))
    top1 = sum(lat[i] for i in ranked[: max(1, len(ids) // 100)]) / len(ids)
    top5 = sum(lat[i] for i in ranked[: max(1, len(ids) // 20)]) / len(ids)
    print(f"\n{boot} {load} {rep_b} - {rep_a}: {len(ids)} prompts, latency shift mean {mean:+.1f} ms, median "
          f"{statistics.median(lat.values()):+.1f}; carried by the 1 % largest moves {top1:+.1f}, the 5 % {top5:+.1f}")
    tokens = {i: (b[i]["completion_tokens"] or 0) - (a[i]["completion_tokens"] or 0) for i in ids}
    print(f"  first audio shift mean {statistics.mean(first.values()):+.1f} ms, after first audio "
          f"{mean - statistics.mean(first.values()):+.1f} ms; corr(latency shift, own audio change) "
          f"{statistics.correlation(list(lat.values()), list(audio.values())):+.2f}, own text token change "
          f"{statistics.correlation(list(lat.values()), list(tokens.values())):+.2f}")
    for name, run in ((rep_a, a), (rep_b, b)):
        order = list(run.values())
        tenth = max(1, len(order) // 10)
        means = [statistics.mean(r["latency_s"] for r in order[k : k + tenth]) * 1e3 for k in range(0, tenth * 10, tenth)]
        firsts = [statistics.mean(r["audio_ttfp_s"] for r in order[k : k + tenth]) * 1e3 for k in range(0, tenth * 10, tenth)]
        total_audio = sum(r["audio_duration_s"] for r in run.values())
        median_audio = statistics.median(r["audio_duration_s"] for r in run.values())
        run_ons = sum(1 for r in run.values() if r["audio_duration_s"] > 3 * median_audio)
        longest = max(r["audio_duration_s"] for r in run.values())
        print(f"  {name}: audio {total_audio:.0f} s, run-ons {run_ons}, longest reply {longest:.1f} s; latency by tenth "
              + " ".join(f"{m:.0f}" for m in means) + "; first audio by tenth " + " ".join(f"{m:.0f}" for m in firsts))
