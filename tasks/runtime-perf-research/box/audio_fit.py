"""Rows of every clean pass read against main's fit on the pass's realized total audio.

usage: python3 audio_fit.py <root> <load> main=<boot>,<boot>,... <arm>=<boot>,... [<arm>=<boot>,...]

Unseeded talker sampling gives each pass about 1 % more or less total reply audio, and at c64 the talker queue turns
that into 1 to 3 % of latency for every request. Each row (QPS, latency mean, first audio mean, inter chunk mean,
latency per audio second) is fitted on main's passes as a + b * total audio. Per arm: each pass's residual from the
fit in percent, their mean and the spread; main's own residuals are leave-one-out (each pass against the fit of the
others), so main's row is the band a candidate's mean residual is read against.
"""

import json
import statistics
import sys
from pathlib import Path

root, load = Path(sys.argv[1]), sys.argv[2]
arms = [(spec.split("=")[0], spec.split("=")[1].split(",")) for spec in sys.argv[3:]]
ROWS = ("qps", "latency_mean_ms", "first_audio_mean_ms", "inter_chunk_mean_ms", "latency_per_audio_s_ms")


def pass_rows(boot):
    out = []
    for rep in ("rep0", "rep1"):
        path = root / boot / f"{load}_clean/bench/{rep}/speed_results.json"
        if not path.exists():
            continue
        else:
            pass
        data = json.loads(path.read_text())
        ok = [r for r in data["per_request"] if r["is_success"] and r["audio_ttfp_s"] is not None]
        chunks = [gap for r in ok for gap in (r["inter_chunk_s"] or [])]
        out.append({
            "pass": f"{boot}/{rep}",
            "audio": sum(r["audio_duration_s"] for r in ok),
            "qps": data["summary"]["throughput_qps"],
            "latency_mean_ms": statistics.mean(r["latency_s"] for r in ok) * 1e3,
            "first_audio_mean_ms": statistics.mean(r["audio_ttfp_s"] for r in ok) * 1e3,
            "inter_chunk_mean_ms": statistics.mean(chunks) * 1e3 if chunks else float("nan"),
            "latency_per_audio_s_ms": statistics.mean(r["latency_s"] / r["audio_duration_s"] for r in ok) * 1e3,
        })
    return out


def fit(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return lambda x: my + slope * (x - mx), slope


passes = {name: [row for boot in boots for row in pass_rows(boot)] for name, boots in arms}
main = passes[arms[0][0]]
print(f"{load}: {len(main)} main passes, total audio {min(p['audio'] for p in main):.0f} to "
      f"{max(p['audio'] for p in main):.0f} s")
for row in ROWS:
    predict, slope = fit([(p["audio"], p[row]) for p in main])
    loo = []
    for k, p in enumerate(main):
        others, _ = fit([(q["audio"], q[row]) for j, q in enumerate(main) if j != k])
        loo.append(100 * (p[row] - others(p["audio"])) / others(p["audio"]))
    print(f"\n  {row}: main fit slope {slope:+.4f} per audio s; main leave-one-out residual sd {statistics.stdev(loo):.2f} %, "
          f"mean of 4 passes sd about {statistics.stdev(loo) / 2:.2f} %")
    for name, _ in arms[1:]:
        res = [100 * (p[row] - predict(p["audio"])) / predict(p["audio"]) for p in passes[name]]
        if res:
            print(f"    {name:10s} passes {len(res)} mean residual {statistics.mean(res):+.2f} % "
                  f"(each {' '.join(f'{r:+.2f}' for r in res)}), audio {statistics.mean(p['audio'] for p in passes[name]):.0f} s")
        else:
            pass
