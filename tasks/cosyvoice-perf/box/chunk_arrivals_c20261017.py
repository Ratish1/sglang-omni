"""Each chunk's arrival from its request's start, paired by request id between a pair's two arms in the same card
order: the median and p95 of the second arm's arrival less the first's, per chunk index, and for the last chunk. Shows
whether an inter chunk rise is a later chunk or an earlier first chunk. The first tree of the pair is the slot 0 arm
in order a and the slot 1 arm in order b (serve_pair_c20261017.py). Reads only copied artifacts, so no box run.

usage: python3 chunk_arrivals_c20261017.py <pair dir> <cell>"""

import json
import sys
from collections import defaultdict
from pathlib import Path

pair, cell = Path(sys.argv[1]), sys.argv[2]


def arrivals(path: Path) -> dict[str, list[float]]:
    by_id = {}
    for row in json.load(open(path))["per_request"]:
        if row["is_success"]:
            times = [row["audio_ttfp_s"]]
            for gap in row["inter_chunk_s"] or []:
                times.append(times[-1] + gap)
            by_id[row["id"]] = times
    return by_id


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


for order, (first_gpu, second_gpu) in (("a", ("gpu0", "gpu1")), ("b", ("gpu1", "gpu0"))):
    first = arrivals(pair / order / first_gpu / cell / "speed_results.json")
    second = arrivals(pair / order / second_gpu / cell / "speed_results.json")
    deltas: dict[str, list[float]] = defaultdict(list)
    for request_id, first_times in first.items():
        second_times = second.get(request_id)
        if second_times is None or len(second_times) != len(first_times):
            continue
        for k, (a, b) in enumerate(zip(first_times, second_times)):
            deltas[f"chunk {k + 1}" if k < 3 else "chunk 4+"].append(b - a)
        deltas["last"].append(second_times[-1] - first_times[-1])
    rows = ", ".join(
        f"{label} {quantile(values, 0.5) * 1000:+.0f} / {quantile(values, 0.95) * 1000:+.0f} ms (n {len(values)})"
        for label, values in sorted(deltas.items())
    )
    print(f"{cell} order {order}, second arm less first, median / p95: {rows}")
