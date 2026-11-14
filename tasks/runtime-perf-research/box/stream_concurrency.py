"""Speech streams in flight by phase, from a closed-loop pass's per request records, beside its chunk gaps.

usage: python3 stream_concurrency.py <run dir> <cell, e.g. c16> boot [boot ...]

Little's law on the measured throughput: requests before first audio = QPS x mean first audio, requests
streaming audio = QPS x mean (latency - first audio). Chunk gaps are pooled over every request of a pass.
"""

import json
import statistics
import sys
from pathlib import Path


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    run_dir, cell, boots = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    print(
        f"{'boot/pass':28s} {'QPS':>6s} {'before 1st audio':>16s} {'streaming':>10s} {'sum':>5s} "
        f"{'gap mean':>9s} {'gap p50':>8s} {'gap p99':>8s} {'gaps/req':>8s}"
    )
    for boot in boots:
        for directory in sorted((run_dir / boot).glob(f"{cell}_*")):
            if directory.name.endswith("_warm"):
                continue
            else:
                pass
            for path in sorted(directory.glob("bench/**/speed_results.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                done = [r for r in data["per_request"] if r["is_success"]]
                qps = data["summary"]["throughput_qps"]
                first = statistics.fmean(r["audio_ttfp_s"] for r in done)
                stream = statistics.fmean(
                    r["latency_s"] - r["audio_ttfp_s"] for r in done
                )
                gaps = [gap for r in done for gap in (r["inter_chunk_s"] or [])]
                label = f"{boot}/{path.parent.relative_to(directory.parent)}".replace(
                    "/bench", ""
                )
                print(
                    f"{label:28s} {qps:6.2f} {qps * first:16.2f} {qps * stream:10.2f} {qps * (first + stream):5.1f} "
                    f"{statistics.fmean(gaps) * 1000:9.1f} {pct(gaps, 0.5) * 1000:8.1f} {pct(gaps, 0.99) * 1000:8.1f} "
                    f"{len(gaps) / len(done):8.2f}"
                )


if __name__ == "__main__":
    main()
