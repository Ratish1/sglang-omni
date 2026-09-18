"""One table row per run directory of g1_stream_identity.sh.

  python speed_summary.py RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FIELDS = (
    "completed_requests",
    "failed_requests",
    "throughput_qps",
    "audio_throughput_s_per_s",
    "rtf_mean",
    "rtf_p99",
    "audio_ttfp_mean_s",
    "audio_ttfp_p95_s",
    "inter_chunk_mean_s",
    "inter_chunk_p99_s",
    "latency_mean_s",
    "latency_p95_s",
    "c50",
    "c100",
    "c200",
)


def main() -> None:
    print("run | head | " + " | ".join(FIELDS) + " | cublas | retract")
    for run in map(Path, sys.argv[1:]):
        results = json.loads(
            (run / "seedtts" / "measured" / "speed_results.json").read_text()
        )
        summary = results.get("summary", results)
        serve_log = (run / "serve.log").read_text(errors="replace")
        print(
            " | ".join(
                [
                    run.name,
                    (run / "head.txt").read_text().strip()[:9],
                    *(str(summary.get(field)) for field in FIELDS),
                    str(serve_log.count("CUBLAS")),
                    str(serve_log.lower().count("retract")),
                ]
            )
        )


if __name__ == "__main__":
    main()
