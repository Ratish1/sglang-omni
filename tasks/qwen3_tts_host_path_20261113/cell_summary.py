"""Speed, first audio, WER, SIM and capped runaways of every benchmark cell under a root, one line each.

usage: python3 cell_summary.py /data/.../retract
"""

import json
import sys
from pathlib import Path


def read_summary(path: Path) -> dict:
    return json.loads(path.read_text())["summary"] if path.exists() else {}


def main() -> None:
    root = Path(sys.argv[1])
    print(
        f"{'cell':56s}{'req/s':>8s}{'rtf':>8s}{'ttfp ms':>9s}{'capped':>8s}{'wer %':>8s}"
        f"{'wer<50 %':>9s}{'sim':>8s}"
    )
    for speed_path in sorted(root.rglob("bench/speed_results.json")):
        bench = speed_path.parent
        speed = json.loads(speed_path.read_text())
        records = speed["per_request"]
        ttfps = [
            record["audio_ttfp_s"]
            for record in records
            if record.get("audio_ttfp_s") is not None
        ]
        ttfp = 1000 * sum(ttfps) / len(ttfps) if ttfps else float("nan")
        wer = read_summary(bench / "wer_results.json")
        sim = read_summary(bench / "similarity_results.json")
        print(
            f"{str(bench.parent.relative_to(root)):56s}{speed['summary']['throughput_qps']:8.2f}"
            f"{speed['summary']['rtf_mean']:8.4f}{ttfp:9.2f}{speed['summary']['max_token_hits']:8d}"
            f"{100 * wer.get('wer_corpus', float('nan')):8.3f}"
            f"{100 * wer.get('wer_below_50_corpus', float('nan')):9.3f}"
            f"{sim.get('speaker_similarity_mean', float('nan')):8.3f}"
        )


if __name__ == "__main__":
    main()
