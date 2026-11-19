"""Per boot of the one-card CI cell: text Video-MME c1 and c16 (every clean repeat and the events pass), the talker test
passes and the preprocessing process's OMP setting, CPUs and threads.

usage: python3 oc_table.py <run dir> <boot> [...]
"""

import json
import statistics
import sys
from pathlib import Path


def video_passes(boot: Path, load: int) -> list[tuple[str, dict]]:
    paths = sorted((boot / f"c{load}_clean" / "bench").glob("rep*/videomme_results.json"))
    paths.append(boot / f"c{load}_events" / "bench" / "videomme_results.json")
    return [(path.parent.name, json.loads(path.read_text())) for path in paths if path.exists()]


def main() -> None:
    run = Path(sys.argv[1])
    for name in sys.argv[2:]:
        boot = run / name
        print(f"== {name}")
        for load in (1, 16):
            for label, result in video_passes(boot, load):
                speed, summary = result["speed"], result["summary"]
                print(
                    f"  c{load} {label:<6} qps {speed['throughput_qps']:.3f} latency mean {speed['latency_mean_s']:.2f} s "
                    f"p95 {speed['latency_p95_s']:.2f} s accuracy {summary['accuracy']:.2f} failed {summary['failed']}"
                )
        talker = boot / "talker" / "summary.jsonl"
        if talker.exists():
            rows = [json.loads(line) for line in talker.read_text().splitlines() if line.strip()]
            for row in rows:
                speed = row["speed"]
                print(
                    f"  talker pass {row['pass']} accuracy {row['accuracy']:.2f} ({row['correct']}/{row['total']}, "
                    f"floor {row['accuracy_floor']}) failed {row['failed']} qps {speed['throughput_qps']} "
                    f"latency mean {speed['latency_mean_s']} rtf {speed['rtf_mean']} tok/req/s {speed['output_tok_per_req_s']}"
                )
            print(f"  talker accuracy mean {statistics.mean(row['accuracy'] for row in rows):.3f}")
        else:
            print("  talker: no summary")
        cpus = boot / "stage_cpus.txt"
        for line in cpus.read_text().splitlines() if cpus.exists() else []:
            print(f"  {line[:150]}")


if __name__ == "__main__":
    main()
