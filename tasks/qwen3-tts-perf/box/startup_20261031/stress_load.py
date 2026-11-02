"""The serving stress harness as an nsys boot's load: one single-workload stage against a running server, between the two
bench.log lines the census cuts its window from.

usage: python3 stress_load.py --harness TREE --spec SPEC --port 26000 --concurrency 64 --requests 128 --out DIR
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path


def stamp() -> str:
    now = datetime.datetime.now()
    return f"{now:%Y-%m-%d %H:%M:%S},{now.microsecond // 1000:03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    spec["base_url"] = f"http://127.0.0.1:{args.port}"
    spec["params"]["file_ref_audio"] = f"file://{Path(args.harness) / 'docs/_static/audio/male-voice.wav'}"
    spec["params"]["load_stages"][0]["max_concurrency"] = args.concurrency
    spec["params"]["load_stages"][0]["request_count"] = args.requests
    spec_path = args.out / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    print(f"{stamp()} Benchmarking {args.requests} requests", flush=True)
    code = subprocess.run(
        [sys.executable, "-m", "benchmarks.eval.benchmark_tts_serving", "--spec", str(spec_path), "--out", str(args.out / "out")],
        cwd=args.harness, env=dict(os.environ, PYTHONPATH=args.harness), check=False,
    ).returncode
    print(f"{stamp()} Results saved to {args.out} harness exit {code}", flush=True)


if __name__ == "__main__":
    main()
