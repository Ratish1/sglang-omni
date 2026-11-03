"""First chunks while long non-streaming requests decode beside them, the case a whole request's bounded pieces exist for.
Two trees at the same time, one server per card, each serving SeedTTS EN to a streaming client at 8 in flight and the
harness's 4,096 character requests, non-streaming, to a thin client at 8 in flight, both card orders. Prints each arm's
streaming first chunk and RTF, the long requests' line, and how many fell back to the left-context decoder.

usage: python3 mixed_long.py --a LABEL=TREE --b LABEL=TREE --harness TREE --root DIR [--port 18000] [--long-requests 32]
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from long_stream_client import start_server  # noqa: E402
from seeded_tail import bench, identity, stop_group, wait_healthy  # noqa: E402

CONCURRENCY_PER_CLIENT = 8
HERE = Path(__file__).resolve().parent


def run_long(harness: str, port: int, requests: int, out: Path, label: str) -> str:
    done = subprocess.run(
        [sys.executable, str(HERE / "long_stream_client.py"), "--port", str(port), "--concurrency", str(CONCURRENCY_PER_CLIENT),
         "--requests", str(requests), "--harness", harness, "--out", str(out), "--label", label, "--no-stream"],
        capture_output=True, text=True, check=False,
    )
    (out.parent / f"{label}.long.txt").write_text(done.stdout + done.stderr, encoding="utf-8")
    lines = [line for line in done.stdout.splitlines() if " of " in line and " ok;" in line]
    return lines[-1] if lines else f"long client rc {done.returncode}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--long-requests", type=int, default=32)
    args = parser.parse_args()
    arms = [tuple(value.split("=", 1)) for value in (args.a, args.b)]
    media_root = Path(args.harness) / "docs/_static/audio"
    port = args.port
    for order, cards in (("o1", (0, 1)), ("o2", (1, 0))):
        root = args.root / order
        root.mkdir(parents=True, exist_ok=True)
        placed = [(label, tree, card, port + 200 * index) for index, ((label, tree), card) in enumerate(zip(arms, cards))]
        servers = []
        long_lines = {}
        try:
            for label, tree, card, arm_port in placed:
                print(f"{order} {label}: card {card} port {arm_port} {identity(tree)}", flush=True)
                servers.append((start_server(tree, card, arm_port, media_root, root / f"serve_{label}.log"), arm_port))
            for server, arm_port in servers:
                wait_healthy(arm_port, server)
            with concurrent.futures.ThreadPoolExecutor(2 * len(placed)) as pool:
                stream_jobs = [
                    pool.submit(bench, tree, arm_port, root / label / "stream", CONCURRENCY_PER_CLIENT, None, True)
                    for label, tree, _, arm_port in placed
                ]
                long_jobs = {
                    label: pool.submit(run_long, args.harness, arm_port, args.long_requests, root / label / "long", label)
                    for label, _, _, arm_port in placed
                }
                for job in stream_jobs:
                    job.result()
                long_lines = {label: job.result() for label, job in long_jobs.items()}
        finally:
            for server, arm_port in servers:
                stop_group(server, arm_port)
        for label, _, card, _ in placed:
            stream = json.loads((root / label / "stream" / "bench" / "speed_results.json").read_text())["summary"]
            log = (root / f"serve_{label}.log").read_text(errors="replace")
            print(
                f"{order} {label} card {card}: stream done {stream['completed_requests']}/{stream['total_requests']} first chunk mean "
                f"{1000 * stream['audio_ttfp_mean_s']:.1f} p50 {1000 * stream['audio_ttfp_median_s']:.1f} p95 "
                f"{1000 * stream['audio_ttfp_p95_s']:.1f} p99 {1000 * stream['audio_ttfp_p99_s']:.1f} ms RTF {stream['rtf_mean']:.4f}; "
                f"tracebacks {log.count('Traceback')}, left context {log.count('left-context decoder')}",
                flush=True,
            )
            print(f"  {long_lines.get(label, 'no long line')}", flush=True)
        port += 400


if __name__ == "__main__":
    main()
