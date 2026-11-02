"""First chunks while non-streaming requests decode on the same worker. Two trees at the same time, one server per card,
each serving SeedTTS EN 1.7B Base to a streaming client and a non-streaming client at once, 8 requests in flight each,
unseeded, every prompt, both card orders. Prints each arm's streaming first chunk and RTF and non-streaming RTF and latency.

usage: python3 mixed_load.py --a LABEL=TREE --b LABEL=TREE --root DIR [--port 22000]
"""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from seeded_tail import bench, identity, start, stop_group, wait_healthy  # noqa: E402

CONCURRENCY_PER_CLIENT = 8


def summary(out: Path) -> dict[str, float]:
    return json.loads((out / "bench" / "speed_results.json").read_text())["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=22000)
    args = parser.parse_args()
    arms = [tuple(value.split("=", 1)) for value in (args.a, args.b)]
    port = args.port
    for order, cards in (("o1", (0, 1)), ("o2", (1, 0))):
        root = args.root / order
        root.mkdir(parents=True, exist_ok=True)
        placed = [(label, tree, card, port + 200 * index) for index, ((label, tree), card) in enumerate(zip(arms, cards))]
        servers = []
        try:
            for label, tree, card, arm_port in placed:
                print(f"{order} {label}: card {card} port {arm_port} {identity(tree)}", flush=True)
                servers.append((start(tree, card, arm_port, root / f"serve_{label}.log"), arm_port))
            for server, arm_port in servers:
                wait_healthy(arm_port, server)
            jobs = [
                (tree, arm_port, root / label / mode, CONCURRENCY_PER_CLIENT, None, mode == "stream")
                for label, tree, _, arm_port in placed
                for mode in ("stream", "nostream")
            ]
            with concurrent.futures.ThreadPoolExecutor(len(jobs)) as pool:
                list(pool.map(lambda job: bench(*job), jobs))
        finally:
            for server, arm_port in servers:
                stop_group(server, arm_port)
        for label, _, card, _ in placed:
            stream, whole = summary(root / label / "stream"), summary(root / label / "nostream")
            log = (root / f"serve_{label}.log").read_text(errors="replace")
            print(
                f"{order} {label} card {card}: stream done {stream['completed_requests']}/{stream['total_requests']} first chunk mean "
                f"{1000 * stream['audio_ttfp_mean_s']:.1f} p50 {1000 * stream['audio_ttfp_median_s']:.1f} p95 {1000 * stream['audio_ttfp_p95_s']:.1f} "
                f"p99 {1000 * stream['audio_ttfp_p99_s']:.1f} ms RTF {stream['rtf_mean']:.4f}; non-streaming done "
                f"{whole['completed_requests']}/{whole['total_requests']} RTF {whole['rtf_mean']:.4f} latency mean "
                f"{1000 * whole['latency_mean_s']:.1f} p95 {1000 * whole['latency_p95_s']:.1f} ms; tracebacks {log.count('Traceback')}, "
                f"left context {log.count('left-context decoder')}",
                flush=True,
            )
        port += 400


if __name__ == "__main__":
    main()
