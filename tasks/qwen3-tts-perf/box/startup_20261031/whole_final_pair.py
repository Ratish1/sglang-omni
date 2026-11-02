"""Two last reads of a whole request head against its base, one server per arm at the same time, on warm caches, both card
orders. Each order: boot time to health, then SeedTTS EN 1.7B Base non-streaming at concurrency 64 on every prompt. The first
order also sends the same seeded requests one at a time to both servers, with x_vector_only_mode (no reference frames, a
path the benchmark has no flag for) and with the reference, and compares the decoded audio between the arms: the talker's
codes match at concurrency 1, so any distance is the decode's. A correctness check on real servers, not a benchmark.

usage: python3 whole_final_pair.py --a LABEL=TREE --b LABEL=TREE --root DIR --references DIR_WITH_BENCH [--port 26000]
"""

import argparse
import concurrent.futures
import io
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from seeded_tail import MODEL, bench, identity, start, stop_group, wait_healthy  # noqa: E402

PROBE_REQUESTS = 16
PROBE_SEED = 1234


def speech(port: int, payload: dict[str, str | int | bool]) -> np.ndarray:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/speech",
        data=json.dumps({"model": MODEL, "response_format": "wav", "seed": PROBE_SEED, **payload}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        samples, _ = soundfile.read(io.BytesIO(response.read()), dtype="float32")
    return samples


def probe(ports: tuple[int, int], references: Path) -> None:
    rows = json.loads((references / "generated.json").read_text())[:PROBE_REQUESTS]
    for mode in ("x_vector_only", "reference"):
        distances = []
        length_mismatches = 0
        silent = 0
        for row in rows:
            payload: dict[str, str | int | bool] = {"input": row["target_text"], "ref_audio": str(Path(row["wav_path"]).resolve())}
            if mode == "x_vector_only":
                payload["x_vector_only_mode"] = True
            else:
                payload["ref_text"] = row["target_text"]
            first, second = speech(ports[0], payload), speech(ports[1], payload)
            if first.shape != second.shape:
                length_mismatches += 1
                continue
            else:
                pass
            silent += int(np.sqrt(np.mean(second**2)) < 1e-3)
            distances.append(float(np.linalg.norm(first - second) / max(np.linalg.norm(first), 1e-9)))
        print(
            f"probe {mode}: {len(rows)} requests, length mismatches {length_mismatches}, silent {silent}, relative L2 between arms "
            f"median {statistics.median(distances) if distances else float('nan'):.4f} max {max(distances) if distances else float('nan'):.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--port", type=int, default=26000)
    args = parser.parse_args()
    arms = [tuple(value.split("=", 1)) for value in (args.a, args.b)]
    port = args.port
    for order, cards in (("o1", (0, 1)), ("o2", (1, 0))):
        root = args.root / order
        root.mkdir(parents=True, exist_ok=True)
        placed = [(label, tree, card, port + 200 * index) for index, ((label, tree), card) in enumerate(zip(arms, cards))]
        servers = []
        started = time.monotonic()
        try:
            for label, tree, card, arm_port in placed:
                print(f"{order} {label}: card {card} port {arm_port} {identity(tree)}", flush=True)
                servers.append((start(tree, card, arm_port, root / f"serve_{label}.log"), arm_port))
            for (label, _, _, _), (server, arm_port) in zip(placed, servers):
                wait_healthy(arm_port, server)
                print(f"{order} {label}: healthy by {time.monotonic() - started:.1f} s", flush=True)
            if order == "o1":
                probe((placed[0][3], placed[1][3]), args.references)
            else:
                pass
            with concurrent.futures.ThreadPoolExecutor(len(placed)) as pool:
                list(pool.map(lambda arm: bench(arm[1], arm[3], root / arm[0], 64, None, False), placed))
        finally:
            for server, arm_port in servers:
                stop_group(server, arm_port)
        for label, _, card, _ in placed:
            summary = json.loads((root / label / "bench" / "speed_results.json").read_text())["summary"]
            log = (root / f"serve_{label}.log").read_text(errors="replace")
            tokens = re.search(r"#tokens: (\d+)", log)
            print(
                f"{order} {label} card {card}: done {summary['completed_requests']}/{summary['total_requests']} req/s {summary['throughput_qps']:.2f} "
                f"RTF {summary['rtf_mean']:.4f} latency mean {1000 * summary['latency_mean_s']:.1f} p95 {1000 * summary['latency_p95_s']:.1f} ms; "
                f"KV tokens {tokens.group(1) if tokens else '-'}; tracebacks {log.count('Traceback')}, left context {log.count('left-context decoder')}",
                flush=True,
            )
        port += 400


if __name__ == "__main__":
    main()
