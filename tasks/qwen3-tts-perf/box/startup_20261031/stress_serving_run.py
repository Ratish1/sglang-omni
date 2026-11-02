"""One server under the TTS serving stress harness: boot a tree on a card, run a spec from the harness tree, print each
workload's successes, first audio, latency and real time factor, then the stage process's error lines.

The spec's base URL and its relative file reference are rewritten for this run, and the server is allowed to read that
reference. Both arms of a pair wait at --barrier, so their load starts together.

usage: python3 stress_serving_run.py --label base_o0 --serve-tree TREE --harness-tree TREE --spec SPEC --root DIR \
       --card 2 --port 26000 [--barrier DIR] [--parties 2]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from seeded_tail import MODEL, identity, stop_group, wait_healthy  # noqa: E402

HARNESS_TIMEOUT_S = 3000
BARRIER_TIMEOUT_S = 1200.0


def start_server(tree: str, card: int, port: int, media_root: Path, log: Path) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1")
    command = [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, "--host", "127.0.0.1",
               "--port", str(port), "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card),
               "--allowed-local-media-path", str(media_root)]
    with open(log, "w", encoding="utf-8") as handle:
        return subprocess.Popen(command, cwd=tree, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)


def wait_at_barrier(barrier: Path | None, parties: int, label: str) -> None:
    if barrier is None:
        return
    else:
        pass
    barrier.mkdir(parents=True, exist_ok=True)
    (barrier / label).write_text("", encoding="utf-8")
    deadline = time.monotonic() + BARRIER_TIMEOUT_S
    while len(list(barrier.iterdir())) < parties:
        if time.monotonic() > deadline:
            print(f"{label}: barrier timed out, the pair is not simultaneous", flush=True)
            return
        else:
            time.sleep(0.05)


def run_harness(harness: str, spec_path: Path, out: Path) -> int:
    env = dict(os.environ, PYTHONPATH=harness, CUDA_VISIBLE_DEVICES="")
    command = [sys.executable, "-m", "benchmarks.eval.benchmark_tts_serving", "--spec", str(spec_path), "--out", str(out)]
    with open(out.parent / "harness.log", "w", encoding="utf-8") as log:
        return subprocess.run(command, cwd=harness, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=HARNESS_TIMEOUT_S, check=False).returncode


def percentile(values: list[float], share: float) -> float:
    return sorted(values)[min(len(values) - 1, int(share * len(values)))] if values else float("nan")


def reduce_raw(out: Path) -> None:
    rows: dict[str, list[dict]] = {}
    for raw in sorted(out.rglob("*.jsonl")):
        for line in raw.read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "latency_s" in row:
                rows.setdefault(str(row.get("workload") or row.get("category") or "other"), []).append(row)
            else:
                pass
    for workload, records in sorted(rows.items()):
        done = [row for row in records if row.get("success")]
        first = [row["ttfa_s"] for row in done if row.get("ttfa_s") is not None]
        latency = [row["latency_s"] for row in done if row.get("latency_s")]
        audio = [row["audio_duration_s"] for row in done if row.get("audio_duration_s")]
        rtf = [row["rtf"] for row in done if row.get("rtf")]
        print(f"{workload}: {len(done)} of {len(records)} ok; first audio p50 {percentile(first, 0.5):.3f} p95 "
              f"{percentile(first, 0.95):.3f}; latency p50 {percentile(latency, 0.5):.2f} p95 {percentile(latency, 0.95):.2f}; "
              f"audio s mean {statistics.mean(audio) if audio else 0:.1f}; RTF mean {statistics.mean(rtf) if rtf else 0:.4f}",
              flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--serve-tree", required=True)
    parser.add_argument("--harness-tree", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--barrier", type=Path, default=None)
    parser.add_argument("--parties", type=int, default=2)
    args = parser.parse_args()
    root = args.root / args.label
    root.mkdir(parents=True, exist_ok=True)
    media_root = Path(args.harness_tree) / "docs/_static/audio"
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    spec["base_url"] = f"http://127.0.0.1:{args.port}"
    spec["params"]["file_ref_audio"] = f"file://{media_root / 'male-voice.wav'}"
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    print(f"{args.label}: tree {args.serve_tree} {identity(args.serve_tree)} card {args.card} spec {args.spec.name}", flush=True)
    server = start_server(args.serve_tree, args.card, args.port, media_root, root / "serve.log")
    try:
        wait_healthy(args.port, server)
        wait_at_barrier(args.barrier, args.parties, args.label)
        started = time.monotonic()
        code = run_harness(args.harness_tree, spec_path, root / "out")
        print(f"{args.label}: harness exit {code} in {time.monotonic() - started:.1f} s", flush=True)
    finally:
        stop_group(server, args.port)
    reduce_raw(root / "out")
    results = root / "out" / "results.json"
    if results.exists():
        overall = json.loads(results.read_text(encoding="utf-8")).get("overall", {})
        print(f"{args.label}: overall passed {overall.get('passed')}; {json.dumps({k: overall.get(k) for k in ('total_requests', 'successful_requests', 'failed_requests', 'throughput_rps')})}", flush=True)
    else:
        pass
    log = (root / "serve.log").read_text(errors="replace")
    print(f"{args.label}: Tracebacks {log.count('Traceback')}, out of memory {log.count('out of memory')}", flush=True)


if __name__ == "__main__":
    main()
