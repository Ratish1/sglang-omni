"""Closed-loop eligible load on a sequence of boots, each on its own port, with no event recording.

Each --boot is LABEL=TREE:PORT, run in the order given (a b b a gives both orders and each arm's two boots are its A/A).
Per boot: warm-up, then per load (name:concurrency:requests) ROUNDS passes where every request is eligible (Ryan,
English, streamed, default sampling). Writes OUT/<index>_<label>/<load>_r<i>.jsonl and prints one line per pass and
a table of per-boot medians.

usage: python3 opening_pair.py --out DIR --boot fix=TREE:31600 --boot proto=TREE:31800 [--loads c1:1:100 c64:64:1088]
"""

import argparse
import collections
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

from opening_stage_trace import SEEDTTS_META, WARMUP_REQUESTS, run_pass
from seedtts_pair import serve, stop
from served_baseline import MODELS, wait_healthy

KEYS = ("req_s", "rtf", "first_byte_p50", "first_byte_mean", "first_byte_p90", "audible_p50", "margin_p50", "margin_p1", "underruns", "stall_ms", "silence_80")


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * len(values)))]


def summarize(rows: list[dict], wall_s: float) -> dict:
    served = [row for row in rows if row["status"] == 200 and row.get("samples")]
    first = [row["first_byte_ms"] for row in served]
    audible = [row["audible_ms"] for row in served if row["audible_ms"] is not None]
    margins = [row["second_margin_ms"] for row in served if row["second_margin_ms"] is not None]
    return {
        "ok": len(served), "n": len(rows), "req_s": len(served) / wall_s,
        "rtf": statistics.mean(row["latency_s"] / row["audio_s"] for row in served),
        "first_byte_p50": statistics.median(first), "first_byte_mean": statistics.mean(first),
        "first_byte_p90": percentile(first, 0.9), "audible_p50": statistics.median(audible),
        "margin_p50": statistics.median(margins), "margin_p1": percentile(margins, 0.01),
        "underruns": sum(1 for row in served if row["stalls"]), "stall_ms": sum(row["stall_ms"] for row in served),
        "silence_80": sum(1 for row in served if (row["leading_silence_ms"] or 0) >= 80),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boot", action="append", required=True, help="LABEL=TREE:PORT, in run order")
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--loads", nargs="+", default=["c1:1:100", "c16:16:400", "c32:32:1088", "c64:64:1088"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--model", default=MODELS["customvoice"][0])
    parser.add_argument("--extra", default="{}", help="JSON of extra request fields")
    parser.add_argument("--server-args", nargs="*", default=[], help="LABEL=ARG ... extra serve arguments for one label")
    args = parser.parse_args()
    boots = []
    for boot in args.boot:
        label, rest = boot.split("=", 1)
        tree, port = rest.rsplit(":", 1)
        boots.append((label, Path(tree), int(port)))
    server_args: dict[str, list[str]] = collections.defaultdict(list)
    for entry in args.server_args:
        label, argument = entry.split("=", 1)
        server_args[label].append(argument)
    sys.path.insert(0, str(boots[0][1]))
    from benchmarks.dataset.seedtts import load_seedtts_samples

    texts = [sample.target_text for sample in load_seedtts_samples(SEEDTTS_META, split="en")]
    extra = {"model": args.model, **json.loads(args.extra)}
    table: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for index, (label, tree, port) in enumerate(boots):
        out = args.out / f"{index}_{label}"
        out.mkdir(parents=True, exist_ok=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD", "HEAD:sglang_omni"], cwd=tree, text=True).split()
        print(f"{time.strftime('%T')} boot {index} {label} {tree} {head[0]} sglang_omni {head[1]} port {port} {server_args[label]}", flush=True)
        url = f"http://127.0.0.1:{port}"
        server = serve(tree, args.model, args.card, port, out / "serve.log", {}, tuple(server_args[label]))
        (out / "serve.pgid").write_text(str(server.pid))
        try:
            wait_healthy(port, server)
            run_pass(url, texts[:WARMUP_REQUESTS], WARMUP_REQUESTS, extra)
            for load in args.loads:
                name, concurrency, requests = load.split(":")
                concurrency, requests = int(concurrency), int(requests)
                run_pass(url, texts[-max(concurrency, 8):], concurrency, extra)
                for round_index in range(args.rounds):
                    rows, wall_s = run_pass(url, texts[:requests], concurrency, extra)
                    with (out / f"{name}_r{round_index}.jsonl").open("w") as handle:
                        for row in rows:
                            handle.write(json.dumps(row) + "\n")
                    summary = summarize(rows, wall_s)
                    table[(label, name)].append(summary)
                    print(
                        f"{time.strftime('%T')} boot {index} {label} {name} r{round_index} ok {summary['ok']}/{summary['n']} "
                        + " ".join(f"{key} {summary[key]:.4g}" for key in KEYS),
                        flush=True,
                    )
        finally:
            stop(server)
    labels = list(dict.fromkeys(label for label, _, _ in boots))
    for load in args.loads:
        name = load.split(":")[0]
        for key in KEYS:
            cells = []
            for label in labels:
                values = [summary[key] for summary in table[(label, name)]]
                cells.append(f"{label} {' '.join(f'{value:.4g}' for value in values)} (mean {statistics.mean(values):.4g})")
            print(f"table {name} {key:16s} " + " | ".join(cells), flush=True)
    print(f"{time.strftime('%T')} PAIR_DONE", flush=True)


if __name__ == "__main__":
    main()
