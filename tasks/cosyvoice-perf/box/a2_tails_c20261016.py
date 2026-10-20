"""A2's tails from a pair's own records, no rerun (LIFECYCLE_20261016.md 9): per request first audio, chunk gaps,
tokens and the server's token rate (speed_results.json), and the servers' per step vocoder lines ("Fun-CosyVoice3
causal|leftover Flow batch size=N" in serve_gpu*.log).

1. Paired per request first audio and latency against a control pair (p_a3: unseeded c16 never reaches A3's change,
   so it is a same code spread on the same box).
2. The latency tail split into first audio, middle gaps and the final gap, the top 2 % per arm and the 2 % with the
   largest paired increase.
3. c16 first audio split into the AR's share (28 tokens at the request's token rate) and the rest.
4. Vocoder steps, rows per step and steps a second per arm, in each cell's window.

usage: python3 a2_tails_c20261016.py <copy of /data/c20261016> (artifacts/cosyvoice-c20261016b)"""

import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ORDERS = (("a", ("gpu0", "gpu1")), ("b", ("gpu1", "gpu0")))


def load(path):
    return {row["id"]: row for row in json.loads(path.read_text())["per_request"] if row["is_success"]}


def arms(root, pair, order, cell):
    base, arm = dict(ORDERS)[order]
    return (load(root / pair / order / base / cell / "speed_results.json"),
            load(root / pair / order / arm / cell / "speed_results.json"))


def parts(row):
    gaps = row["inter_chunk_s"] or []
    return {"lat": row["latency_s"], "ttfp": row["audio_ttfp_s"], "mid": sum(gaps[:-1]),
            "last": gaps[-1] if gaps else 0.0, "ar": row["completion_tokens"] / row["output_token_rate"],
            "tok": row["completion_tokens"]}


def paired(root, pair, cell, key):
    for order, _ in ORDERS:
        base, arm = arms(root, pair, order, cell)
        diffs = sorted(1000 * (arm[i][key] - base[i][key]) for i in base if i in arm)
        print(f"  {pair} {cell} {key} {order}: median {statistics.median(diffs):+.1f} ms, mean "
              f"{statistics.mean(diffs):+.1f} ms, p10 {diffs[len(diffs) // 10]:+.1f}, p90 {diffs[9 * len(diffs) // 10]:+.1f}")


def tail(root, pair, cell):
    for order, _ in ORDERS:
        base, arm = arms(root, pair, order, cell)
        for name, run in (("base", base), ("arm", arm)):
            rows = sorted((parts(row) for row in run.values()), key=lambda p: p["lat"])
            top = rows[-len(rows) // 50:]
            print(f"  {pair} {cell} {order} {name} top 2 %: " + " ".join(
                f"{k} {statistics.mean(r[k] for r in top):.2f}" for k in ("lat", "ttfp", "mid", "last", "ar", "tok")))
        ids = [i for i in base if i in arm]
        worst = [i for _, i in sorted(((arm[i]["latency_s"] - base[i]["latency_s"], i) for i in ids), reverse=True)]
        worst = worst[: len(ids) // 50]
        print(f"  {pair} {cell} {order} paired worst 2 %: " + " ".join(
            f"{k} {statistics.mean(parts(arm[i])[k] - parts(base[i])[k] for i in worst):+.2f}"
            for k in ("lat", "ttfp", "mid", "last", "ar", "tok")))


def first_audio_split(root, pair):
    for order, _ in ORDERS:
        base, arm = arms(root, pair, order, "c16")

        def split(row):
            ar = min(28, row["completion_tokens"]) / row["output_token_rate"]
            return ar, row["audio_ttfp_s"] - ar

        ids = [i for i in base if i in arm]
        for name, run in (("base", base), ("arm", arm)):
            print(f"  {pair} c16 {order} {name}: AR to 28 tokens {statistics.mean(split(run[i])[0] for i in ids):.3f} s, "
                  f"the rest {statistics.mean(split(run[i])[1] for i in ids):.3f} s")


def vocoder_steps(root, pair):
    def stamp(line):
        return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()

    for order, gpus in ORDERS:
        for gpu in gpus:
            starts = {}
            for cell, concurrency in (("c16", 16), ("c64", 64)):
                log = root / pair / order / f"bench_{gpu}_{cell}.log"
                if log.exists():
                    starts[cell] = next(stamp(line) for line in log.open()
                                        if f"max_concurrency={concurrency})" in line and "Benchmarking" in line)
            steps = [(stamp(line), m.group(1), int(m.group(2))) for line in (root / pair / order / f"serve_{gpu}.log").open()
                     if (m := re.search(r"Fun-CosyVoice3 (causal|leftover) Flow batch size=(\d+)", line))]
            bounds = (("c16", starts.get("c16"), starts.get("c64", float("inf"))), ("c64", starts.get("c64"), float("inf")))
            for cell, low, high in bounds:
                window = [s for s in steps if low is not None and low <= s[0] < high]
                if not window:
                    continue
                span = window[-1][0] - window[0][0]
                print(f"  {pair} {order} {gpu} {cell}: " + "; ".join(
                    f"{kind} {len(rows)} steps, rows {statistics.mean(rows):.2f}, {len(rows) / span:.1f}/s"
                    for kind in ("causal", "leftover") if (rows := [s[2] for s in window if s[1] == kind])))


if __name__ == "__main__":
    root = Path(sys.argv[1])
    print("1. paired first audio and latency")
    for key in ("audio_ttfp_s", "latency_s"):
        paired(root, "p_a3", "c16", key)
        paired(root, "p_a2", "c16", key)
        paired(root, "p_a2", "c64", key)
    print("2. latency tail split")
    for cell in ("c16", "c64"):
        tail(root, "p_a2", cell)
    print("3. c16 first audio split")
    first_audio_split(root, "p_a3")
    first_audio_split(root, "p_a2")
    print("4. vocoder steps")
    vocoder_steps(root, "p_a2")
