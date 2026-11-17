"""Same-time engine loop ledgers of two trees: each order boots both arms at once under nsys with the
loop probe (one per card, each pinned to its card's half of the socket), warms them, then records a
streaming c64 window and a streaming c16 window on both at once, stops both by process group, and
reads each report with ar_loop_ledger.py. The second order swaps the cards.
Results under OUT/order{0,1}/{a,b}/: <cell>.nsys-rep, <cell>.sqlite, ledger_<cell>.txt, bench_<cell>/.

usage: python3 loop_pair.py --a-tree MAIN --b-tree BRANCH --out DIR
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from seedtts_pair import CARD_CPUS, bench, stop
from served_baseline import MODELS, wait_healthy

HERE = Path(__file__).resolve().parent
WARM_SAMPLES = 128
# (cell, concurrency, samples): a c64 window of about 8 s and a c16 window of about 8 s
CELLS = (("c64", 64, 512), ("c16", 16, 320))
LEDGER_SKIP_S = 1.0


def launch(tree: Path, model: str, card: int, port: int, session: str, log: Path, should_trace_gil: bool) -> subprocess.Popen:
    if should_trace_gil:
        # note (ratish): lock waits per step for the ledger, and hold time per thread and marked function for gil_owners.py.
        marks = log.parent / "marks_attribution.json"
        marks.write_text(
            json.dumps(json.loads((HERE / "marks.json").read_text()) + json.loads((HERE / "marks_capture.json").read_text()))
        )
        tracing = ["--trace=cuda,nvtx,osrt,python-gil", f"--python-functions-trace={marks}"]
    else:
        tracing = ["--trace=cuda,nvtx"]
    command = [
        "taskset", "-c", CARD_CPUS[card],
        "nsys", "launch", f"--session-new={session}", *tracing, "--cuda-graph-trace=node",
        sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", model, "--host", "127.0.0.1",
        "--port", str(port), "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": f"{HERE / 'loop_probe'}:{tree}",
        "OMNI_LOOP_PROBE": "1",
        "SGLANG_OMNI_STRICT_PORT": "1",
    }
    with log.open("w") as handle:
        return subprocess.Popen(
            command, cwd=tree, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )


def run_order(
    order: int, trees: dict[str, Path], cards: dict[str, int], model: str, extra: list[str], port: int, out: Path,
    should_trace_gil: bool,
) -> None:
    ports = {"a": port, "b": port + 200}
    sessions = {arm: f"loop{order}{arm}" for arm in ("a", "b")}
    arm_outs = {arm: out / f"order{order}" / arm for arm in ("a", "b")}
    servers = {}
    try:
        for arm in ("a", "b"):
            arm_outs[arm].mkdir(parents=True, exist_ok=True)
            identity = subprocess.check_output(["git", "rev-parse", "HEAD", "HEAD:sglang_omni"], cwd=trees[arm], text=True)
            (arm_outs[arm] / "tree.txt").write_text(f"{trees[arm]} card {cards[arm]}\n{identity}")
            servers[arm] = launch(
                trees[arm], model, cards[arm], ports[arm], sessions[arm], arm_outs[arm] / "serve.log", should_trace_gil
            )
        for arm in ("a", "b"):
            wait_healthy(ports[arm], servers[arm])
        print(f"{time.strftime('%T')} order {order} healthy, a card {cards['a']}, b card {cards['b']}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for future in [
                pool.submit(bench, trees[arm], model, [*extra, "--max-samples", str(WARM_SAMPLES)], ports[arm], 64, True, arm_outs[arm] / "warm")
                for arm in ("a", "b")
            ]:
                future.result()
        for cell, concurrency, samples in CELLS:
            for arm in ("a", "b"):
                subprocess.run(
                    ["nsys", "start", f"--session={sessions[arm]}", "--sample=none", "--cpuctxsw=none",
                     f"--output={arm_outs[arm] / cell}", "--force-overwrite=true"],
                    check=True,
                )
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    arm: pool.submit(bench, trees[arm], model, [*extra, "--max-samples", str(samples)], ports[arm],
                                     concurrency, True, arm_outs[arm] / f"bench_{cell}")
                    for arm in ("a", "b")
                }
                summaries = {arm: future.result() for arm, future in futures.items()}
            for arm in ("a", "b"):
                subprocess.run(["nsys", "stop", f"--session={sessions[arm]}"], check=True)
            print(
                f"{time.strftime('%T')} order {order} {cell} recorded, rtf a {summaries['a'].get('rtf_mean')} b {summaries['b'].get('rtf_mean')}",
                flush=True,
            )
    finally:
        for server in servers.values():
            stop(server)
    for arm in ("a", "b"):
        for cell, _, _ in CELLS:
            report = arm_outs[arm] / f"{cell}.nsys-rep"
            sqlite = arm_outs[arm] / f"{cell}.sqlite"
            subprocess.run(
                ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "--output", str(sqlite), str(report)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with (arm_outs[arm] / f"ledger_{cell}.txt").open("w") as handle:
                subprocess.run(
                    [sys.executable, str(HERE / "ar_loop_ledger.py"), str(sqlite),
                     "--skip-head-s", str(LEDGER_SKIP_S), "--skip-tail-s", str(LEDGER_SKIP_S)],
                    stdout=handle, stderr=subprocess.STDOUT, check=False,
                )
            if should_trace_gil:
                with (arm_outs[arm] / f"gil_{cell}.txt").open("w") as handle:
                    subprocess.run(
                        [sys.executable, str(HERE / "gil_owners.py"), str(sqlite)],
                        stdout=handle, stderr=subprocess.STDOUT, check=False,
                    )
            else:
                pass
    print(f"{time.strftime('%T')} order {order} ledgers written", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="base")
    parser.add_argument("--port", type=int, default=25000)
    parser.add_argument("--gil", action="store_true", help="add OS waits and interpreter lock tracing")
    args = parser.parse_args()
    model, extra = MODELS[args.model]
    trees = {"a": args.a_tree, "b": args.b_tree}
    for order, cards in enumerate(({"a": 0, "b": 1}, {"a": 1, "b": 0})):
        run_order(order, trees, cards, model, extra, args.port, args.out, args.gil)
    print(f"{time.strftime('%T')} loop_pair done", flush=True)


if __name__ == "__main__":
    main()
