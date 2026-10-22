"""Same-time pair of two server arms of one tree on the two cards, then the same pair with the cards swapped.

usage: python3 pair.py --model asr --out /data/runtime-perf/p_asr_coalesce --loads 8,16,64 --repeats 3
       --a-extra "" --b-extra "--asr.factory.prefill_coalesce_after_builds_during_decode false" [--port 24000]
       [--events-load 16] [--no-swap] [--bench asr_longform --asr-dataset longlibriheavy-30] [--warm-bench seedtts]
       [--fresh-per-load] [--meta PATH] [--bench-extra "..."]

Each round boots arm a on one card and arm b on the other at once, waits for both, warms them with one pass of
--warm-bench (default: the measured bench), then per load runs both arms' benchmarks at the same moment (--repeats
measured passes each; ASR repeats inside its benchmark, the others run the benchmark that many times). --events-load
adds one recorder pass per arm at that load after the measured passes. With --fresh-per-load each load gets its own
boot and warm pass, and the measured pass is the first pass of the measured corpus on that server: no request of it
can hit an encoder or prefix cache filled by an earlier pass (use --repeats 1). Round 2 swaps the cards. Each arm and
round writes the same files as baseline.py under <out>/r<round>_<arm>/.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import (  # noqa: E402
    MODELS,
    TREE,
    bench_command,
    gpu_snapshot,
    group_pids,
    post,
    process_table,
    stop_group,
    thread_cpu,
    wait_healthy,
)

BENCH_TIMEOUT_S = 10800.0


def boot(model: str, card: int, port: int, extra: str, out: Path) -> subprocess.Popen:
    spec = MODELS[model]
    env = dict(os.environ, PYTHONPATH=str(TREE), SGLANG_OMNI_STRICT_PORT="1")
    serve = [
        sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", spec["path"],
        *(["--config", spec["config"]] if spec["config"] else []),
        "--host", "127.0.0.1", "--port", str(port),
        *shlex.split(spec["placement"].format(card=card)), *shlex.split(extra),
    ]
    identity = subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo WORKING_TREE_DIRTY",
        shell=True, cwd=TREE, capture_output=True, text=True,
    ).stdout
    out.mkdir(parents=True, exist_ok=True)
    (out / "identity.txt").write_text(f"tree {TREE}\n{identity}serve {shlex.join(serve)}\ncard {card}\n"
                                      f"loadavg {os.getloadavg()}\n", encoding="utf-8")
    with open(out / "serve.log", "a", encoding="utf-8") as serve_log:
        return subprocess.Popen(serve, cwd=TREE, env=env, stdout=serve_log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def bench_for(args, bench: str, port: int, load: int, out: Path, repeats: int) -> list[list[str]]:
    extra = shlex.split(args.bench_extra)
    if args.model == "asr":
        command = bench_command(args.model, port, load, out, warmup=False, repeats=repeats, bench=bench,
                                meta=args.meta, asr_dataset=args.asr_dataset)
        return [command + extra]
    else:
        return [bench_command(args.model, port, load, out / f"rep{index}", warmup=False, bench=bench, meta=args.meta,
                              asr_dataset=args.asr_dataset) + extra
                for index in range(repeats)]


def run_cell(args, arms, load, kind, bench, repeats, progress):
    """Run one cell on both arms at the same moment; each arm runs its commands in sequence."""
    procs = {}
    starts = {}
    for name, (server, port, out) in arms.items():
        cell = out / f"c{load}_{kind}"
        cell.mkdir(parents=True, exist_ok=True)
        if kind == "events":
            post(port, "/start_request_profile", {"run_id": f"c{load}", "event_dir": str(cell / "events")})
        else:
            pass
        commands = bench_for(args, bench, port, load, cell / "bench", repeats)
        (cell / "bench_command.txt").write_text("\n".join(shlex.join(c) for c in commands) + "\n", encoding="utf-8")
        script = " && ".join(shlex.join(c) for c in commands)
        starts[name] = (thread_cpu(server.pid), time.time())
        procs[name] = subprocess.Popen(
            ["bash", "-c", script], cwd=TREE, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(TREE)),
            stdout=open(cell / "bench.log", "w"), stderr=subprocess.STDOUT,
        )
    for name, proc in procs.items():
        rc = proc.wait(timeout=BENCH_TIMEOUT_S)
        server, port, out = arms[name]
        cell = out / f"c{load}_{kind}"
        cpu_start, t0 = starts[name]
        (cell / "thread_cpu.json").write_text(json.dumps({"start": cpu_start, "end": thread_cpu(server.pid)}),
                                              encoding="utf-8")
        (cell / "window.json").write_text(json.dumps({"start": t0, "end": time.time(), "rc": rc}), encoding="utf-8")
        if kind == "events":
            post(port, "/stop_request_profile", {"run_id": f"c{load}"})
        else:
            pass
        progress.write(f"{time.strftime('%H:%M:%S')} {out.name} c{load} {kind} rc={rc} "
                       f"{time.time() - t0:.1f}s load={os.getloadavg()[0]:.1f}\n")
        progress.flush()


def start_arms(args, index, cards, ports, progress):
    servers = {}
    for name, card, extra in (("a", cards[0], args.a_extra), ("b", cards[1], args.b_extra)):
        servers[name] = boot(args.model, card, ports[name], extra, Path(args.out) / f"r{index}_{name}")
        (Path(args.out) / f"r{index}_{name}" / "server.pgid").write_text(f"{servers[name].pid}\n", encoding="utf-8")
    for name, server in servers.items():
        wait_healthy(ports[name], server)
        (Path(args.out) / f"r{index}_{name}" / "processes.txt").write_text(process_table(server.pid), encoding="utf-8")
    progress.write(f"{time.strftime('%H:%M:%S')} round {index} healthy, a card {cards[0]}, b card {cards[1]}\n")
    progress.flush()
    return servers


def stop_arms(servers, index, progress):
    for server in servers.values():
        stop_group(server)
    progress.write(f"{time.strftime('%H:%M:%S')} round {index} stopped, groups left "
                   f"{[group_pids(s.pid) for s in servers.values()]}\n")
    progress.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--loads", default="16")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--a-extra", default="")
    parser.add_argument("--b-extra", default="")
    parser.add_argument("--port", type=int, default=24000)
    parser.add_argument("--events-load", type=int, default=0)
    parser.add_argument("--no-swap", action="store_true")
    parser.add_argument("--bench", default="seedtts", choices=("seedtts", "videomme", "videomme_text", "asr_longform"))
    parser.add_argument("--warm-bench", default="", help="bench of the warm pass (default: --bench)")
    parser.add_argument("--meta", default="")
    parser.add_argument("--bench-extra", default="")
    parser.add_argument("--asr-dataset", default="meanwhile")
    parser.add_argument("--fresh-per-load", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    warm_bench = args.warm_bench or args.bench
    progress = open(out / "progress.txt", "a", encoding="utf-8")
    rounds = [(0, 1)] if args.no_swap else [(0, 1), (1, 0)]
    loads = [int(item) for item in args.loads.split(",")]
    for index, cards in enumerate(rounds, start=1):
        groups = [[load] for load in loads] if args.fresh_per_load else [loads]
        for group_index, group in enumerate(groups):
            # note (ratish): each server also binds its port plus 100; a 20-port step per group keeps every port of
            # a round distinct, so no boot waits on a port an earlier group left in TIME_WAIT
            base = args.port + 1000 * (index - 1) + 20 * group_index
            ports = {"a": base, "b": base + 500}
            servers = {}
            try:
                servers = start_arms(args, index, cards, ports, progress)
                arms = {name: (servers[name], ports[name], out / f"r{index}_{name}") for name in servers}
                run_cell(args, arms, group[0], f"warm{group_index}", warm_bench, 1, progress)
                for load in group:
                    run_cell(args, arms, load, "clean", args.bench, args.repeats, progress)
                    if load == args.events_load:
                        run_cell(args, arms, load, "events", args.bench, 1, progress)
                    else:
                        pass
                for name, (server, port, arm_out) in arms.items():
                    (arm_out / "gpu_after.txt").write_text(gpu_snapshot(), encoding="utf-8")
                    (arm_out / "model_info.json").write_text(
                        httpx.get(f"http://127.0.0.1:{port}/model_info", timeout=30, trust_env=False).text,
                        encoding="utf-8",
                    )
            finally:
                stop_arms(servers, index, progress)
    progress.write(f"{time.strftime('%H:%M:%S')} done\n")


if __name__ == "__main__":
    main()
