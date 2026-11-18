"""Pipeshift's bar past the knee, two trees at once on two cards: per seed, both servers boot (each
pinned to its card's half of the socket), Nari's client (unmodified, pinned to the other socket) runs
every rate against both at once, and the cards swap for the next seed. Each rate writes the client's
own summary.json under OUT/seed<seed>/<arm>/rps<rate>, and rps<rate>.procs.jsonl: every
PROCESS_SAMPLE_S, each server and client thread's CPU share and the /health round trip, which waits
behind everything else on the API event loop.

usage: python3 bar_overload_pair.py --a-tree MAIN --b-tree BRANCH --out DIR [--rates 60 70 80] [--seeds 0 1 2]
       [--a-arg ARG ...] [--b-arg ARG ...]
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx

from seedtts_pair import CARD_CPUS, CLIENT_CPUS, serve, stop
from served_baseline import MODELS, wait_healthy

CLIENT_DIR = Path("/data/tts_bench_src")
CLIENT_COMMIT = "a2d65e1"
DATASET = CLIENT_DIR / "tts_bench/datasets/seed-tts-eval.jsonl"
RATE_TIMEOUT_S = 1800
SETTLE_S = 30
MEMORY_SAMPLE_S = 5
PROCESS_SAMPLE_S = 2
HEALTH_TIMEOUT_S = 30
ENGINE_THREAD = "scheduler-tts_e"
PYSPY_WINDOW_S = 20


def prepare_client() -> None:
    if DATASET.exists():
        return
    else:
        pass
    subprocess.run(["git", "clone", "-q", "https://github.com/nari-labs/benchmarks", str(CLIENT_DIR)], check=True)
    subprocess.run(["git", "-C", str(CLIENT_DIR), "checkout", "-q", CLIENT_COMMIT], check=True)
    subprocess.run(["uv", "sync", "-q", "--all-groups"], cwd=CLIENT_DIR / "tts_bench", check=True)
    subprocess.run(["uv", "run", "-q", "bench", "dataset", "prepare"], cwd=CLIENT_DIR / "tts_bench", check=True)


def sample_gpu_memory(out: Path, stop_event: threading.Event) -> None:
    """Every card's used memory every MEMORY_SAMPLE_S while a rate runs, one CSV row per sample."""
    with out.open("w") as handle:
        while not stop_event.is_set():
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            ).stdout.split()
            handle.write(f"{time.time():.0f},{','.join(used)}\n")
            handle.flush()
            stop_event.wait(MEMORY_SAMPLE_S)


def thread_ticks(pid: int) -> dict[int, tuple[str, int]]:
    """Each thread of pid: its name and its user plus system clock ticks so far."""
    threads = {}
    for task in Path(f"/proc/{pid}/task").glob("[0-9]*"):
        try:
            stat = (task / "stat").read_text()
        except OSError:
            continue
        fields = stat[stat.rindex(")") + 2:].split()
        threads[int(task.name)] = (stat[stat.index("(") + 1:stat.rindex(")")], int(fields[11]) + int(fields[12]))
    return threads


def arm_processes(server: subprocess.Popen, port: int) -> dict[int, str]:
    """The arm's processes by role: the API process is the server's own pid, the stages share its
    process group, and the client is the bench process aimed at the arm's port."""
    roles = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(entry.name)
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if pid == server.pid:
                roles[pid] = "api"
            elif os.getpgid(pid) == server.pid:
                roles[pid] = "stage"
            elif f"127.0.0.1:{port} " in cmdline and "bench run" in cmdline:
                roles[pid] = "client"
            else:
                pass
        except (OSError, ValueError):
            continue
    return roles


def sample_processes(server: subprocess.Popen, port: int, out: Path, stop_event: threading.Event) -> None:
    """Every PROCESS_SAMPLE_S: the /health round trip and payload, and each arm thread's CPU share
    (cores) since the previous sample."""
    clock_ticks = os.sysconf("SC_CLK_TCK")
    previous: dict[tuple[int, int], int] = {}
    previous_s = time.monotonic()
    with out.open("w") as handle, httpx.Client(timeout=HEALTH_TIMEOUT_S) as client:
        while not stop_event.is_set():
            start_s = time.monotonic()
            try:
                health = client.get(f"http://127.0.0.1:{port}/health").json()
                health_s = time.monotonic() - start_s
            except (httpx.HTTPError, ValueError):
                health = {}
                health_s = None
            now_s = time.monotonic()
            current = {}
            threads = []
            for pid, role in arm_processes(server, port).items():
                for tid, (name, ticks) in thread_ticks(pid).items():
                    current[(pid, tid)] = ticks
                    if (pid, tid) in previous and ticks > previous[(pid, tid)]:
                        cpu = (ticks - previous[(pid, tid)]) / clock_ticks / (now_s - previous_s)
                        threads.append({"role": role, "pid": pid, "tid": tid, "name": name, "cpu": round(cpu, 3)})
                    else:
                        pass
            health.pop("stages", None)
            handle.write(json.dumps({"t": time.time(), "health_s": health_s, "health": health, "threads": threads}) + "\n")
            handle.flush()
            previous, previous_s = current, now_s
            stop_event.wait(PROCESS_SAMPLE_S)


def engine_process(server: subprocess.Popen) -> int:
    """The server process (same process group) that runs the talker's scheduler thread."""
    for task in Path("/proc").glob("[0-9]*/task/[0-9]*/comm"):
        try:
            pid = int(task.parts[2])
            if os.getpgid(pid) == server.pid and task.read_text().strip() == ENGINE_THREAD:
                return pid
            else:
                pass
        except (OSError, ValueError):
            continue
    raise RuntimeError(f"no {ENGINE_THREAD} thread under process group {server.pid}")


def record_pyspy(server: subprocess.Popen, delay_s: int, out: Path) -> None:
    """After delay_s, interpreter lock holders in the engine process and in the API process (the server's own pid)
    over the same window, then every engine thread's stacks, PYSPY_WINDOW_S each."""
    time.sleep(delay_s)
    pid = engine_process(server)
    api = subprocess.Popen(
        ["py-spy", "record", "--pid", str(server.pid), "--gil", "--nonblocking", "--threads", "--rate", "100",
         "--duration", str(PYSPY_WINDOW_S), "--format", "raw", "--output", f"{out}.api.pyspy.txt"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    for mode, flag in (("gil", "--gil"), ("all", "--idle")):
        subprocess.run(
            ["py-spy", "record", "--pid", str(pid), flag, "--nonblocking", "--threads", "--rate", "100",
             "--duration", str(PYSPY_WINDOW_S), "--format", "raw", "--output", f"{out}.{mode}.pyspy.txt"],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False,
        )
    api.wait(timeout=PYSPY_WINDOW_S * 4)


def run_rate(port: int, model: str, rate: int, seed: int, out: Path, duration_s: int) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset", "-c", CLIENT_CPUS, "uv", "run", "-q", "bench", "run", "--target", "sglang-omni",
        "--base-url", f"http://127.0.0.1:{port}", "--model", model, "--voice", "Ryan", "--language", "English",
        "--dataset", "datasets/seed-tts-eval.jsonl", "--rps", str(rate), "--warmup", "30s", "--duration", f"{duration_s}s",
        "--arrival", "poisson", "--seed", str(seed), "--timeout", "120s", "--max-in-flight", "4096",
        "--output", str(out),
    ]
    with (out.parent / f"bench_rps{rate}.log").open("w") as log:
        return subprocess.run(
            command, cwd=CLIENT_DIR / "tts_bench", stdout=log, stderr=subprocess.STDOUT, timeout=RATE_TIMEOUT_S,
            check=False,
        ).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rates", type=int, nargs="+", default=[60, 70, 80])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--port", type=int, default=22000)
    parser.add_argument("--duration", type=int, default=120, help="measured seconds per rate")
    parser.add_argument("--pyspy-at", type=int, default=None, help="seconds into each rate to start the py-spy windows")
    parser.add_argument("--a-arg", action="append", default=[], help="one more serve argument for arm a, repeatable")
    parser.add_argument("--b-arg", action="append", default=[], help="one more serve argument for arm b, repeatable")
    args = parser.parse_args()
    prepare_client()
    model, _ = MODELS["customvoice"]
    trees = {"a": args.a_tree, "b": args.b_tree}
    server_args = {"a": tuple(args.a_arg), "b": tuple(args.b_arg)}
    ports = {"a": args.port, "b": args.port + 200}
    for index, seed in enumerate(args.seeds):
        cards = {"a": 0, "b": 1} if index % 2 == 0 else {"a": 1, "b": 0}
        seed_out = args.out / f"seed{seed}"
        servers = {}
        try:
            for arm in ("a", "b"):
                (seed_out / arm).mkdir(parents=True, exist_ok=True)
                identity = subprocess.check_output(["git", "rev-parse", "HEAD", "HEAD:sglang_omni"], cwd=trees[arm], text=True)
                (seed_out / arm / "tree.txt").write_text(
                    f"{trees[arm]} card {cards[arm]} cpus {CARD_CPUS[cards[arm]]} args {' '.join(server_args[arm])}\n{identity}"
                )
                servers[arm] = serve(
                    trees[arm], model, cards[arm], ports[arm], seed_out / arm / "serve.log", {}, server_args[arm]
                )
            for arm in ("a", "b"):
                wait_healthy(ports[arm], servers[arm])
            print(f"{time.strftime('%T')} seed {seed} healthy, a card {cards['a']}, b card {cards['b']}", flush=True)
            for rate in args.rates:
                stop_event = threading.Event()
                (seed_out / f"cards_rps{rate}.txt").write_text(f"a card {cards['a']}, b card {cards['b']}\n")
                sampler = threading.Thread(
                    target=sample_gpu_memory, args=(seed_out / f"gpu_memory_rps{rate}.csv", stop_event), daemon=True
                )
                sampler.start()
                process_samplers = [
                    threading.Thread(
                        target=sample_processes,
                        args=(servers[arm], ports[arm], seed_out / arm / f"rps{rate}.procs.jsonl", stop_event),
                        daemon=True,
                    )
                    for arm in ("a", "b")
                ]
                for process_sampler in process_samplers:
                    process_sampler.start()
                profilers = [
                    threading.Thread(
                        target=record_pyspy, args=(servers[arm], args.pyspy_at, seed_out / arm / f"rps{rate}"), daemon=True
                    )
                    for arm in ("a", "b")
                    if args.pyspy_at is not None
                ]
                for profiler in profilers:
                    profiler.start()
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {
                        arm: pool.submit(run_rate, ports[arm], model, rate, seed, seed_out / arm / f"rps{rate}", args.duration)
                        for arm in ("a", "b")
                    }
                    codes = {arm: future.result() for arm, future in futures.items()}
                stop_event.set()
                sampler.join()
                for process_sampler in process_samplers:
                    process_sampler.join()
                for profiler in profilers:
                    profiler.join()
                print(f"{time.strftime('%T')} seed {seed} rps {rate} rc {codes}", flush=True)
                time.sleep(SETTLE_S)
        finally:
            for server in servers.values():
                stop(server)
    print(f"{time.strftime('%T')} bar_overload_pair done", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("UV_NO_PROGRESS", "1")
    main()
