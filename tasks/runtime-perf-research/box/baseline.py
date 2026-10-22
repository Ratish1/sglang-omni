"""Runtime baseline for one representative pipeline: a clean pass, a request event pass and a py-spy pass per load.

usage: python3 baseline.py --model tts --card 1 --port 22000 --out /data/runtime-perf/b1_tts --loads 1,16,64
       [--passes clean,events,pyspy] [--pyspy-delay 8] [--pyspy-seconds 15] [--serve-extra "..."]

One server from this tree, placed on --card with the stage gpu flags, healthy before any pass. Per load and pass:

  clean   the model's SeedTTS EN benchmark over the full corpus, nothing else attached
  events  the same with the request event recorder on (/start_request_profile into events/c<load>)
  pyspy   the same with py-spy record (--threads --idle --nonblocking, 100 Hz) on every server process for
          --pyspy-seconds, starting --pyspy-delay seconds into the benchmark

Every pass also stores the CPU seconds of every server thread at its start and end (thread_cpu.json) and the
benchmark window. Identity (head, tree hash, dirty check, imported file) goes to identity.txt before the boot.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

TREE = Path(__file__).resolve().parents[3]
HEALTH_TIMEOUT_S = 1800.0
STOP_TIMEOUT_S = 90.0
BENCH_TIMEOUT_S = 10800.0

MODELS = {
    "tts": {
        "path": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "config": "examples/configs/qwen3_tts_1_7b.yaml",
        "placement": "--tts_engine.gpu {card} --vocoder.gpu {card}",
    },
    "asr": {
        "path": "Qwen/Qwen3-ASR-1.7B",
        "config": "",
        "placement": "--asr.gpu {card}",
    },
    "omni_colocated": {
        "path": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "config": "examples/configs/qwen3_omni_colocated_h200.yaml",
        "placement": "",
    },
    "omni_speech": {
        "path": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "config": "",
        "placement": "",
    },
}


def bench_command(model: str, port: int, load: int, out: Path, warmup: bool) -> list[str]:
    spec = MODELS[model]
    if model == "tts":
        return [
            sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", spec["path"],
            "--use-existing-server", "--host", "127.0.0.1", "--port", str(port), "--generate-only", "--stream",
            "--lang", "en", "--warmup", "1" if warmup else "0", "--max-concurrency", str(load),
            "--output-dir", str(out), "--disable-tqdm",
        ]
    elif model == "asr":
        command = [
            sys.executable, "-m", "benchmarks.eval.benchmark_asr_seedtts", "--host", "127.0.0.1", "--port", str(port),
            "--lang", "en", "--concurrencies", str(load), "--repeats", "1", "--disable-resource-monitor",
            "--output", str(out / "asr.json"), "--save-raw-dir", str(out / "raw"),
        ]
        return command + (["--warmup"] if warmup else [])
    else:
        return [
            sys.executable, "-m", "benchmarks.eval.benchmark_omni_seedtts", "--host", "127.0.0.1", "--port", str(port),
            "--generate-only", "--stream", "--voice-clone", "--lang", "en", "--warmup", "1" if warmup else "0",
            "--max-concurrency", str(load), "--output-dir", str(out), "--disable-tqdm",
        ]


def wait_healthy(port: int, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server on port {port} exited with {server.returncode}")
        else:
            pass
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2, trust_env=False).status_code == 200:
                return
            else:
                pass
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError(f"server on port {port} not healthy in {HEALTH_TIMEOUT_S} s")


def group_pids(pgid: int) -> list[int]:
    listing = subprocess.run(["ps", "-eo", "pid=,pgid=,stat="], capture_output=True, text=True).stdout
    return [
        int(fields[0]) for fields in (line.split() for line in listing.splitlines())
        if len(fields) == 3 and fields[1] == str(pgid) and not fields[2].startswith("Z")
    ]


def stop_group(server: subprocess.Popen) -> None:
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        if not group_pids(server.pid):
            return
        else:
            pass
        time.sleep(1)
    os.killpg(server.pid, signal.SIGKILL)


def thread_cpu(pgid: int) -> dict[str, dict[str, object]]:
    """CPU seconds (user plus system) and name of every thread in the server's process group, keyed pid/tid."""
    ticks = os.sysconf("SC_CLK_TCK")
    threads = {}
    for pid in group_pids(pgid):
        try:
            tasks = list(Path(f"/proc/{pid}/task").iterdir())
        except FileNotFoundError:
            continue
        for task in tasks:
            try:
                stat = (task / "stat").read_text()
                name = (task / "comm").read_text().strip()
            except FileNotFoundError:
                continue
            fields = stat[stat.rindex(")") + 2:].split()
            threads[f"{pid}/{task.name}"] = {"name": name, "cpu_s": (int(fields[11]) + int(fields[12])) / ticks}
    return threads


def process_table(pgid: int) -> str:
    rows = []
    for pid in group_pids(pgid):
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            comm = Path(f"/proc/{pid}/comm").read_text().strip()
            affinity = len(os.sched_getaffinity(pid))
        except (FileNotFoundError, ProcessLookupError):
            continue
        rows.append(f"{pid}\t{comm}\tcpus={affinity}\t{cmdline[:300]}")
    return "\n".join(rows) + "\n"


def gpu_snapshot() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True
    ).stdout + subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout


def post(port: int, path: str, body: dict) -> str:
    return httpx.post(f"http://127.0.0.1:{port}{path}", json=body, timeout=60, trust_env=False).text


def run_pass(args, server: subprocess.Popen, load: int, kind: str, warmup: bool) -> None:
    out = Path(args.out) / f"c{load}_{kind}"
    out.mkdir(parents=True, exist_ok=True)
    if kind == "events":
        (out / "events_start.json").write_text(
            post(args.port, "/start_request_profile", {"run_id": f"c{load}", "event_dir": str(out / "events")}),
            encoding="utf-8",
        )
    else:
        pass
    command = bench_command(args.model, args.port, load, out / "bench", warmup)
    (out / "bench_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    cpu_start = thread_cpu(server.pid)
    window_start = time.time()
    with open(out / "bench.log", "w", encoding="utf-8") as bench_log:
        bench = subprocess.Popen(
            command, cwd=TREE, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(TREE)),
            stdout=bench_log, stderr=subprocess.STDOUT,
        )
        spies = []
        if kind == "pyspy":
            time.sleep(args.pyspy_delay)
            for pid in group_pids(server.pid):
                spies.append(subprocess.Popen(
                    ["py-spy", "record", "--pid", str(pid), "--rate", "100", "--duration", str(args.pyspy_seconds),
                     "--threads", "--idle", "--nonblocking", "--format", "raw", "--output", str(out / f"pyspy_{pid}.txt")],
                    stdout=subprocess.DEVNULL, stderr=open(out / f"pyspy_{pid}.err", "w"),
                ))
        else:
            pass
        rc = bench.wait(timeout=BENCH_TIMEOUT_S)
        for spy in spies:
            spy.wait(timeout=120)
    window_end = time.time()
    cpu_end = thread_cpu(server.pid)
    if kind == "events":
        (out / "events_stop.json").write_text(post(args.port, "/stop_request_profile", {"run_id": f"c{load}"}),
                                              encoding="utf-8")
    else:
        pass
    (out / "thread_cpu.json").write_text(json.dumps({"start": cpu_start, "end": cpu_end}), encoding="utf-8")
    (out / "window.json").write_text(json.dumps({"start": window_start, "end": window_end, "rc": rc}), encoding="utf-8")
    (out / "gpu_after.txt").write_text(gpu_snapshot(), encoding="utf-8")
    with open(Path(args.out) / "progress.txt", "a", encoding="utf-8") as progress:
        progress.write(f"{time.strftime('%H:%M:%S')} c{load} {kind} rc={rc} {window_end - window_start:.1f}s "
                       f"load={os.getloadavg()[0]:.1f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--loads", default="1,16,64")
    parser.add_argument("--passes", default="clean,events,pyspy")
    parser.add_argument("--pyspy-delay", type=float, default=8.0)
    parser.add_argument("--pyspy-seconds", type=float, default=15.0)
    parser.add_argument("--serve-extra", default="")
    parser.add_argument("--probe", action="store_true",
                        help="boot with hop_probe (queue hops, loop lag, GC) writing to <out>/probe/p.<pid>.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.model]
    env = dict(os.environ, PYTHONPATH=str(TREE), SGLANG_OMNI_STRICT_PORT="1")
    if args.probe:
        (out / "probe").mkdir(exist_ok=True)
        env.update(PYTHONPATH=f"{Path(__file__).resolve().parent / 'hop_probe'}:{TREE}",
                   OMNI_HOP_PROBE=str(out / "probe" / "p"))
    else:
        pass
    identity = subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo WORKING_TREE_DIRTY",
        shell=True, cwd=TREE, capture_output=True, text=True,
    ).stdout
    imported = subprocess.run(
        [sys.executable, "-c", "import sglang_omni; print(sglang_omni.__file__)"],
        cwd=TREE, env=env, capture_output=True, text=True,
    ).stdout
    serve = [
        sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", spec["path"],
        *(["--config", spec["config"]] if spec["config"] else []),
        "--host", "127.0.0.1", "--port", str(args.port),
        *shlex.split(spec["placement"].format(card=args.card)), *shlex.split(args.serve_extra),
    ]
    (out / "identity.txt").write_text(
        f"tree {TREE}\n{identity}imports {imported}serve {shlex.join(serve)}\npythonpath {env['PYTHONPATH']}\n"
        f"loadavg {os.getloadavg()}\n",
        encoding="utf-8",
    )
    spawned = time.monotonic()
    with open(out / "serve.log", "w", encoding="utf-8") as serve_log:
        server = subprocess.Popen(serve, cwd=TREE, env=env, stdout=serve_log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    (out / "server.pgid").write_text(f"{server.pid}\n", encoding="utf-8")
    try:
        wait_healthy(args.port, server)
        (out / "startup.txt").write_text(f"{time.monotonic() - spawned:.1f}\n", encoding="utf-8")
        (out / "processes.txt").write_text(process_table(server.pid), encoding="utf-8")
        (out / "gpu_healthy.txt").write_text(gpu_snapshot(), encoding="utf-8")
        first = True
        for load in (int(item) for item in args.loads.split(",")):
            for kind in args.passes.split(","):
                run_pass(args, server, load, kind, warmup=first)
                first = False
        (out / "model_info.json").write_text(
            httpx.get(f"http://127.0.0.1:{args.port}/model_info", timeout=30, trust_env=False).text, encoding="utf-8"
        )
    finally:
        stop_group(server)
        with open(out / "progress.txt", "a", encoding="utf-8") as progress:
            progress.write(f"{time.strftime('%H:%M:%S')} done, group left {group_pids(server.pid)}\n")


if __name__ == "__main__":
    main()
