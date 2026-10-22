"""Item 3's worst step check (ITEM3_RESERVE_DERIVATION_20261011.md, verification): one server of a tree on one card
with the vocoder at ROWS rows and the memory probe, then the cells W<ROWS> (ROWS identical seeded requests at the API's
text limit from make_worst_meta_c20261011.py, so their finals land in one step near the AR's context) and stream c64.
Reports the startup memory lines (the step measurement, the reserve, the pool, each graph capture), every request's
outcome, and per process the peak device use, the memory outside torch and the allocator's out of memory retries and
failures. The server is stopped by its process group.
10-21 copy under /data/c20261021; PAIR_HOME names another run home (10-22, /data/c20261022r).
usage: python3 worst_step_c20261021.py <tree> <card> <cpus> <port> <out> <rows> <worst meta>"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

D = os.environ.get("PAIR_HOME", "/data/c20261021")
MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
COSYVOICE = "/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"
PROBE = str(Path(__file__).resolve().parent / "mem_probe_c20261020")
STARTUP = re.compile(
    r"largest admitted step|prefix cache reserve|Flow prefix cache|graphs captured|prefix compile warmup|"
    r"KV Cache is allocated|avail mem|out of memory|Traceback"
)


def healthy(port: int) -> bool:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status == 200
    except OSError:
        return False


def main() -> None:
    tree, card, cpus, port, out, rows, worst_meta = sys.argv[1:8]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": card,
        "PYTHONPATH": f"{PROBE}:{tree}:{COSYVOICE}",
        "SGLANG_OMNI_STRICT_PORT": "1",
        "TORCHINDUCTOR_CACHE_DIR": f"{D}/inductor_{Path(tree).name}",
        "MEM_PROBE": "1",
        "MEM_PROBE_OUT": str(out / "mem"),
    }
    identity = subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo DIRTY",
        cwd=tree, shell=True, capture_output=True, text=True,
    ).stdout
    (out / "identity.txt").write_text(identity)
    server_args = [] if rows == "16" else ["--vocoder.factory.max_batch_size", rows]
    server = subprocess.Popen(
        ["taskset", "-c", cpus, "python3", "-m", "sglang_omni.cli", "serve", "--model-path", MODEL,
         "--host", "127.0.0.1", "--port", port, *server_args],
        cwd=tree, env=env, stdout=open(out / "serve.log", "w"), stderr=subprocess.STDOUT, start_new_session=True,
    )
    cells = (
        (f"W{rows}", ["--max-concurrency", rows, "--max-samples", rows, "--seed", "1234", "--meta", worst_meta]),
        ("c64", ["--max-concurrency", "64"]),
    )
    results = {}
    try:
        deadline = time.monotonic() + 1800
        while not healthy(int(port)):
            assert server.poll() is None, "server exited before it was healthy"
            assert time.monotonic() < deadline, "server not healthy in 30 minutes"
            time.sleep(5)
        for cell, flags in cells:
            client_env = {**env, "CUDA_VISIBLE_DEVICES": "", "MEM_PROBE": "0"}
            rc = subprocess.run(
                ["taskset", "-c", cpus, "python3", "-u", "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model",
                 MODEL, "--port", port, "--lang", "en", "--warmup", "1", "--use-existing-server", "--generate-only",
                 "--stream", "--output-dir", str(out / cell), *flags],
                cwd=tree, env=client_env, stdout=open(out / f"bench_{cell}.log", "w"), stderr=subprocess.STDOUT,
            ).returncode
            summary = json.loads((out / cell / "speed_results.json").read_text())["summary"]
            results[cell] = {
                "rc": rc,
                **{key: summary.get(key) for key in (
                    "completed_requests", "failed_requests", "throughput_qps", "max_token_hits",
                    "latency_mean_s", "latency_p99_s")},
            }
    finally:
        # note (ratish): a server whose whole group already exited raises here (10-22)
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    print(identity.strip())
    print("startup:")
    for line in open(out / "serve.log", errors="replace"):
        if STARTUP.search(line):
            print("  " + line.strip()[-260:])
    print("cells:")
    for cell, row in results.items():
        print(f"  {cell}: {row}")
    print("memory per process (GiB; peaks over the run):")
    for path in sorted(out.glob("mem.*.jsonl")):
        samples = [json.loads(line) for line in open(path)]
        if not samples:
            continue
        gib = 2**30
        print(
            f"  pid {path.stem.split('.')[-1]}: samples {len(samples)}, used max {max(s['used'] for s in samples) / gib:.2f}, "
            f"free min {min(s['free'] for s in samples) / gib:.2f}, torch reserved max "
            f"{max(s['torch_reserved'] for s in samples) / gib:.2f}, outside torch max "
            f"{max(s['non_torch'] for s in samples) / gib:.2f}, alloc retries {samples[-1]['num_alloc_retries']}, "
            f"ooms {samples[-1]['num_ooms']}"
        )


if __name__ == "__main__":
    main()
