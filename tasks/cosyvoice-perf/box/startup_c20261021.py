"""Startup and memory of two sglang-omni trees, Fun-CosyVoice3, booted at the same time on cards 0 and 1, then swapped:
the seconds from launch to a healthy /health, the card's used memory at health and after a buffered b16 and a streaming
c16 load of 64 requests each, and the AR's KV pool and the Flow prefix pool lines of each server log. For #2944's
skipped padded DiT startup (PR2260_REVIEW.md 5): its compile and graphs only cost startup, and the vocoder builds
before the engine, so any memory it frees could reach the KV pool.
usage: python3 startup_c20261021.py <first tree> <second tree> <out dir> <boots per order>"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

D = "/data/c20261021"
MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
COSYVOICE = "/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"
PORTS = (27000, 27400)
CARDS = ("0", "1")
CPUS = ("0-15,64-79", "16-31,80-95")
POOL_LINES = re.compile(r"KV Cache is allocated.*|Flow prefix cache: .*")


def environment(tree, card):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": card, "PYTHONPATH": f"{tree}:{COSYVOICE}",
            "SGLANG_OMNI_STRICT_PORT": "1", "TORCHINDUCTOR_CACHE_DIR": f"{D}/inductor_{Path(tree).name}"}


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            return response.status == 200
    except OSError:
        return False


def used_mib(card):
    output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", card],
                            capture_output=True, text=True).stdout
    return int(output.strip())


def load(tree, gpu, port, out, name, is_stream):
    return subprocess.Popen(
        ["taskset", "-c", CPUS[gpu], "python3", "-u", "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL,
         "--port", str(port), "--lang", "en", "--warmup", "1", "--use-existing-server", "--generate-only",
         *(["--stream"] if is_stream else []), "--max-concurrency", "16", "--max-samples", "64",
         "--output-dir", str(out / name)],
        cwd=tree, env=environment(tree, ""), stdout=open(out / f"{name}.log", "w"), stderr=subprocess.STDOUT)


def boot(trees, out):
    servers, launched = [], time.monotonic()
    for gpu, (tree, port) in enumerate(zip(trees, PORTS)):
        (out / f"gpu{gpu}").mkdir(parents=True, exist_ok=True)
        servers.append(subprocess.Popen(
            ["taskset", "-c", CPUS[gpu], "python3", "-m", "sglang_omni.cli", "serve", "--model-path", MODEL,
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=tree, env=environment(tree, CARDS[gpu]), stdout=open(out / f"gpu{gpu}" / "serve.log", "w"),
            stderr=subprocess.STDOUT, start_new_session=True))
    ready = [None, None]
    try:
        while None in ready:
            for gpu, port in enumerate(PORTS):
                assert servers[gpu].poll() is None, f"server {gpu} exited before it was healthy"
                if ready[gpu] is None and healthy(port):
                    ready[gpu] = (time.monotonic() - launched, used_mib(CARDS[gpu]))
            assert time.monotonic() - launched < 1800, "servers not healthy in 30 minutes"
            time.sleep(1)
        for name, is_stream in (("b16", False), ("c16", True)):
            clients = [load(tree, gpu, port, out / f"gpu{gpu}", name, is_stream)
                       for gpu, (tree, port) in enumerate(zip(trees, PORTS))]
            assert all(client.wait() == 0 for client in clients), f"a {name} load failed"
        loaded = [used_mib(card) for card in CARDS]
    finally:
        for server in servers:
            os.killpg(server.pid, signal.SIGTERM)
        for server in servers:
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
    return [
        {"ready_s": round(ready[gpu][0], 1), "ready_mib": ready[gpu][1], "loaded_mib": loaded[gpu],
         "pools": POOL_LINES.findall((out / f"gpu{gpu}" / "serve.log").read_text())}
        for gpu in range(2)
    ]


def main():
    first, second, out, boots = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
    names = (Path(first).name, Path(second).name)
    rows = []
    for index in range(boots):
        for order, trees in (("a", (first, second)), ("b", (second, first))):
            arms = boot(trees, out / f"{order}{index}")
            for tree, arm in zip(trees, arms):
                rows.append({"boot": f"{order}{index}", "tree": Path(tree).name, "card": CARDS[trees.index(tree)], **arm})
    (out / "startup.json").write_text(json.dumps(rows, indent=1))
    print(f"| boot | tree | card | ready s | MiB at ready | MiB after load |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print(f"| {row['boot']} | {row['tree']} | {row['card']} | {row['ready_s']} | {row['ready_mib']} | {row['loaded_mib']} |")
    for name in names:
        arm_rows = [row for row in rows if row["tree"] == name]
        print(f"{name}: ready s mean {sum(r['ready_s'] for r in arm_rows) / len(arm_rows):.1f}, "
              f"MiB at ready mean {sum(r['ready_mib'] for r in arm_rows) / len(arm_rows):.0f}, "
              f"after load mean {sum(r['loaded_mib'] for r in arm_rows) / len(arm_rows):.0f}")
    for row in rows:
        print(f"pools {row['boot']} {row['tree']}: {' ; '.join(row['pools'])[:300]}")


main()
