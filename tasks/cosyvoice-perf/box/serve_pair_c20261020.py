"""Same time serving pair of two sglang-omni trees, Fun-CosyVoice3, one server per GPU, the cookbook's SeedTTS
benchmark against both at once. Order a puts the first tree in slot 0, order b swaps the slots. Every metric row of
both arms is printed. Slot s is card CARDS[s] on cores CPUS[s] (server and client pinned), each tree's identity and
Inductor cache its own.
10-18 copy of serve_pair_c20261017.py under /data/c20261018: adds the buffered cells b1 (256 samples), b16 and b64,
the same requests without --stream, whose audio comes from the vocoder's whole utterance path (mel2wav_batch), and a
latency median row. Later on 10-18: the open loop cells r12 and r16 (Poisson arrivals at 12 and 16 requests a second
from one arrival seed, so both arms and both orders get one schedule; 64 slots, never the bound).
10-20 copy under /data/c20261020 (this lease's paths and Inductor caches), with the open loop cell r20 (20 requests a
second, near the c64 cell's throughput, where a faster vocoder cannot refill the AR's queue as a closed loop does).
Later on 10-20: the long form cell L16 (the 48 entries of make_longform_meta.py at concurrency 16, item 3's cell), and
c1f, every reference seeded at concurrency 1, whose tails rest on 1,088 requests where c1's p99 is one of 64.
usage: python3 serve_pair_c20261020.py <first tree> <second tree> <out dir>"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

D = "/data/c20261020"
MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
COSYVOICE = "/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS"
PORTS = (24000, 25000)
CARDS = ("2", "3")
CPUS = ("0-15,64-79", "16-31,80-95")
ALL_CELLS = {
    "c1": (["--max-concurrency", "1", "--max-samples", "64", "--seed", "1234"], True),
    "c1u": (["--max-concurrency", "1", "--max-samples", "256"], True),
    "c1f": (["--max-concurrency", "1", "--seed", "1234"], True),
    "c16": (["--max-concurrency", "16"], True),
    "c64": (["--max-concurrency", "64"], True),
    "b1": (["--max-concurrency", "1", "--max-samples", "256"], False),
    "b16": (["--max-concurrency", "16"], False),
    "b64": (["--max-concurrency", "64"], False),
    "r12": (["--max-concurrency", "64", "--request-rate", "12", "--arrival-seed", "2910"], True),
    "r16": (["--max-concurrency", "64", "--request-rate", "16", "--arrival-seed", "2910"], True),
    "r20": (["--max-concurrency", "64", "--request-rate", "20", "--arrival-seed", "2910"], True),
    "L16": (["--max-concurrency", "16", "--meta", f"{D}/longform/meta.lst"], True),
}
CELLS = tuple((name, *ALL_CELLS[name]) for name in os.environ.get("PAIR_CELLS", "c1,c16").split(","))
ROWS = (
    ("throughput_qps", "req/s"),
    ("rtf_mean", "RTF"),
    ("audio_ttfp_mean_s", "first audio mean"),
    ("audio_ttfp_p95_s", "first audio p95"),
    ("audio_ttfp_p99_s", "first audio p99"),
    ("inter_chunk_mean_s", "inter chunk mean"),
    ("inter_chunk_p95_s", "inter chunk p95"),
    ("inter_chunk_p99_s", "inter chunk p99"),
    ("latency_mean_s", "latency mean"),
    ("latency_median_s", "latency median"),
    ("latency_p95_s", "latency p95"),
    ("latency_p99_s", "latency p99"),
    ("completed_requests", "completed"),
    ("failed_requests", "failed"),
    ("max_token_hits", "at the token cap"),
)


def environment(tree, card):
    return {**os.environ, "CUDA_VISIBLE_DEVICES": card, "PYTHONPATH": f"{tree}:{COSYVOICE}",
            "SGLANG_OMNI_STRICT_PORT": "1", "TORCHINDUCTOR_CACHE_DIR": f"{D}/inductor_{Path(tree).name}"}


def identity(tree):
    lines = [subprocess.run(command, cwd=tree, capture_output=True, text=True, shell=True).stdout.strip() for command in (
        "git rev-parse HEAD HEAD:sglang_omni",
        "git diff --quiet HEAD -- sglang_omni || echo WORKING_TREE_DIRTY",
        f"PYTHONPATH={tree} python3 -c 'import sglang_omni; print(sglang_omni.__file__)'",
    )]
    return "\n".join(line for line in lines if line)


def healthy(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).status == 200
    except OSError:
        return False


def run_order(trees, out):
    out.mkdir(parents=True, exist_ok=True)
    servers = []
    for gpu, (tree, port) in enumerate(zip(trees, PORTS)):
        (out / f"identity_gpu{gpu}.txt").write_text(identity(tree) + "\n")
        log = open(out / f"serve_gpu{gpu}.log", "w")
        servers.append(subprocess.Popen(
            ["taskset", "-c", CPUS[gpu], "python3", "-m", "sglang_omni.cli", "serve", "--model-path", MODEL,
             "--host", "127.0.0.1", "--port", str(port)],
            cwd=tree, env=environment(tree, CARDS[gpu]), stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    try:
        deadline = time.monotonic() + 1800
        while not all(healthy(port) for port in PORTS):
            assert all(server.poll() is None for server in servers), "a server exited before it was healthy"
            assert time.monotonic() < deadline, "servers not healthy in 30 minutes"
            time.sleep(5)
        for cell, flags, is_stream in CELLS:
            clients = [
                subprocess.Popen(
                    ["taskset", "-c", CPUS[gpu], "python3", "-u", "-m", "benchmarks.eval.benchmark_tts_seedtts",
                     "--model", MODEL, "--port", str(port),
                     "--lang", "en", "--warmup", "1", "--use-existing-server", "--generate-only",
                     *(["--stream"] if is_stream else []),
                     "--output-dir", str(out / f"gpu{gpu}" / cell), *flags],
                    cwd=tree, env=environment(tree, ""),
                    stdout=open(out / f"bench_gpu{gpu}_{cell}.log", "w"), stderr=subprocess.STDOUT)
                for gpu, (tree, port) in enumerate(zip(trees, PORTS))
            ]
            assert all(client.wait() == 0 for client in clients), f"a {cell} client failed"
    finally:
        for server in servers:
            os.killpg(server.pid, signal.SIGTERM)
        for server in servers:
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)


def summary(path):
    return json.loads(path.read_text())["summary"]


def report(out, names):
    print(f"| order | cell | metric | {names[0]} | {names[1]} | change |")
    print("| --- | --- | --- | --- | --- | --- |")
    for order, gpus in (("a", (0, 1)), ("b", (1, 0))):
        for cell, _, _ in CELLS:
            first = summary(out / order / f"gpu{gpus[0]}" / cell / "speed_results.json")
            second = summary(out / order / f"gpu{gpus[1]}" / cell / "speed_results.json")
            for key, label in ROWS:
                a, b = first.get(key), second.get(key)
                change = f"{100 * (b - a) / a:+.1f} %" if isinstance(a, float) and a else ""
                print(f"| {order} | {cell} | {label} | {a} | {b} | {change} |")


if __name__ == "__main__":
    first_tree, second_tree, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    run_order((first_tree, second_tree), out_dir / "a")
    run_order((second_tree, first_tree), out_dir / "b")
    report(out_dir, (Path(first_tree).name, Path(second_tree).name))
