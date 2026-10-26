"""One arm on one card: boot a tree, run the SeedTTS bench once, record the card's memory, stop. For a lease with a
single card, where pair.py's two arms at once do not fit; two arms' seeded runs compare with --compare.

usage: python3 serve_arm.py --label c1s_pr --tree /workspace/pr_52983c59d --card 0 --port 4400 --concurrency 1
       [--seed 1234] [--no-stream] [--meta LIST]
       python3 serve_arm.py --compare c1s_pr c1s_inline

The server takes its card from the stage gpu flags and binds 127.0.0.1 with SGLANG_OMNI_STRICT_PORT; the bench is
pair.py's (full corpus or --meta, warmup 1, streaming unless --no-stream). nvidia-smi samples the card's used memory
every 100 ms from health to the bench's end; memory.txt has the card at health, its peak during the bench and the
peak's rise over health. Results under $ARM_ROOT/<label> (default /data/inline_20261026/arms).
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3_20261020"))

from pair import BENCH_TIMEOUT_S, MODEL, audio_hashes, stop_group, wait_healthy  # noqa: E402

ROOT = Path(os.environ.get("ARM_ROOT", "/data/inline_20261026/arms"))
SAMPLE_INTERVAL_MS = 100
FAILURE_PATTERN = re.compile(r"CUBLAS_STATUS_ALLOC_FAILED|OutOfMemoryError|out of memory|Traceback")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label")
    parser.add_argument("--tree")
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--port", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--meta", default="")
    parser.add_argument("--compare", nargs=2)
    args = parser.parse_args()
    if args.compare:
        first, second = (audio_hashes(ROOT / label) for label in args.compare)
        same = sum(first[name] == second.get(name) for name in first)
        print(f"byte identical {same} of {len(first)} ({len(second)} in the second)", flush=True)
        return
    else:
        pass

    out = ROOT / args.label
    out.mkdir(parents=True, exist_ok=True)
    identity = subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo WORKING_TREE_DIRTY",
        shell=True, cwd=args.tree, capture_output=True, text=True,
    ).stdout
    server_env = dict(os.environ, PYTHONPATH=args.tree, SGLANG_OMNI_STRICT_PORT="1")
    imported = subprocess.run(
        [sys.executable, "-c", "import sglang_omni; print(sglang_omni.__file__)"],
        cwd=args.tree, env=server_env, capture_output=True, text=True,
    ).stdout
    (out / "tree.txt").write_text(f"{args.tree} card {args.card}\n{identity}imports {imported}", encoding="utf-8")
    command = [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, "--host", "127.0.0.1",
               "--port", str(args.port), "--tts_engine.gpu", str(args.card), "--vocoder.gpu", str(args.card)]
    with open(out / "serve.log", "w", encoding="utf-8") as serve_log:
        spawned = time.monotonic()
        server = subprocess.Popen(
            command, cwd=args.tree, env=server_env, stdout=serve_log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    sampler = None
    bench_rc = None
    try:
        wait_healthy(args.port, server)
        (out / "startup.txt").write_text(f"{time.monotonic() - spawned:.1f}\n", encoding="utf-8")
        with open(out / "memory_trace.csv", "w", encoding="utf-8") as trace:
            sampler = subprocess.Popen(
                ["nvidia-smi", "-i", str(args.card), "--query-gpu=timestamp,memory.used", "--format=csv,noheader,nounits",
                 "-lms", str(SAMPLE_INTERVAL_MS)],
                stdout=trace, stderr=subprocess.STDOUT,
            )
        bench = [sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL,
                 "--use-existing-server", "--host", "127.0.0.1", "--port", str(args.port), "--generate-only",
                 *([] if args.no_stream else ["--stream"]), "--lang", "en", "--warmup", "1",
                 "--max-concurrency", str(args.concurrency), "--output-dir", str(out / "bench")]
        if args.seed is not None:
            bench += ["--seed", str(args.seed)]
        else:
            pass
        if args.meta:
            bench += ["--meta", args.meta]
        else:
            pass
        with open(out / "bench.log", "w", encoding="utf-8") as bench_log:
            bench_rc = subprocess.run(
                bench, cwd=args.tree, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=args.tree),
                stdout=bench_log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S, check=False,
            ).returncode
    finally:
        if sampler is not None:
            sampler.terminate()
            sampler.wait()
        else:
            pass
        stop_group(server, args.port)

    used_mib = [int(line.rsplit(",", 1)[1]) for line in (out / "memory_trace.csv").read_text().splitlines()
                if line.rsplit(",", 1)[-1].strip().isdigit()]
    memory = (
        f"card MiB at health {used_mib[0]}, peak {max(used_mib)}, rise {max(used_mib) - used_mib[0]}, "
        f"samples {len(used_mib)}"
    )
    (out / "memory.txt").write_text(memory + "\n", encoding="utf-8")
    outcome = re.findall(r"Completed requests: +\d+|Failed requests: +\d+", (out / "bench.log").read_text(errors="replace"))
    failures = len(FAILURE_PATTERN.findall((out / "serve.log").read_text(errors="replace")))
    print(
        f"{args.label} bench rc {bench_rc} {', '.join(outcome[-2:]) if outcome else 'no outcome line'}; serve log failure lines "
        f"{failures}; {memory}; {identity.split()[1] if len(identity.split()) > 1 else identity!r}",
        flush=True,
    )


if __name__ == "__main__":
    main()
