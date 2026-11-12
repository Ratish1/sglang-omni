"""Second queue of review cells for #3477: two trees at the same time, one server per card, SeedTTS EN 1.7B Base, warmup 1.

review_cells.py's mix16_48 cell restarted its non-streaming client between passes, and for a quarter to a third of the
stream run no non-streaming request was in flight. Here the streaming client of a mixed cell sends only the first 256
prompts, so it ends inside the non-streaming client's one pass over all 1,088. Phases:
  o1m      main card 0, head card 1: the mixed cell three times, streaming at 64 three times, then n16 and n64
  o2m      head card 0, main card 1: the same
  bandm    main on both cards: the same, the same-time band
  nograph  main card 0, head card 1, both with an incremental graph headroom no card has, so no codec graph captures
  nograph2 the same, cards swapped
  slots    head with 16 codec state slots card 0, head with its 64 card 1
  slots2   the same, cards swapped
  o1x      main card 0, head card 1: streaming at 16, non-streaming at 64 twice, then one at a time
  o2x      head card 0, main card 1: the same
  s16o1    main card 0, head card 1: streaming at 16 three times
  s16o2    head card 0, main card 1: the same
  s16band  main on both cards: the same, the same-time band
Cells: mixa to mixc 16 streaming and 48 non-streaming in flight at once; n16 and n64 non-streaming over every prompt.

usage: python3 review_cells_v2.py --main TREE --head TREE --root DIR --phases o1m,o2m,bandm
       --card-cpus "0=0-15,64-79;1=16-31,80-95" --client-cpus 32-63,96-127 [--port 23000]
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3_20261020"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from pair import BENCH_TIMEOUT_S, MODEL, stop_group, wait_healthy  # noqa: E402
from seeded_tail import identity  # noqa: E402

STREAM_PROMPTS = 256
NO_GRAPH = ["--vocoder.factory.incremental_codec_cuda_graph_min_free_gb", "4096"]
SLOTS_16 = ["--vocoder.factory.codec_state_slots", "16"]
MIXED = [
    (f"mix{letter}", [("stream", 16, True, STREAM_PROMPTS), ("whole", 48, False, None)])
    for letter in "abc"
]
STREAM_64 = [(f"s64{letter}", [("stream", 64, True, None)]) for letter in "abc"]
WHOLE = [("n16", [("whole", 16, False, None)]), ("n64", [("whole", 64, False, None)])]
WHOLE_TWICE = WHOLE + [("n64b", [("whole", 64, False, None)])]
STREAM_16 = [(f"s16{letter}", [("stream", 16, True, None)]) for letter in "abc"]
STREAM_16_WHOLE_1 = [
    ("s16", [("stream", 16, True, None)]),
    ("n64", [("whole", 64, False, None)]),
    ("n64b", [("whole", 64, False, None)]),
    ("n1", [("whole", 1, False, None)]),
]
PHASES = {
    "o1m": (
        (("main", "main", 0, []), ("head", "head", 1, [])),
        MIXED + STREAM_64 + WHOLE,
    ),
    "o2m": (
        (("head", "head", 0, []), ("main", "main", 1, [])),
        MIXED + STREAM_64 + WHOLE,
    ),
    "bandm": ((("maina", "main", 0, []), ("mainb", "main", 1, [])), MIXED),
    "nograph": ((("main", "main", 0, NO_GRAPH), ("head", "head", 1, NO_GRAPH)), WHOLE),
    "nograph2": ((("head", "head", 0, NO_GRAPH), ("main", "main", 1, NO_GRAPH)), WHOLE),
    "slots": (
        (("head16", "head", 0, SLOTS_16), ("head64", "head", 1, [])),
        WHOLE_TWICE,
    ),
    "slots2": (
        (("head64", "head", 0, []), ("head16", "head", 1, SLOTS_16)),
        WHOLE_TWICE,
    ),
    "o1x": ((("main", "main", 0, []), ("head", "head", 1, [])), STREAM_16_WHOLE_1),
    "o2x": ((("head", "head", 0, []), ("main", "main", 1, [])), STREAM_16_WHOLE_1),
    "s16o1": ((("main", "main", 0, []), ("head", "head", 1, [])), STREAM_16),
    "s16o2": ((("head", "head", 0, []), ("main", "main", 1, [])), STREAM_16),
    "s16band": ((("maina", "main", 0, []), ("mainb", "main", 1, [])), STREAM_16),
}


def start(
    tree: str, card: int, port: int, cpus: str, extra: list[str], log: Path
) -> subprocess.Popen:
    command = [
        "taskset",
        "-c",
        cpus,
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tts_engine.gpu",
        str(card),
        "--vocoder.gpu",
        str(card),
        *extra,
    ]
    with open(log, "w", encoding="utf-8") as handle:
        return subprocess.Popen(
            command,
            cwd=tree,
            env=dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1"),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def bench(
    tree: str,
    port: int,
    out: Path,
    concurrency: int,
    stream: bool,
    prompts: int | None,
    client_cpus: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    command = [
        "taskset",
        "-c",
        client_cpus,
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--model",
        MODEL,
        "--use-existing-server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--generate-only",
        *(["--stream"] if stream else []),
        "--lang",
        "en",
        "--warmup",
        "1",
        "--max-concurrency",
        str(concurrency),
        *([] if prompts is None else ["--max-samples", str(prompts)]),
        "--output-dir",
        str(out / "bench"),
    ]
    with open(out / "bench.log", "w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=tree,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=tree),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=BENCH_TIMEOUT_S,
            check=False,
        )
    subprocess.run(["rm", "-rf", str(out / "bench" / "audio")], check=False)


def line(out: Path) -> str:
    path = out / "bench" / "speed_results.json"
    if not path.exists():
        return "no speed_results.json"
    else:
        pass
    summary = json.loads(path.read_text())["summary"]
    wanted = (
        "completed_requests",
        "total_requests",
        "throughput_qps",
        "rtf_mean",
        "latency_mean_s",
        "latency_median_s",
        "latency_p95_s",
        "audio_ttfp_mean_s",
        "audio_ttfp_median_s",
        "audio_ttfp_p95_s",
        "audio_ttfp_p99_s",
    )
    return " ".join(
        (
            f"{key}={summary[key]:.4f}"
            if isinstance(summary[key], float)
            else f"{key}={summary[key]}"
        )
        for key in wanted
        if key in summary and summary[key] is not None
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phases", required=True)
    parser.add_argument("--card-cpus", required=True)
    parser.add_argument("--client-cpus", required=True)
    parser.add_argument("--port", type=int, default=23000)
    args = parser.parse_args()
    trees = {"main": args.main, "head": args.head}
    card_cpus = {
        int(card): cpus
        for card, cpus in (item.split("=") for item in args.card_cpus.split(";"))
    }
    port = args.port
    for phase in args.phases.split(","):
        arms, cells = PHASES[phase]
        root = args.root / phase
        root.mkdir(parents=True, exist_ok=True)
        placed = [
            (label, trees[tree], card, port + 200 * index, extra)
            for index, (label, tree, card, extra) in enumerate(arms)
        ]
        servers = []
        try:
            for label, tree, card, arm_port, extra in placed:
                print(
                    f"{phase} {label}: card {card} cpus {card_cpus[card]} port {arm_port} extra {extra} {identity(tree)}",
                    flush=True,
                )
                servers.append(
                    (
                        start(
                            tree,
                            card,
                            arm_port,
                            card_cpus[card],
                            extra,
                            root / f"serve_{label}.log",
                        ),
                        arm_port,
                    )
                )
            for server, arm_port in servers:
                wait_healthy(arm_port, server)
            print(f"{phase}: both healthy", flush=True)
            for cell, jobs in cells:
                runs = [
                    (
                        tree,
                        arm_port,
                        root / cell / label / name,
                        concurrency,
                        stream,
                        prompts,
                        args.client_cpus,
                    )
                    for label, tree, _, arm_port, _ in placed
                    for name, concurrency, stream, prompts in jobs
                ]
                with concurrent.futures.ThreadPoolExecutor(len(runs)) as pool:
                    list(pool.map(lambda run: bench(*run), runs))
                for label, _, card, _, _ in placed:
                    for name, _, _, _ in jobs:
                        print(
                            f"{phase} {cell} {label} card {card} {name}: {line(root / cell / label / name)}",
                            flush=True,
                        )
        finally:
            for server, arm_port in servers:
                stop_group(server, arm_port)
        for label, _, _, _, _ in placed:
            log = (root / f"serve_{label}.log").read_text(errors="replace")
            print(
                f"{phase} {label}: tracebacks {log.count('Traceback')}, left context {log.count('left-context decoder')}, "
                f"capture failed {log.count('capture_failed')}, whole request failed {log.count('whole request')}",
                flush=True,
            )
        port += 400
    print("review cells v2 done", flush=True)


if __name__ == "__main__":
    main()
