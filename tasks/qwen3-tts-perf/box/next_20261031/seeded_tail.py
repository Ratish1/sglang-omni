"""Seeded SeedTTS runs on warm server pairs, one scoring pass per card, then the speaker-loss tail table.

usage: python3 seeded_tail.py --main /workspace/n31_main --branch /workspace/n31_prio_<sha> --root /data/next_20261031/tail
       python3 seeded_tail.py ... --table-only

Block 34b (10-30) left one question on the talker priority branch: at non-streaming c64 it had 11 requests with SIM
below 40 over two unseeded runs against main's 0, while every other load was even. Each request here carries the run's
seed (the bench sends one seed on every request, the server derives each request's draws from it), so a prompt draws
the same random numbers in both arms and the arms differ only by batch numerics. Phases, each a server pair booted
once with the default layout on SeedTTS EN 1.7B Base:
  o1: main card 0, branch card 1: c1 non-streaming and streaming at seed 1234 (byte identity), then c64 non-streaming
      seeds 1 to 4;
  o2: branch card 0, main card 1: c64 non-streaming seeds 5 to 8;
  band: main on both cards: c64 non-streaming seeds 1 to 4.
Then each card's c64 runs are scored under one ASR server (WER), then speaker similarity on that card, and the table
counts SIM below 40 and 50 per arm, pairs SIM per request and lists every sub-40 request with its WER and its SIM in
the other runs.
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3_20261020"))

from pair import (  # noqa: E402
    ASR_CONCURRENCY,
    ASR_MODEL,
    BENCH_TIMEOUT_S,
    MODEL,
    audio_hashes,
    stop_group,
    wait_healthy,
)

C64_SEEDS = {"o1": (1, 2, 3, 4), "o2": (5, 6, 7, 8), "band": (1, 2, 3, 4)}


def identity(tree: str) -> str:
    return subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni | tr '\\n' ' '; git diff --quiet HEAD -- sglang_omni || echo DIRTY",
        shell=True, cwd=tree, capture_output=True, text=True,
    ).stdout.strip()


def start(tree: str, card: int, port: int, log: Path) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1")
    command = [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, "--host", "127.0.0.1",
               "--port", str(port), "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card)]
    with open(log, "w", encoding="utf-8") as handle:
        return subprocess.Popen(command, cwd=tree, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                start_new_session=True)


def bench(tree: str, port: int, out: Path, concurrency: int, seed: int | None, stream: bool) -> str:
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL, "--use-existing-server",
               "--host", "127.0.0.1", "--port", str(port), "--generate-only", *(["--stream"] if stream else []),
               "--lang", "en", "--warmup", "1", "--max-concurrency", str(concurrency),
               *([] if seed is None else ["--seed", str(seed)]), "--output-dir", str(out / "bench")]
    with open(out / "bench.log", "w", encoding="utf-8") as log:
        subprocess.run(command, cwd=tree, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=tree),
                       stdout=log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S, check=False)
    lines = (out / "bench.log").read_text(errors="replace").splitlines()
    wanted = ("Completed requests", "Failed requests", "RTF mean", "Throughput (req/s)", "TTFC mean")
    return "; ".join(" ".join(line.split()) for line in lines if line.strip().startswith(wanted))


def serve_phase(phase: str, arms: list[tuple[str, str, int, int]], runs: list[tuple[str, int, int, bool]],
                root: Path) -> None:
    servers = []
    try:
        for label, tree, card, port in arms:
            print(f"{phase} {label}: card {card} port {port} tree {tree} {identity(tree)}", flush=True)
            servers.append((start(tree, card, port, root / f"serve_{phase}_{label}.log"), port))
        for server, port in servers:
            wait_healthy(port, server)
        print(f"{phase}: both healthy", flush=True)
        for run, concurrency, seed, stream in runs:
            with concurrent.futures.ThreadPoolExecutor(len(arms)) as pool:
                results = list(pool.map(
                    lambda arm: bench(arm[1], arm[3], root / run / arm[0], concurrency, seed, stream), arms))
            print(f"{phase} {run}: " + " | ".join(f"{arm[0]} {result}" for arm, result in zip(arms, results)),
                  flush=True)
    finally:
        for server, port in servers:
            stop_group(server, port)
    for label, _, _, _ in arms:
        text = (root / f"serve_{phase}_{label}.log").read_text(errors="replace")
        errors = sum(text.count(word) for word in ("Traceback", "CUDA error", "device-side assert", "out of memory"))
        print(f"{phase} {label}: serve log error lines {errors}", flush=True)


def score_card(card: int, port: int, tree: str, outs: list[Path], root: Path) -> None:
    env = dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1")
    scorer = [sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts", "--model", MODEL, "--lang", "en",
              "--skip-gpu-cleanup"]
    with open(root / f"asr_card{card}.log", "w", encoding="utf-8") as log:
        asr = subprocess.Popen(
            [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", ASR_MODEL, "--host", "127.0.0.1",
             "--port", str(port), "--asr.gpu", str(card)],
            cwd=tree, env=dict(env, MASTER_PORT=str(port + 50)), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        wait_healthy(port, asr)
        for out in outs:
            with open(out / "score--transcribe-only.log", "w", encoding="utf-8") as log:
                subprocess.run([*scorer, "--output-dir", str(out / "bench"), "--transcribe-only",
                                "--use-existing-server", "--host", "127.0.0.1", "--port", str(port),
                                "--asr-concurrency", str(ASR_CONCURRENCY)],
                               cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S,
                               check=False)
    finally:
        stop_group(asr, port)
    for out in outs:
        with open(out / "score--similarity-only.log", "w", encoding="utf-8") as log:
            subprocess.run([*scorer, "--output-dir", str(out / "bench"), "--similarity-only", "--device",
                            f"cuda:{card}"],
                           cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_S,
                           check=False)
        print(f"scored {out.parent.name}/{out.name} on card {card}", flush=True)


def per_sample(out: Path, name: str, field: str) -> dict[str, float]:
    rows = json.loads((out / "bench" / name).read_text())["per_sample"]
    return {row["id"]: float(row[field]) for row in rows if row.get("is_success", True) and row.get(field) is not None}


def table(root: Path) -> None:
    for mode in ("c1n", "c1s"):
        first, second = audio_hashes(root / mode / "main"), audio_hashes(root / mode / "prio")
        same = sum(first[name] == second.get(name) for name in first)
        print(f"{mode} seed 1234: byte identical {same} of {len(first)} ({len(second)} in the branch)", flush=True)
    sims = {}
    wers = {}
    for out in sorted(root.glob("c64n_s*/*")):
        key = f"{out.parent.name}/{out.name}"
        sims[key] = per_sample(out, "similarity_results.json", "speaker_similarity")
        wers[key] = per_sample(out, "wer_results.json", "wer")
    comparisons = {
        "branch against main": [(f"c64n_s{seed}/main", f"c64n_s{seed}/prio") for seed in range(1, 9)],
        "main against main (band)": [(f"c64n_s{seed}/maina", f"c64n_s{seed}/mainb") for seed in range(1, 5)],
    }
    for title, pairs in comparisons.items():
        deltas = []
        counts = {"first": [0, 0, 0], "second": [0, 0, 0]}
        for first_key, second_key in pairs:
            first, second = sims[first_key], sims[second_key]
            common = sorted(set(first) & set(second))
            deltas += [second[k] - first[k] for k in common]
            for side, values in (("first", first), ("second", second)):
                counts[side][0] += sum(values[k] < 40 for k in common)
                counts[side][1] += sum(values[k] < 50 for k in common)
                counts[side][2] += len(common)
            print(f"  {second_key} minus {first_key}: n {len(common)}, below 40 {sum(first[k] < 40 for k in common)} / "
                  f"{sum(second[k] < 40 for k in common)}, mean SIM {statistics.mean(first[k] for k in common):.3f} / "
                  f"{statistics.mean(second[k] for k in common):.3f}", flush=True)
        half_width = 1.96 * statistics.pstdev(deltas) / len(deltas) ** 0.5
        print(f"{title}: requests {counts['first'][2]}; below 40 {counts['first'][0]} / {counts['second'][0]}; below 50 "
              f"{counts['first'][1]} / {counts['second'][1]}; paired SIM {statistics.mean(deltas):+.3f} +- {half_width:.3f}; "
              f"more than 15 under the other arm {sum(d > 15 for d in deltas)} / {sum(d < -15 for d in deltas)}",
              flush=True)
    for key, values in sims.items():
        for prompt, value in sorted(values.items()):
            if value < 40:
                others = sorted((round(sims[other].get(prompt, float("nan")), 1), other) for other in sims if other != key)
                print(f"sub-40 {key} {prompt} SIM {value:.1f} WER {wers[key].get(prompt)}; lowest elsewhere {others[:3]}",
                      flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--table-only", action="store_true")
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if not args.table_only:
        c64 = lambda phase: [(f"c64n_s{seed}", 64, seed, False) for seed in C64_SEEDS[phase]]
        serve_phase("o1", [("main", args.main, 0, 20000), ("prio", args.branch, 1, 20200)],
                    [("c1n", 1, 1234, False), ("c1s", 1, 1234, True), *c64("o1")], root)
        serve_phase("o2", [("prio", args.branch, 0, 21000), ("main", args.main, 1, 21200)], c64("o2"), root)
        serve_phase("band", [("maina", args.main, 0, 22000), ("mainb", args.main, 1, 22200)], c64("band"), root)
        by_card = {0: [], 1: []}
        for out in sorted(root.glob("c64n_s*/*")):
            seed = int(out.parent.name.split("_s")[1])
            served_on_card_0 = (out.name in ("main", "maina") and seed <= 4) or (out.name == "prio" and seed > 4)
            by_card[0 if served_on_card_0 else 1].append(out)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda card: score_card(card, 23000 + 200 * card, args.main, by_card[card], root), (0, 1)))
    else:
        pass
    table(root)


if __name__ == "__main__":
    main()
