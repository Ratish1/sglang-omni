"""Arms on the H200 box's one card, round by round (two cards: same-time pairs, then swapped).

Run on the box from the box branch's worktree: python3 /workspace/box/tasks/pr2800_review/box/pair_check.py
    --out /data/pr2800/<run> --rounds upn@0,s3z@1;s3z@0,upn@1
Every tree is fetched from its remote by full commit into its own worktree; nothing comes from a copy.
Each server is the cookbook's colocated bf16 H100 command (examples/configs/qwen3_omni_colocated_h100_bf16.yaml,
the TTS CI's profile, its engine settings kept on the H200) on one card, pinned to eight cores of the card's NUMA
node 1 and their siblings. Each arm gets SeedTTS EN streamed
with voice clone (the CI's request shape) through benchmark_omni_seedtts at c16 and c64 on the full corpus, warmup
1, and the seeded first 200 at c1 on its first round. After the rounds, one Qwen3-ASR server scores WER and WavLM
scores SIM for every c16 pass. Playback continuity is rebuilt from each request's chunk gaps and the arm's window
walk. Writes <out>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

REPO = Path("/workspace/sglang-omni")
WORKTREES = REPO / ".tmp/wt"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
CONFIG = "examples/configs/qwen3_omni_colocated_h100_bf16.yaml"
CARD_CPUS = {
    int(card): cpus
    for card, cpus in (
        entry.split("=", 1)
        for entry in os.environ.get("PR2800_CARD_CPUS", "0=56-63,168-175;1=64-71,176-183").split(";")
    )
}
NODE = os.environ.get("PR2800_NODE", "1")
SERVER_PORT = 21000
ASR_PORT = 23000
# tree: (worktree directory, remote, full commit), fetched from the remote by that commit
TREES = {
    "bench2": ("bench_8ec777b33", "origin", "8ec777b330d9b82cdc17721ee33f0e942fedf100"),
    "up2": ("up_6bdd47967", "origin", "6bdd47967caa55c38430d18175b750008aecab61"),
    "upn": ("upn_e7339958b", "upstream", "e7339958ba733af4500571b43beabc0d397e4ae7"),
    "s3z": ("s3z_d0da0626f", "origin", "d0da0626f4feae7ff86ea5c665b21d0585cfc940"),
    "s4z": ("s4z_e3ab1c533", "origin", "e3ab1c53392281dc077e9effab78fb99b3e94313"),
}
# note (ratish): every arm is measured by one client, main's benchmark plus each chunk's audio length, so the
# playback continuity is the client's own whatever the arm's window walk.
BENCH_TREE = "bench2"
ASR_TREE = "up2"
# arm: (tree, extra serve args, leading chunk seconds of the window walk before the 0.8 s windows)
ARMS = {
    "upn": ("upn", [], [0.7769]),
    "s3z": ("s3z", [], [0.7769]),
    "s4z": ("s4z", [], [0.7769]),
}
# arm: benchmark args every pass of the arm adds
ARM_BENCH_ARGS: dict[str, list[str]] = {}
PASSES = (
    ("c16", ["--max-concurrency", "16"]),
    ("c64", ["--max-concurrency", "64"]),
    ("c1_seeded", ["--max-concurrency", "1", "--seed", "1234", "--max-samples", "200"]),
    ("c1", ["--max-concurrency", "1", "--max-samples", "200"]),
    ("c16_seeded", ["--max-concurrency", "16", "--seed", "1234"]),
    ("c1_seeded_full", ["--max-concurrency", "1", "--seed", "1234"]),
    ("c1_greedy_seeded_full", ["--max-concurrency", "1", "--seed", "1234", "--temperature", "0"]),
)
TREE_LOCK = threading.Lock()


def log(out: Path, message: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {message}"
    print(line, flush=True)
    with open(out / "driver.log", "a") as handle:
        handle.write(line + "\n")


def tree(name: str) -> Path:
    directory, remote, commit = TREES[name]
    path = WORKTREES / directory
    with TREE_LOCK:
        if path.exists():
            return path
        else:
            pass
        subprocess.run(["git", "fetch", "-q", "--depth", "1", remote, commit], cwd=REPO, check=True)
        subprocess.run(["git", "worktree", "add", "-q", "--detach", str(path), commit], cwd=REPO, check=True)
        return path


def pinned(card: int, command: list[str]) -> list[str]:
    return ["numactl", f"--physcpubind={CARD_CPUS[card]}", f"--membind={NODE}", *command]


def card_memory_mib(card: int) -> int:
    used = subprocess.run(
        ["nvidia-smi", "-i", str(card), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout.split()
    # note (ratish): nvidia-smi can answer with a sentence while a context tears down; that sample is skipped.
    return int(used[0]) if used and used[0].isdigit() else 0


class MemorySampler:
    """The card's memory every 2 s while a server runs; the peak goes into the summary."""

    def __init__(self, card: int) -> None:
        self.card = card
        self.peak_mib = 0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        while not self.stopped.wait(2.0):
            self.peak_mib = max(self.peak_mib, card_memory_mib(self.card))

    def __enter__(self) -> "MemorySampler":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stopped.set()
        self.thread.join()


def card_env(card: int, tree_path: Path) -> dict[str, str]:
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(card), "PYTHONPATH": str(tree_path)}


def serve(card: int, tree_path: Path, model: str, port: int, args: list[str], log_path: Path) -> subprocess.Popen:
    env = {**card_env(card, tree_path), "SGLANG_OMNI_STRICT_PORT": "1", "SGLANG_OMNI_STARTUP_TIMEOUT": "1800"}
    handle = open(log_path, "w")
    process = subprocess.Popen(
        pinned(card, [sys.executable, "-u", "-m", "sglang_omni.cli", "serve", "--model-path", model, *args,
                      "--host", "127.0.0.1", "--port", str(port)]),
        cwd=tree_path,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_path.with_suffix(".pgid").write_text(str(process.pid))
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited, see {log_path}")
        else:
            pass
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200:
                return process
            else:
                pass
        except httpx.HTTPError:
            pass
        time.sleep(5)
    stop(card, process)
    raise RuntimeError(f"server not healthy in 30 min, see {log_path}")


def stop(card: int, process: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 15)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    for _ in range(60):
        if card_memory_mib(card) < 1024:
            return
        else:
            pass
        time.sleep(2)


def benchmark(card: int, args: list[str], log_path: Path) -> int:
    bench_path = tree(BENCH_TREE)
    with open(log_path, "w") as handle:
        return subprocess.run(
            pinned(card, [sys.executable, "-m", "benchmarks.eval.benchmark_omni_seedtts", *args]),
            cwd=bench_path,
            env=card_env(card, bench_path),
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def walk(audio_s: float, leading: list[float]) -> list[float]:
    durations: list[float] = []
    total = 0.0
    while total < audio_s - 1e-3:
        duration = leading[len(durations)] if len(durations) < len(leading) else 0.8
        durations.append(duration)
        total += duration
    return durations


def max_underrun(gaps: list[float], durations: list[float]) -> float:
    arrivals = [0.0]
    for gap in gaps:
        arrivals.append(arrivals[-1] + gap)
    deadline = arrivals[0] + durations[0]
    worst = 0.0
    for arrival, duration in zip(arrivals[1:], durations[1:]):
        worst = max(worst, arrival - deadline)
        deadline = max(deadline, arrival) + duration
    return worst


def speed_summary(pass_dir: Path, leading: list[float]) -> dict[str, object]:
    results = json.loads((pass_dir / "speed_results.json").read_text())
    summary = results["summary"]
    requests = [r for r in results["per_request"] if r["is_success"]]
    matched, underruns, first_gaps, first_audio = 0, [], [], []
    for request in requests:
        if request.get("audio_ttfp_s") is not None:
            first_audio.append(request["audio_ttfp_s"])
        else:
            pass
        gaps = request["inter_chunk_s"] or []
        durations = walk(request["audio_duration_s"], leading)
        if len(gaps) + 1 != max(1, len(durations)):
            continue
        else:
            matched += 1
        if gaps:
            first_gaps.append(gaps[0])
            underruns.append(max_underrun(gaps, durations))
        else:
            pass
    keys = (
        "completed_requests", "failed_requests", "throughput_qps", "audio_throughput_s_per_s",
        "audio_ttfp_mean_s", "audio_ttfp_median_s", "audio_ttfp_p95_s", "audio_ttfp_p99_s",
        "text_ttft_mean_s", "inter_chunk_mean_s", "inter_chunk_p95_s", "inter_chunk_p99_s",
        "latency_mean_s", "latency_median_s", "latency_p95_s", "latency_p99_s", "rtf_mean",
        "rtf_median", "audio_duration_mean_s",
    )
    out: dict[str, object] = {key: summary.get(key) for key in keys}
    out.update(
        requests=len(requests),
        chunk_count_matched=matched,
        first_gap_mean_s=round(statistics.mean(first_gaps), 4) if first_gaps else None,
        first_gap_p95_s=round(percentile(first_gaps, 95), 4) if first_gaps else None,
        max_underrun_p99_s=round(percentile(underruns, 99), 4) if underruns else None,
        max_underrun_max_s=round(max(underruns), 4) if underruns else None,
        continuity_pct={
            f"c{int(t * 1000)}": round(100 * sum(u <= t for u in underruns) / len(underruns), 2)
            for t in (0.05, 0.1, 0.2)
        }
        if underruns
        else None,
    )
    return out


def server_log_facts(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(errors="replace")
    chunk_gate = [line for line in text.splitlines() if "talker chunk gate" in line]
    return {
        "talker_chunk_gate_last": chunk_gate[-1][-120:] if chunk_gate else None,
        "tracebacks": text.count("Traceback"),
    }


def run_arm(out: Path, round_index: int, card: int, arm: str, with_c1: bool) -> dict[str, object]:
    tree_name, extra, leading = ARMS[arm]
    tree_path = tree(tree_name)
    boot_dir = out / f"r{round_index}_card{card}_{arm}"
    boot_dir.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree_path, capture_output=True, text=True).stdout.strip()
    port = SERVER_PORT + 1000 * round_index + 200 * card
    serve_args = ["--config", CONFIG, "--colocate", *extra]
    (boot_dir / "serve_args.txt").write_text(f"{head}\n{' '.join(serve_args)}\n")
    log(out, f"round {round_index} card {card} {arm} {head[:9]} start, load {open('/proc/loadavg').read().strip()}")
    began = time.monotonic()
    record: dict[str, object] = {"arm": arm, "card": card, "round": round_index, "head": head, "passes": {}}
    with MemorySampler(card) as memory:
        server = serve(card, tree_path, MODEL, port, serve_args, boot_dir / "server.log")
        record["startup_s"] = round(time.monotonic() - began, 1)
        log(out, f"round {round_index} card {card} {arm} healthy after {record['startup_s']} s")
        try:
            for name, args in PASSES:
                if (name == "c1" or name.startswith("c1_")) and not with_c1:
                    continue
                else:
                    pass
                pass_dir = boot_dir / name
                rc = benchmark(
                    card,
                    ["--host", "127.0.0.1", "--port", str(port), "--model", "qwen3-omni", "--lang", "en",
                     "--voice-clone", "--stream", "--warmup", "1", "--generate-only", "--disable-tqdm",
                     "--output-dir", str(pass_dir), *ARM_BENCH_ARGS.get(arm, []), *args],
                    boot_dir / f"{name}.log",
                )
                record["passes"][name] = {"rc": rc, **(speed_summary(pass_dir, leading) if rc == 0 else {})}
                log(out, f"round {round_index} card {card} {arm} {name} rc {rc} {json.dumps(record['passes'][name])[:300]}")
        finally:
            stop(card, server)
    record["peak_card_mib"] = memory.peak_mib
    record.update(server_log_facts(boot_dir / "server.log"))
    log(out, f"round {round_index} card {card} {arm} stopped, peak {memory.peak_mib} MiB")
    return record


def score(out: Path, records: list[dict[str, object]], card: int, pass_names: list[str]) -> None:
    passes = [
        (record, name, out / f"r{record['round']}_card{record['card']}_{record['arm']}" / name)
        for record in records
        for name in pass_names
        if name in record["passes"]
    ]
    if not passes:
        return
    else:
        pass
    asr_tree = tree(ASR_TREE)
    asr = serve(card, asr_tree, ASR_MODEL, ASR_PORT + 100 * card, [], out / "asr_server.log")
    try:
        for record, name, pass_dir in passes:
            rc = benchmark(
                card,
                ["--transcribe-only", "--port", str(ASR_PORT + 100 * card), "--host", "127.0.0.1", "--lang", "en",
                 "--output-dir", str(pass_dir)],
                pass_dir.parent / f"{name}_wer.log",
            )
            log(out, f"wer {pass_dir.parent.name} {name} rc {rc}")
    finally:
        stop(card, asr)
    for record, name, pass_dir in passes:
        rc = benchmark(
            card,
            ["--similarity-only", "--lang", "en", "--device", "cuda:0", "--output-dir", str(pass_dir)],
            pass_dir.parent / f"{name}_sim.log",
        )
        log(out, f"sim {pass_dir.parent.name} {name} rc {rc}")
    for record, name, pass_dir in passes:
        quality: dict[str, object] = {}
        wer_path = pass_dir / "wer_results.json"
        if wer_path.exists():
            wer = json.loads(wer_path.read_text())["summary"]
            quality.update(
                wer_corpus=wer.get("wer_corpus"),
                wer_below_50_corpus=wer.get("wer_below_50_corpus"),
                n_above_50_pct_wer=wer.get("n_above_50_pct_wer"),
            )
        else:
            pass
        sim_path = pass_dir / "similarity_results.json"
        if sim_path.exists():
            quality["sim"] = json.loads(sim_path.read_text())["summary"].get("speaker_similarity_mean")
        else:
            pass
        record["passes"][name]["quality"] = quality
        if name.startswith("c1_seeded"):
            pass
        else:
            for wav in pass_dir.rglob("*.wav"):
                wav.unlink()


def main() -> None:
    global PASSES, BENCH_TREE
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", help="rounds separated by ';', each 'arm@card,arm@card'")
    parser.add_argument("--no-score", action="store_true", help="speed only: delete the audio, score nothing")
    parser.add_argument("--score-only", action="store_true", help="score the c16 passes of <out>/summary.json")
    parser.add_argument("--score-card", type=int, default=0, help="the card the ASR server and SIM run on")
    parser.add_argument("--score-passes", default="c16", help="passes WER and SIM score, comma separated")
    parser.add_argument("--passes", default=",".join(name for name, _ in PASSES), help="passes to run, comma separated")
    parser.add_argument("--c1-every-round", action="store_true", help="run the seeded c1 pass in every round")
    parser.add_argument("--card-cpus", action="append", default=[], help="'card=cpulist', replaces a card's cores")
    parser.add_argument("--bench-tree", default=BENCH_TREE, help="the tree whose benchmark client every arm runs")
    args = parser.parse_args()
    PASSES = tuple(entry for entry in PASSES if entry[0] in args.passes.split(","))
    BENCH_TREE = args.bench_tree
    for entry in args.card_cpus:
        card, cpus = entry.split("=", 1)
        CARD_CPUS[int(card)] = cpus
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        records = json.loads((out / "summary.json").read_text())
        score(out, records, args.score_card, args.score_passes.split(","))
        (out / "summary.json").write_text(json.dumps(records, indent=1))
        log(out, "scored")
        return
    elif args.rounds is None:
        parser.error("--rounds is required unless --score-only")
    else:
        pass
    (out / "driver.pgid").write_text(str(os.getpgrp()))
    tree(BENCH_TREE)
    seen_arms: set[str] = set()
    records: list[dict[str, object]] = []
    for round_index, round_spec in enumerate(args.rounds.split(";"), start=1):
        placements = [(arm, int(card)) for arm, card in (entry.split("@") for entry in round_spec.split(","))]
        with ThreadPoolExecutor(max_workers=len(placements)) as pool:
            futures = {
                arm: pool.submit(run_arm, out, round_index, card, arm, args.c1_every_round or arm not in seen_arms)
                for arm, card in placements
            }
        for arm, card in placements:
            try:
                records.append(futures[arm].result())
            except Exception as error:  # noqa: BLE001 a failed boot is recorded and the run goes on
                records.append({"arm": arm, "card": card, "round": round_index, "error": repr(error), "passes": {}})
                log(out, f"round {round_index} card {card} {arm} failed: {error!r}")
            seen_arms.add(arm)
        (out / "summary.json").write_text(json.dumps(records, indent=1))
    if args.no_score:
        for wav in out.rglob("*.wav"):
            wav.unlink()
    else:
        score(out, records, args.score_card, args.score_passes.split(","))
    (out / "summary.json").write_text(json.dumps(records, indent=1))
    log(out, "done")


if __name__ == "__main__":
    main()
