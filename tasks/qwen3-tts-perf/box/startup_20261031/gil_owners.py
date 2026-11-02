"""Who holds the interpreter lock at concurrency 64: one server on one card, SeedTTS EN 1.7B Base, every prompt, py-spy on the
server and its stage processes while the lock is held, thread names kept, for a window in the steady part of the run. Then
per thread its share of the lock samples, and for the largest threads the functions with the most inclusive and leaf
samples. Shares only: a py-spy record gives ownership, not held time (block 82's census holds that).

usage: python3 gil_owners.py --tree TREE --root DIR --card 0 --port 25000 [--no-stream] [--warmup-s 6] [--duration-s 10]
"""

import argparse
import concurrent.futures
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from seeded_tail import bench, identity, start, stop_group, wait_healthy  # noqa: E402

PROC = Path("/proc")
RATE_HZ = 200
FRAME = re.compile(r"^(.*?) \(([^()]*?)(?::\d+)?\)$")
THREAD_DIGITS = re.compile(r"\d+")


def session_pythons(server_pid: int) -> list[int]:
    session = int((PROC / str(server_pid) / "stat").read_text().rsplit(")", 1)[1].split()[3])
    pids = []
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        else:
            pass
        try:
            same_session = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[3]) == session
            command = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if same_session and (b"spawn_main" in command or b"sglang_omni.cli" in command):
            pids.append(int(entry.name))
        else:
            pass
    return pids


def short_frame(frame: str) -> str:
    match = FRAME.match(frame.strip())
    if match is None:
        return frame.strip()[:110]
    else:
        pass
    path = match.group(2)
    for marker in ("site-packages/", "/workspace/", "lib/python3.12/", "python/sglang/"):
        if marker in path:
            path = path.split(marker, 1)[1]
            break
        else:
            pass
    return f"{match.group(1)} ({path})"


def reduce_records(records: list[Path]) -> None:
    threads: Counter[str] = Counter()
    inclusive: dict[str, Counter[str]] = defaultdict(Counter)
    leaf: dict[str, Counter[str]] = defaultdict(Counter)
    other_process_samples = 0
    for record in records:
        text = record.read_text(errors="replace")
        is_stage_process = "scheduler-tts_engine" in text
        for line in text.splitlines():
            stack, _, count_text = line.rpartition(" ")
            if not stack or not count_text.isdigit():
                continue
            else:
                pass
            count = int(count_text)
            if not is_stage_process:
                other_process_samples += count
                continue
            else:
                pass
            frames = stack.split(";")
            thread = THREAD_DIGITS.sub("N", frames[0].split("): ", 1)[-1]) if frames[0].startswith("thread ") else "unnamed"
            body = [frame for frame in frames if not frame.startswith(("thread ", "process "))]
            threads[thread] += count
            for name in {short_frame(frame) for frame in body}:
                inclusive[thread][name] += count
            if body:
                leaf[thread][short_frame(body[-1])] += count
            else:
                pass
    total = sum(threads.values())
    print(f"stage process lock samples {total}; other processes, on their own locks, {other_process_samples}")
    for thread, count in threads.most_common(12):
        print(f"{thread}: {100 * count / total:.1f} %")
    for thread, count in threads.most_common(6):
        print(f"== {thread}: {count} samples")
        print("  inclusive:")
        for name, samples in inclusive[thread].most_common(70):
            print(f"  {100 * samples / count:6.1f} %  {name[:130]}")
        print("  leaf:")
        for name, samples in leaf[thread].most_common(40):
            print(f"  {100 * samples / count:6.1f} %  {name[:130]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--port", type=int, default=25000)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--warmup-s", type=float, default=6.0)
    parser.add_argument("--duration-s", type=int, default=10)
    parser.add_argument("--reduce-glob", default=None, help="reduce existing gil_*.txt records under this glob, no run")
    args = parser.parse_args()
    if args.reduce_glob:
        reduce_records(sorted(Path("/").glob(args.reduce_glob.lstrip("/"))))
        return
    else:
        pass
    args.root.mkdir(parents=True, exist_ok=True)
    print(f"tree {args.tree} {identity(args.tree)} card {args.card}", flush=True)
    server = start(args.tree, args.card, args.port, args.root / "serve.log")
    records: list[Path] = []
    try:
        wait_healthy(args.port, server)
        pids = session_pythons(server.pid)
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            running = pool.submit(bench, args.tree, args.port, args.root / "run", 64, None, not args.no_stream)
            time.sleep(args.warmup_s)
            recorders = []
            for pid in pids:
                record = args.root / f"gil_{pid}.txt"
                records.append(record)
                recorders.append(
                    subprocess.Popen(
                        ["py-spy", "record", "--pid", str(pid), "--rate", str(RATE_HZ), "--duration", str(args.duration_s),
                         "--gil", "--nonblocking", "--threads", "--format", "raw", "--output", str(record)],
                        stdout=subprocess.DEVNULL, stderr=open(args.root / f"gil_{pid}.log", "w"),
                    )
                )
            for recorder in recorders:
                recorder.wait()
            print(running.result(), flush=True)
    finally:
        stop_group(server, args.port)
    reduce_records([record for record in records if record.exists()])


if __name__ == "__main__":
    main()
