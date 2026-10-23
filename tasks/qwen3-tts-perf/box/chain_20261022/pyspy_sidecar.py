"""py-spy on both arms' stage processes (preprocessing, talker and vocoder share one) while a pair.py run serves: waits
for the arms' gpu_healthy.txt, which pair.py writes once both servers answer and before the benchmark starts; finds each
arm's server by its --port (arm a the pair's port, arm b the port plus 200) and, in that server's session, the process
running the vocoder's initial worker thread (its name truncated to qwen3-tts-vocod); then records it without pausing
it (--nonblocking) at --rate Hz for --duration seconds, idle frames included, as raw collapsed stacks in
PAIR_ROOT/<label>/<arm>/pyspy.txt. Start it alongside pair.py.

usage: PAIR_ROOT=<pairs> python3 pyspy_sidecar.py LABEL --port PORT [--rate 100] [--duration 60] [--timeout 900]
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

PROC = Path("/proc")
THREAD_NAME = "qwen3-tts-vocod"


def server_session(port: int) -> int | None:
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        else:
            pass
        try:
            arguments = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"serve" in arguments and b"--port" in arguments and str(port).encode() in arguments:
            return int((entry / "stat").read_text().rsplit(")", 1)[1].split()[3])
        else:
            pass
    return None


def stage_process(session: int) -> int | None:
    for entry in PROC.iterdir():
        if not entry.name.isdigit():
            continue
        else:
            pass
        try:
            if int((entry / "stat").read_text().rsplit(")", 1)[1].split()[3]) != session:
                continue
            else:
                pass
            names = [(task / "comm").read_text().strip() for task in (entry / "task").iterdir()]
        except OSError:
            continue
        if THREAD_NAME in names:
            return int(entry.name)
        else:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rate", type=int, default=100)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    root = Path(os.environ["PAIR_ROOT"]) / args.label
    deadline = time.time() + args.timeout
    while not all((root / arm / "gpu_healthy.txt").exists() for arm in ("a", "b")):
        if time.time() > deadline:
            print(f"{args.label}: servers not healthy within {args.timeout} s", flush=True)
            return
        else:
            time.sleep(0.2)
    recorders = []
    for arm, port in (("a", args.port), ("b", args.port + 200)):
        session = server_session(port)
        pid = None if session is None else stage_process(session)
        if pid is None:
            print(f"{args.label} {arm}: no stage process for port {port} (session {session})", flush=True)
            continue
        else:
            pass
        recorders.append(
            subprocess.Popen(
                ["py-spy", "record", "--pid", str(pid), "--rate", str(args.rate), "--duration", str(args.duration),
                 "--nonblocking", "--idle", "--threads", "--format", "raw", "--output", str(root / arm / "pyspy.txt")],
                stdout=subprocess.DEVNULL, stderr=open(root / arm / "pyspy.log", "w"),
            )
        )
        print(f"{args.label} {arm}: recording stage process {pid} (session {session}) {time.strftime('%H:%M:%S')}",
              flush=True)
    for recorder in recorders:
        recorder.wait()


if __name__ == "__main__":
    main()
