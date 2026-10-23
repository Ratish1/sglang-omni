"""py-spy on both arms' stage processes (preprocessing, talker and vocoder share one) while a pair.py run serves: waits
for the arms' gpu_healthy.txt, which pair.py writes once both servers answer and before the benchmark starts; finds each
arm's server by its --port (arm a the pair's port, arm b the port plus 200); then records the server and its spawned
stage processes in that server's session without pausing them (--nonblocking) at --rate Hz for --duration seconds, idle frames included, as raw
collapsed stacks in PAIR_ROOT/<label>/<arm>/pyspy_<pid>.txt. Start it alongside pair.py.

usage: PAIR_ROOT=<pairs> python3 pyspy_sidecar.py LABEL --port PORT [--rate 100] [--duration 60] [--timeout 900]
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

PROC = Path("/proc")


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


def session_pythons(session: int) -> list[int]:
    """The server and its spawned stage processes in its session (not its compile workers or resource tracker): Python
    thread names never reach /proc, so each is recorded and the stacks under the vocoder's initial worker pick out the
    stage process."""
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
        pids = [] if session is None else session_pythons(session)
        if not pids:
            print(f"{args.label} {arm}: no Python process for port {port} (session {session})", flush=True)
            continue
        else:
            pass
        for pid in pids:
            recorders.append(
                subprocess.Popen(
                    ["py-spy", "record", "--pid", str(pid), "--rate", str(args.rate), "--duration", str(args.duration),
                     "--nonblocking", "--idle", "--threads", "--format", "raw",
                     "--output", str(root / arm / f"pyspy_{pid}.txt")],
                    stdout=subprocess.DEVNULL, stderr=open(root / arm / f"pyspy_{pid}.log", "w"),
                )
            )
        print(f"{args.label} {arm}: recording {pids} (session {session}) {time.strftime('%H:%M:%S')}", flush=True)
    for recorder in recorders:
        recorder.wait()


if __name__ == "__main__":
    main()
