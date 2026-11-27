"""Stop a Qwen3-Omni server under video load the way Ctrl-C does, and time how long its processes take to exit.

Boots the server as baseline.py does, keeps --videos Video-MME CI requests in flight with videomme_speed.py, and --after
seconds later sends SIGINT to the server's process group. Records the seconds until the group is empty, every
"Terminating stuck process" line the launcher logs (a stage process its join timed out on), and a py-spy dump of each
process still alive --dump-after seconds after the signal. A process alive --deadline seconds after the signal is
killed by the group's pgid. Writes <out>/stop.json and prints one summary line.

usage: python3 stop_under_load.py --tree <tree> --model omni_colocated --card 0 --port <port> --out <dir> \
           --serve-extra "<flags>" --videos 16 --after 40
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import baseline

KIT = Path(__file__).resolve().parent


def video_client(port: int, videos: int, out: Path) -> list[str]:
    return [
        sys.executable,
        str(KIT / "videomme_speed.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--repo-id",
        "zhaochenyang20/Video_MME_ci",
        "--video-fps",
        "2",
        "--video-max-frames",
        "128",
        "--video-max-pixels",
        "401408",
        "--max-tokens",
        "256",
        "--timeout-s",
        "500",
        "--warmup",
        "0",
        "--max-concurrency",
        str(videos),
        "--output-dir",
        str(out),
        "--disable-tqdm",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--model", required=True, choices=sorted(baseline.MODELS))
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--serve-extra", default="")
    parser.add_argument("--videos", type=int, default=16)
    parser.add_argument("--after", type=float, default=40.0)
    parser.add_argument("--dump-after", type=float, default=20.0)
    parser.add_argument("--deadline", type=float, default=120.0)
    args = parser.parse_args()
    tree = Path(args.tree)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = baseline.MODELS[args.model]
    env = dict(os.environ, PYTHONPATH=str(tree), SGLANG_OMNI_STRICT_PORT="1")
    serve = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        spec["path"],
        *(["--config", spec["config"]] if spec["config"] else []),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        *shlex.split(spec["placement"].format(card=args.card)),
        *shlex.split(args.serve_extra),
    ]
    identity = subprocess.run(
        "git rev-parse HEAD HEAD:sglang_omni; git diff --quiet HEAD -- sglang_omni || echo WORKING_TREE_DIRTY",
        shell=True,
        cwd=tree,
        capture_output=True,
        text=True,
    ).stdout
    (out / "identity.txt").write_text(
        f"tree {tree}\n{identity}serve {shlex.join(serve)}\n", encoding="utf-8"
    )
    with open(out / "serve.log", "w", encoding="utf-8") as serve_log:
        server = subprocess.Popen(
            serve,
            cwd=tree,
            env=env,
            stdout=serve_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (out / "server.pgid").write_text(f"{server.pid}\n", encoding="utf-8")
    client = None
    dumped: list[int] = []
    left: list[int] = []
    exit_s = None
    try:
        baseline.wait_healthy(args.port, server)
        with open(out / "client.log", "w", encoding="utf-8") as client_log:
            client = subprocess.Popen(
                video_client(args.port, args.videos, out / "video"),
                cwd=tree,
                env=dict(os.environ, PYTHONPATH=str(tree)),
                stdout=client_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        time.sleep(args.after)
        (out / "processes.txt").write_text(
            baseline.process_table(server.pid), encoding="utf-8"
        )
        sent = time.monotonic()
        os.killpg(server.pid, signal.SIGINT)
        while time.monotonic() < sent + args.deadline:
            left = baseline.group_pids(server.pid)
            if not left:
                exit_s = time.monotonic() - sent
                break
            else:
                pass
            if not dumped and time.monotonic() > sent + args.dump_after:
                dumped = left
                with open(out / "pyspy_left.txt", "w", encoding="utf-8") as dumps:
                    for pid in left:
                        dumps.write(f"== pid {pid}\n")
                        dumps.flush()
                        subprocess.run(
                            ["py-spy", "dump", "--pid", str(pid)],
                            stdout=dumps,
                            stderr=subprocess.STDOUT,
                            timeout=60,
                            check=False,
                        )
            else:
                pass
            time.sleep(0.2)
    finally:
        if baseline.group_pids(server.pid):
            os.killpg(server.pid, signal.SIGKILL)
        else:
            pass
        if client is not None and client.poll() is None:
            os.killpg(client.pid, signal.SIGKILL)
        else:
            pass
    stuck = [
        line.strip()
        for line in (out / "serve.log").read_text(errors="replace").splitlines()
        if "Terminating stuck process" in line
    ]
    result = {
        "exit_s": exit_s,
        "left_at_deadline": left if exit_s is None else [],
        "alive_at_dump": dumped,
        "stuck_lines": stuck,
    }
    (out / "stop.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"stop {out.name}: exit {'none' if exit_s is None else f'{exit_s:.1f} s'}, "
        f"alive at {args.dump_after:.0f} s {len(dumped)}, stuck {stuck}",
        flush=True,
    )


if __name__ == "__main__":
    main()
