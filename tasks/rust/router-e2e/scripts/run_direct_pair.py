#!/usr/bin/env python3
"""Run one benchmark command concurrently against both model workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from common import benchmark_process_env, substitute_placeholders


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker-port", action="append", type=int, required=True)
    parser.add_argument("benchmark", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.benchmark and args.benchmark[0] == "--":
        args.benchmark = args.benchmark[1:]
    if len(args.worker_port) != 2:
        parser.error("exactly two --worker-port values are required")
    if not args.benchmark:
        parser.error("benchmark command is required after --")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    commands: list[list[str]] = []
    logs = []
    processes: list[subprocess.Popen[bytes]] = []
    started = time.monotonic()
    try:
        for index, port in enumerate(args.worker_port, start=1):
            output = args.output_dir / f"worker-{index}"
            output.mkdir()
            command = substitute_placeholders(
                args.benchmark,
                {
                    "router_port": str(port),
                    "router_url": f"http://127.0.0.1:{port}",
                    "output_dir": str(output),
                },
            )
            commands.append(command)
            log = (output / "benchmark.log").open("wb")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=benchmark_process_env(),
                )
            )
        return_codes = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()
    metadata = {
        "commands": commands,
        "worker_ports": args.worker_port,
        "wall_s": time.monotonic() - started,
        "return_codes": return_codes,
    }
    (args.output_dir / "direct-pair.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if any(return_codes):
        raise RuntimeError(f"direct worker benchmark failed: {return_codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
