"""A server whose engine queue is unbounded has no in-flight cap to size the codec arena from.

Boots TREE on one card with --tts_engine.engine.max_queued_requests none twice: without a slot count it must
exit at startup naming the missing cap; with --vocoder.factory.codec_state_slots SLOTS it must serve a stream
and log that streams past those slots take the left-context path. Also reads each boot's KV pool line.

usage: python3 null_cap_check.py --tree TREE --card N --out DIR [--slots 128]
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from seedtts_pair import card_cpus
from slot_wait_serving import MODELS, request_body, run_request, wait_healthy

EXIT_TIMEOUT_S = 900
REFUSAL_TEXT = "admits without one"
WARNING_TEXT = "admission has no in-flight cap"
KV_POOL_TEXT = "KV pool holds"


def boot(tree: Path, card: int, port: int, log: Path, extra: list[str]) -> subprocess.Popen:
    command = [
        "taskset", "-c", card_cpus(card), sys.executable, "-m", "sglang_omni.cli", "serve",
        "--model-path", MODELS["customvoice"], "--host", "127.0.0.1", "--port", str(port),
        "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card),
        "--tts_engine.engine.max_queued_requests", "none", *extra,
    ]
    with log.open("w") as handle:
        return subprocess.Popen(
            command, cwd=tree, env={**os.environ, "PYTHONPATH": str(tree), "SGLANG_OMNI_STRICT_PORT": "1"},
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )


def stop(server: subprocess.Popen) -> None:
    if server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    else:
        pass


def kv_pool_lines(log: Path) -> list[str]:
    return [line.strip() for line in log.read_text(errors="replace").splitlines() if KV_POOL_TEXT in line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--port", type=int, default=29000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}

    refused_log = args.out / "no_slots_serve.log"
    server = boot(args.tree, args.card, args.port, refused_log, [])
    try:
        server.wait(timeout=EXIT_TIMEOUT_S)
        text = refused_log.read_text(errors="replace")
        report["no_slots"] = {"exit_code": server.returncode, "names_the_cap": REFUSAL_TEXT in text}
    except subprocess.TimeoutExpired:
        report["no_slots"] = {"exit_code": None, "names_the_cap": False}
    finally:
        stop(server)

    served_log = args.out / "slots_serve.log"
    server = boot(args.tree, args.card, args.port + 200, served_log, ["--vocoder.factory.codec_state_slots", str(args.slots)])
    url = f"http://127.0.0.1:{args.port + 200}"
    try:
        wait_healthy(url, server)
        result = run_request(url, request_body("customvoice", "stream", 0, args.tree), "stream")
        text = served_log.read_text(errors="replace")
        report["slots"] = {
            "outcome": result["outcome"], "samples": result["samples"], "warns": WARNING_TEXT in text,
            "kv_pool": kv_pool_lines(served_log),
        }
    finally:
        stop(server)

    is_ok = (
        report["no_slots"]["exit_code"] not in (None, 0)
        and report["no_slots"]["names_the_cap"]
        and report["slots"]["outcome"] == "complete"
        and report["slots"]["samples"] > 0
        and report["slots"]["warns"]
    )
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{time.strftime('%T')} null_cap_check {'PASS' if is_ok else 'FAIL'} {json.dumps(report)}", flush=True)


if __name__ == "__main__":
    main()
