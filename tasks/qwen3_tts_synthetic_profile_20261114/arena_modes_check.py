"""Each vocoder mode on a real server, uncapped where the mode allows it: whether it starts, what its startup says about
Codec slots, and whether a stream completes.

Modes on 1.7B CustomVoice, one card, one after another:
- async_uncapped: the default asynchronous decode with max_queued_requests none and no slot count; must refuse to start
  naming the missing cap.
- synchronous_uncapped: async_decode false, uncapped; must serve with no arena.
- rollback_uncapped: enable_stateful_codec_decoder false, uncapped; must serve with no arena.
- capped: the defaults; must serve with an arena of the in-flight cap.

usage: python3 arena_modes_check.py --tree TREE --card N --out DIR
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
UNCAPPED = ["--tts_engine.engine.max_queued_requests", "none"]
# (mode, extra serve arguments, whether it must serve)
MODES = (
    ("async_uncapped", UNCAPPED, False),
    ("synchronous_uncapped", [*UNCAPPED, "--vocoder.factory.async_decode", "false"], True),
    ("rollback_uncapped", [*UNCAPPED, "--vocoder.factory.enable_stateful_codec_decoder", "false"], True),
    ("capped", [], True),
)
ARENA_LINE = "incremental Codec state: "
SYNCHRONOUS_LINE = "decodes synchronously, each stream on its own Codec state"
ROLLBACK_LINE = "stateful Codec decoder off"
REFUSAL_LINE = "admits without one"


def boot(tree: Path, card: int, port: int, log: Path, extra: list[str]) -> subprocess.Popen:
    command = [
        "taskset", "-c", card_cpus(card), sys.executable, "-m", "sglang_omni.cli", "serve",
        "--model-path", MODELS["customvoice"], "--host", "127.0.0.1", "--port", str(port),
        "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card), *extra,
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


def arena_lines(text: str) -> list[str]:
    return [line.split(ARENA_LINE, 1)[1][:160] for line in text.splitlines() if ARENA_LINE in line and "stream slots" in line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29400)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for index, (mode, extra, should_serve) in enumerate(MODES):
        log = args.out / f"{mode}_serve.log"
        port = args.port + 200 * index
        server = boot(args.tree, args.card, port, log, extra)
        result = None
        try:
            if should_serve:
                wait_healthy(f"http://127.0.0.1:{port}", server)
                result = run_request(f"http://127.0.0.1:{port}", request_body("customvoice", "stream", 0, args.tree), "stream")
            else:
                server.wait(timeout=EXIT_TIMEOUT_S)
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
            result = {"error": str(error)}
        finally:
            stop(server)
        text = log.read_text(errors="replace")
        report[mode] = {
            "exit_code": server.returncode if not should_serve else None,
            "refused_naming_the_cap": REFUSAL_LINE in text,
            "arena": arena_lines(text),
            "says_synchronous": SYNCHRONOUS_LINE in text,
            "says_rollback": ROLLBACK_LINE in text,
            "stream": {key: result.get(key) for key in ("outcome", "samples", "error")} if result else None,
        }
    checks = {
        "async_uncapped": report["async_uncapped"]["refused_naming_the_cap"] and not report["async_uncapped"]["arena"],
        "synchronous_uncapped": report["synchronous_uncapped"]["says_synchronous"]
        and not report["synchronous_uncapped"]["arena"]
        and (report["synchronous_uncapped"]["stream"] or {}).get("outcome") == "complete",
        "rollback_uncapped": report["rollback_uncapped"]["says_rollback"]
        and not report["rollback_uncapped"]["arena"]
        and (report["rollback_uncapped"]["stream"] or {}).get("outcome") == "complete",
        "capped": any(line.startswith("128 stream slots") for line in report["capped"]["arena"])
        and (report["capped"]["stream"] or {}).get("outcome") == "complete",
    }
    report["checks"] = checks
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{time.strftime('%T')} arena_modes_check {'PASS' if all(checks.values()) else 'FAIL'} {json.dumps(checks)}", flush=True)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
