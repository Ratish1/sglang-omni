"""Each vocoder mode on a real server, uncapped where the mode allows it: whether it starts, what its startup says about
Codec slots, and whether a stream completes.

Modes on 1.7B CustomVoice, one card, one after another, each serving mode sent one stream and one whole request:
- async_uncapped: the default asynchronous decode with max_queued_requests none and no slot count; must refuse to start
  naming the missing cap.
- explicit_none_slots: codec_state_slots none on a capped server; must refuse to start naming the missing count.
- fractional_slots and zero_slots: codec_state_slots 0.5 or 0; must refuse to start.
- synchronous_uncapped: async_decode false, uncapped; must serve with no arena.
- rollback_uncapped: enable_stateful_codec_decoder false, uncapped, on the default asynchronous workers; must serve
  with no arena.
- capped: the defaults; must serve with one arena of the in-flight cap that streams and whole requests share.

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
# (mode, extra serve arguments, whether it must serve, the startup refusal it must log when it must not)
MODES = (
    ("async_uncapped", UNCAPPED, False, "no Codec state slot count"),
    ("explicit_none_slots", ["--vocoder.factory.codec_state_slots", "none"], False, "no Codec state slot count"),
    ("fractional_slots", ["--vocoder.factory.codec_state_slots", "0.5"], False, "must be a positive integer"),
    ("zero_slots", ["--vocoder.factory.codec_state_slots", "0"], False, "must be a positive integer"),
    ("synchronous_uncapped", [*UNCAPPED, "--vocoder.factory.async_decode", "false"], True, None),
    ("rollback_uncapped", [*UNCAPPED, "--vocoder.factory.enable_stateful_codec_decoder", "false"], True, None),
    ("capped", [], True, None),
)
ARENA_LINE = "incremental Codec state: "
SYNCHRONOUS_LINE = "decodes synchronously, each stream on its own Codec state"
ROLLBACK_LINE = "stateful Codec decoder off"


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
    return [line.split(ARENA_LINE, 1)[1][:160] for line in text.splitlines() if ARENA_LINE in line and " MiB a slot" in line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29400)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for index, (mode, extra, should_serve, refusal) in enumerate(MODES):
        log = args.out / f"{mode}_serve.log"
        port = args.port + 200 * index
        url = f"http://127.0.0.1:{port}"
        server = boot(args.tree, args.card, port, log, extra)
        results = {}
        try:
            if should_serve:
                wait_healthy(url, server)
                for kind in ("stream", "whole"):
                    results[kind] = run_request(url, request_body("customvoice", kind, 0, args.tree), kind)
            else:
                server.wait(timeout=EXIT_TIMEOUT_S)
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
            results["error"] = {"outcome": "error", "error": str(error)}
        finally:
            stop(server)
        text = log.read_text(errors="replace")
        report[mode] = {
            "exit_code": server.returncode if not should_serve else None,
            "logged_refusal": refusal is not None and refusal in text,
            "arena": arena_lines(text),
            "says_synchronous": SYNCHRONOUS_LINE in text,
            "says_rollback": ROLLBACK_LINE in text,
            "requests": {
                kind: {key: result.get(key) for key in ("outcome", "samples", "error")} for kind, result in results.items()
            },
        }

    def served_both(mode: str) -> bool:
        requests = report[mode]["requests"]
        return all(
            requests.get(kind, {}).get("outcome") == "complete" and (requests.get(kind, {}).get("samples") or 0) > 0
            for kind in ("stream", "whole")
        )

    checks = {
        mode: report[mode]["exit_code"] not in (None, 0) and report[mode]["logged_refusal"] and not report[mode]["arena"]
        for mode, _, should_serve, _ in MODES
        if not should_serve
    }
    checks["synchronous_uncapped"] = (
        report["synchronous_uncapped"]["says_synchronous"] and not report["synchronous_uncapped"]["arena"]
        and served_both("synchronous_uncapped")
    )
    checks["rollback_uncapped"] = (
        report["rollback_uncapped"]["says_rollback"] and not report["rollback_uncapped"]["arena"]
        and served_both("rollback_uncapped")
    )
    checks["capped"] = (
        any(line.startswith("128 slots, 2.74 MiB a slot, 350.5 MiB total") for line in report["capped"]["arena"])
        and served_both("capped")
    )
    report["checks"] = checks
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{time.strftime('%T')} arena_modes_check {'PASS' if all(checks.values()) else 'FAIL'} {json.dumps(checks)}", flush=True)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
