"""Arm one torch profiler window on a server that is already under load.

The load is somebody else's process (the seed-tts benchmark at a fixed concurrency); this
only opens a window of --steps scheduler forwards on the stage, waits for the finished
trace, and stops the profile. The window closes by itself after --steps forwards, so
under load the stop is a no-op; without load it exports what was captured.

usage: python live_window.py --url http://127.0.0.1:8000 --out DIR --label live_c16
         --steps 600 [--with-stack]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

TRACE_WAIT_S = 900.0


def post(url: str, route: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"{url}{route}",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--step-stage", default="tts_engine")
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args()
    trace_dir = Path(args.out).resolve() / args.label
    trace_dir.mkdir(parents=True, exist_ok=False)
    run_id = f"{args.label}-{int(time.time())}"
    armed_at = time.time()
    started = post(
        args.url,
        "/start_profile",
        {
            "run_id": run_id,
            "trace_path_template": str(trace_dir / "trace"),
            "num_steps": args.steps,
            "step_stage": args.step_stage,
            "with_stack": args.with_stack,
            "record_shapes": False,
        },
    )
    deadline = time.monotonic() + TRACE_WAIT_S
    while True:
        traces = sorted(trace_dir.glob("trace*.trace.json.gz"))
        if traces and not list(trace_dir.glob("trace*.trace.json")):
            break
        if time.monotonic() > deadline:
            post(args.url, "/stop_profile", {"run_id": run_id})
            raise TimeoutError(f"no finished trace under {trace_dir}")
        time.sleep(1.0)
    post(args.url, "/stop_profile", {"run_id": run_id})
    record = {
        "run_id": run_id,
        "steps": args.steps,
        "with_stack": args.with_stack,
        "armed_at_unix": armed_at,
        "trace_ready_at_unix": time.time(),
        "start_profile": started,
        "traces": [str(path) for path in traces],
    }
    (trace_dir / "window.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
