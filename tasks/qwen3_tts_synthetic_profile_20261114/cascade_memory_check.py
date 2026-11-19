"""The streaming cascade's Qwen3-TTS launches on main and on a branch: whether each starts, its Codec arena, its KV
pool, and whether a stream and a whole request complete.

The cascade profiles serve Qwen3-TTS Base with examples/configs and a small --mem-fraction-static (1.7B at 0.12, 0.6B
at 0.08), so an arena that grows comes out of a small KV budget. Each arm boots alone on the card, one after another,
the arm order swapped between the two launches.

usage: python3 cascade_memory_check.py --tree main=/data/trees/main --tree pr=/data/trees/head6 --card 0 --out DIR
"""

import argparse
import base64
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from seedtts_pair import card_cpus
from slot_wait_serving import INPUTS, REFERENCE_TEXT, REFERENCE_WAV, run_request, wait_healthy

LAUNCHES = (
    ("base_1_7b_fraction_0_12", "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "examples/configs/qwen3_tts_1_7b.yaml", "0.12"),
    ("base_0_6b_fraction_0_08", "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "examples/configs/qwen3_tts_0_6b.yaml", "0.08"),
)
KV_PATTERN = re.compile(r"KV Cache is allocated\..*?#tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB")
BUDGET_PATTERN = re.compile(r"mem_fraction_static=[\d.]+ total=[^\n]{0,200}")
ARENA_PATTERN = re.compile(r"incremental Codec state: ([^\n]{0,160})")
RUNNING_PATTERN = re.compile(r"max_running_requests=(\d+)")


def boot(tree: Path, card: int, port: int, log: Path, model: str, config: str, fraction: str) -> subprocess.Popen:
    command = [
        "taskset", "-c", card_cpus(card), sys.executable, "-m", "sglang_omni.cli", "serve",
        "--model-path", model, "--config", config, "--host", "127.0.0.1", "--port", str(port),
        "--mem-fraction-static", fraction,
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


def body(model: str, kind: str, tree: Path) -> dict:
    clip = base64.b64encode((tree / REFERENCE_WAV).read_bytes()).decode()
    return {
        "model": model,
        "input": INPUTS[0],
        "language": "English",
        "response_format": "pcm",
        "stream": kind != "whole",
        "references": [{"audio_path": f"data:audio/wav;base64,{clip}", "text": REFERENCE_TEXT}],
    }


def read_log(text: str) -> dict:
    kv = KV_PATTERN.findall(text)
    return {
        "kv_tokens": [int(tokens) for tokens, _, _ in kv],
        "kv_gb": [round(float(k) + float(v), 3) for _, k, v in kv],
        "budget": BUDGET_PATTERN.findall(text)[-2:],
        "arena": ARENA_PATTERN.findall(text)[-1:],
        "max_running_requests": sorted(set(RUNNING_PATTERN.findall(text))),
        "has_traceback": "Traceback (most recent call last)" in text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", action="append", required=True, help="arm=path")
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29600)
    args = parser.parse_args()
    arms = [(arm, Path(path)) for arm, path in (item.split("=", 1) for item in args.tree)]
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    boot_index = 0
    for launch_index, (launch, model, config, fraction) in enumerate(LAUNCHES):
        order = arms if launch_index % 2 == 0 else list(reversed(arms))
        for arm, tree in order:
            log = args.out / f"{launch}_{arm}_serve.log"
            port = args.port + 200 * boot_index
            boot_index += 1
            url = f"http://127.0.0.1:{port}"
            print(f"{time.strftime('%T')} boot {launch} {arm}", flush=True)
            server = boot(tree, args.card, port, log, model, config, fraction)
            results = {}
            try:
                wait_healthy(url, server)
                for kind in ("stream", "whole"):
                    results[kind] = run_request(url, body(model, kind, tree), kind)
            except (RuntimeError, TimeoutError) as error:
                results["error"] = {"outcome": "error", "error": str(error)}
            finally:
                stop(server)
            entry = {
                **read_log(log.read_text(errors="replace")),
                "requests": {
                    kind: {key: result.get(key) for key in ("outcome", "samples", "error")}
                    for kind, result in results.items()
                },
            }
            report.setdefault(launch, {})[arm] = entry
            print(f"{time.strftime('%T')} {launch} {arm} {json.dumps(entry)}", flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{time.strftime('%T')} cascade_memory_check done", flush=True)


if __name__ == "__main__":
    main()
