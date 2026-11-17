"""Codec slot waiting on a real server with few slots, in the ways clients use it.

One server per model (1.7B Base with a reference clip, 1.7B CustomVoice with Ryan) on one card, with
--vocoder.factory.codec_state_slots SLOTS so streams wait for a slot at modest concurrency. Each round
sends at once: full streaming requests, whole (non-streaming) requests, streams the client abandons
before their first audio (while they wait for a slot), and streams the client drops after the first
chunk. After the round, one streaming request on the idle server (no later release can wake a stranded
waiter then). A request that should finish must return audio within REQUEST_TIMEOUT_S. After every round
the vocoder's own codec stats must show no active slot and no waiting stream.

usage: python3 slot_wait_serving.py --tree TREE --card N --out DIR [--slots 2] [--rounds 4]
"""

import argparse
import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from seedtts_pair import CARD_CPUS

MODELS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}
INPUTS = (
    "The weather cleared just before we reached the station.",
    "We took the long way home, stopping at the bakery to buy fresh bread and a small cake for later.",
    "Please confirm the meeting time with the rest of the team before noon tomorrow.",
)
REFERENCE_TEXT = "It was the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
REFERENCE_WAV = "docs/_static/audio/ref_voice.wav"
REQUEST_TIMEOUT_S = 120
CANCEL_BEFORE_AUDIO_S = 0.3
HEALTH_TIMEOUT_S = 900
# (kind, count) per round
ROUND_MIX = (("stream", 14), ("whole", 2), ("cancel_before_audio", 3), ("drop_after_first_chunk", 3))


def request_body(model: str, kind: str, index: int, tree: Path) -> dict:
    body = {
        "model": MODELS[model],
        "input": INPUTS[index % len(INPUTS)],
        "language": "English",
        "response_format": "pcm",
        "stream": kind != "whole",
    }
    if model == "base":
        clip = base64.b64encode((tree / REFERENCE_WAV).read_bytes()).decode()
        body["references"] = [{"audio_path": f"data:audio/wav;base64,{clip}", "text": REFERENCE_TEXT}]
    else:
        body["voice"] = "Ryan"
    return body


def run_request(url: str, body: dict, kind: str) -> dict:
    start = time.monotonic()
    first_byte_s = None
    received = 0
    outcome = "complete"
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        with client.stream("POST", f"{url}/v1/audio/speech", json=body) as response:
            if response.status_code != 200:
                return {"kind": kind, "outcome": f"http_{response.status_code}", "samples": 0}
            else:
                pass
            if kind == "cancel_before_audio":
                closer = threading.Timer(CANCEL_BEFORE_AUDIO_S, response.close)
                closer.start()
            else:
                closer = None
            try:
                for audio in response.iter_bytes():
                    if first_byte_s is None:
                        first_byte_s = time.monotonic() - start
                    else:
                        pass
                    received += len(audio)
                    if kind == "drop_after_first_chunk":
                        outcome = "dropped"
                        break
                    else:
                        pass
            except (httpx.StreamClosed, httpx.ReadError, httpx.RemoteProtocolError):
                outcome = "cancelled"
            finally:
                if closer is not None:
                    closer.cancel()
                else:
                    pass
    if kind == "cancel_before_audio" and first_byte_s is None:
        outcome = "cancelled"
    else:
        pass
    return {
        "kind": kind,
        "outcome": outcome,
        "samples": received // 2,
        "first_byte_s": first_byte_s,
        "total_s": time.monotonic() - start,
    }


def codec_state(url: str) -> dict:
    """The vocoder's codec stats from /model_info, wherever the stage reports them."""
    found: list[dict] = []

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if "codec_state" in value and isinstance(value["codec_state"], dict):
                found.append(value["codec_state"])
            else:
                pass
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            pass

    walk(httpx.get(f"{url}/model_info", timeout=30).json())
    return found[0] if found else {}


def wait_healthy(url: str, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited with {server.returncode}")
        else:
            pass
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return
            else:
                pass
        except httpx.RequestError:
            pass
        time.sleep(2)
    raise TimeoutError("server never became healthy")


def serve_rounds(tree: Path, model: str, card: int, port: int, slots: int, rounds: int, out: Path) -> bool:
    url = f"http://127.0.0.1:{port}"
    command = [
        "taskset", "-c", CARD_CPUS[card], sys.executable, "-m", "sglang_omni.cli", "serve",
        "--model-path", MODELS[model], "--host", "127.0.0.1", "--port", str(port),
        "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card),
        "--tts_engine.engine.max_running_requests", "16",
        "--vocoder.factory.codec_state_slots", str(slots),
    ]
    out.mkdir(parents=True, exist_ok=True)
    is_ok = True
    with (out / f"{model}_serve.log").open("w") as log:
        server = subprocess.Popen(
            command, cwd=tree, env={**os.environ, "PYTHONPATH": str(tree), "SGLANG_OMNI_STRICT_PORT": "1"},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        wait_healthy(url, server)
        kinds = [kind for kind, count in ROUND_MIX for _ in range(count)]
        for round_index in range(rounds):
            with ThreadPoolExecutor(max_workers=len(kinds)) as pool:
                results = list(pool.map(
                    lambda pair: run_request(url, request_body(model, pair[1], pair[0], tree), pair[1]),
                    enumerate(kinds),
                ))
            followup = run_request(url, request_body(model, "stream", round_index, tree), "stream")
            time.sleep(1)
            stats = codec_state(url)
            should_finish = [r for r in results if r["kind"] in ("stream", "whole")] + [followup]
            failures = [r for r in should_finish if r["outcome"] != "complete" or r["samples"] == 0]
            is_round_ok = not failures and stats.get("active_slots") == 0
            is_ok = is_ok and is_round_ok
            record = {
                "model": model, "round": round_index, "ok": is_round_ok, "failures": failures,
                "outcomes": {kind: [r["outcome"] for r in results if r["kind"] == kind] for kind, _ in ROUND_MIX},
                "followup": followup, "codec_state": stats,
                "max_first_byte_s": max((r["first_byte_s"] or 0) for r in should_finish),
            }
            with (out / "rounds.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"{time.strftime('%T')} {model} round {round_index} ok={is_round_ok} "
                f"slot_waits={stats.get('slot_waits')} active={stats.get('active_slots')} "
                f"fallbacks={stats.get('left_context_fallbacks')} failures={len(failures)}",
                flush=True,
            )
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        else:
            pass
    return is_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--port", type=int, default=27000)
    args = parser.parse_args()
    results = {
        model: serve_rounds(args.tree, model, args.card, args.port + 200 * index, args.slots, args.rounds, args.out)
        for index, model in enumerate(MODELS)
    }
    print(f"{time.strftime('%T')} slot_wait_serving {'PASS' if all(results.values()) else 'FAIL'} {results}", flush=True)


if __name__ == "__main__":
    main()
