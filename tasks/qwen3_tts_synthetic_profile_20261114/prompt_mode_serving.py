"""The speech API's non_streaming_mode on a real server, for every value a client can send.

One server per model on one card. For each value (absent, true, false), seeded requests one at a
time, streamed and whole, over a few inputs. A streamed request must arrive in more than one chunk
with its first byte inside the first half of the request. Per model, the audio hashes show which
values build the same prompt: a value equal to the task's default must repeat the absent bytes, and
the two values must differ (seeded single requests repeat bytes on this model).

usage: python3 prompt_mode_serving.py --tree TREE --card N --out DIR [--models base customvoice voicedesign]
"""

import argparse
import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from seedtts_pair import CARD_CPUS
from slot_wait_serving import INPUTS, REFERENCE_TEXT, REFERENCE_WAV, wait_healthy

MODELS = {
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voicedesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
# the prompt mode each task builds when the request does not say
DEFAULT_MODE = {"base": False, "customvoice": True, "voicedesign": True}
VALUES = (None, True, False)
SEED = 1234
REQUEST_TIMEOUT_S = 300


def request_body(model: str, tree: Path, text: str, value: bool | None, stream: bool) -> dict:
    body = {
        "model": MODELS[model], "input": text, "language": "English", "response_format": "pcm",
        "stream": stream, "seed": SEED,
    }
    if model == "base":
        clip = base64.b64encode((tree / REFERENCE_WAV).read_bytes()).decode()
        body["references"] = [{"audio_path": f"data:audio/wav;base64,{clip}", "text": REFERENCE_TEXT}]
    elif model == "customvoice":
        body["voice"] = "Ryan"
    else:
        body["instructions"] = "A calm adult male voice with a slow, clear delivery."
    if value is not None:
        body["non_streaming_mode"] = value
    else:
        pass
    return body


def run_request(url: str, body: dict) -> dict:
    start = time.monotonic()
    first_byte_s = None
    chunks = 0
    audio = bytearray()
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        with client.stream("POST", f"{url}/v1/audio/speech", json=body) as response:
            if response.status_code != 200:
                return {"status": response.status_code, "error": response.read().decode(errors="replace")[:300]}
            else:
                pass
            for piece in response.iter_bytes():
                if first_byte_s is None:
                    first_byte_s = time.monotonic() - start
                else:
                    pass
                chunks += 1
                audio.extend(piece)
    return {
        "status": 200, "chunks": chunks, "first_byte_s": first_byte_s, "total_s": time.monotonic() - start,
        "samples": len(audio) // 2, "sha256": hashlib.sha256(bytes(audio)).hexdigest()[:16],
    }


def serve_model(tree: Path, model: str, card: int, port: int, out: Path) -> bool:
    url = f"http://127.0.0.1:{port}"
    command = [
        "taskset", "-c", CARD_CPUS[card], sys.executable, "-m", "sglang_omni.cli", "serve",
        "--model-path", MODELS[model], "--host", "127.0.0.1", "--port", str(port),
        "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card),
    ]
    with (out / f"{model}_serve.log").open("w") as log:
        server = subprocess.Popen(
            command, cwd=tree, env={**os.environ, "PYTHONPATH": str(tree), "SGLANG_OMNI_STRICT_PORT": "1"},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    is_ok = True
    try:
        wait_healthy(url, server)
        hashes: dict[tuple[bool | None, bool, int], str] = {}
        for value in VALUES:
            for stream in (True, False):
                for index, text in enumerate(INPUTS):
                    result = run_request(url, request_body(model, tree, text, value, stream))
                    record = {"model": model, "value": value, "stream": stream, "input": index, **result}
                    is_request_ok = result["status"] == 200 and result["samples"] > 0 and (
                        not stream or (result["chunks"] > 1 and result["first_byte_s"] < result["total_s"] / 2)
                    )
                    record["ok"] = is_request_ok
                    is_ok = is_ok and is_request_ok
                    if result["status"] == 200:
                        hashes[(value, stream, index)] = result["sha256"]
                    else:
                        pass
                    with (out / "requests.jsonl").open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(
                        f"{time.strftime('%T')} {model} value={value} stream={stream} input={index} ok={is_request_ok} "
                        f"chunks={result.get('chunks')} first_byte_s={result.get('first_byte_s')} "
                        f"total_s={result.get('total_s')} sha={result.get('sha256')}",
                        flush=True,
                    )
        default = DEFAULT_MODE[model]
        for stream in (True, False):
            same_as_absent = [hashes.get((default, stream, i)) == hashes.get((None, stream, i)) for i in range(len(INPUTS))]
            modes_differ = [hashes.get((True, stream, i)) != hashes.get((False, stream, i)) for i in range(len(INPUTS))]
            print(
                f"{time.strftime('%T')} {model} stream={stream} default value repeats absent bytes {same_as_absent}, "
                f"true and false differ {modes_differ}",
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
    parser.add_argument("--port", type=int, default=28000)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {
        model: serve_model(args.tree, model, args.card, args.port + 200 * index, args.out)
        for index, model in enumerate(args.models)
    }
    print(f"{time.strftime('%T')} prompt_mode_serving {'PASS' if all(results.values()) else 'FAIL'} {results}", flush=True)


if __name__ == "__main__":
    main()
